import logging
import torch
import os
import numpy as np
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy
from torchvision import models, transforms
import pandas as pd
import pytorch_lightning as pl
import json
from monai.networks.nets import SEResNet50
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torchmetrics
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import timm
from torch.autograd import Function
from monai.networks.blocks import SEBlock
from torch.optim.lr_scheduler import ReduceLROnPlateau


repo_path = os.getcwd()

################### Helper functions. ###################
class SELayer(nn.Module):
        def __init__(self, channel, reduction=16):
            super(SELayer, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)

def focal_loss(logits, targets, alpha=0.7, gamma=1, device = 0):
    # bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float().to(f"cuda:{device}"), reduction='none')
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return torch.mean(focal_loss)

def compute_sensitivity_specificity(preds, labels):
    true_positives = (preds == 1) & (labels == 1)
    true_negatives = (preds == 0) & (labels == 0)
    false_positives = (preds == 1) & (labels == 0)
    false_negatives = (preds == 0) & (labels == 1)
    sensitivity = true_positives.sum() / (true_positives.sum() + false_negatives.sum())
    specificity = true_negatives.sum() / (true_negatives.sum() + false_positives.sum())
    return sensitivity, specificity

#####################################

class ImageClassifier(pl.LightningModule):
    def __init__(self, back, name_experiment, model='eff_s', num_classes=1, learning_rate=1e-5, weight_decay=1e-3, optimizer='adam', momentum=0.9, 
                 lr_scheduler=None, step_size=15, gamma=0.1, class_weights=[1,1], pw_based="mean", test_pw_based ="mean", threshold=0.5, num_features=0, 
                 loss="focal", fold = 0, task = "2d_pred", save_predictions=False):
        
        super().__init__()
        self.name_experiment = name_experiment 
        self.fold = fold
        self.back = back
        self.task = task 
        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.momentum = momentum
        self.lr_scheduler = lr_scheduler
        self.num_classes = num_classes
        self.pw_based = pw_based
        self.test_pw_based = test_pw_based
        self.threshold = threshold
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.best_roc_auc = 0
        self.num_features = num_features
        self.save_predictions = save_predictions
        self.gamma = gamma
        self.step_size= step_size

        if self.model == 'resnet34':
            backbone = torchvision.models.resnet34(weights=None)
            num_features = backbone.fc.in_features
            backbone.fc = nn.Linear(num_features, 1)
            self.backbone = backbone

        if self.model=='eff_s':
            eff = torchvision.models.efficientnet_b0(weights="DEFAULT")
            num_features = eff.classifier[1].in_features
            eff.classifier = nn.Linear(num_features, 1)
            self.backbone = eff

        if self.model == "densenet121":
            densenet = torchvision.models.densenet121(pretrained=True)
            num_features = densenet.classifier.in_features
            densenet.classifier = nn.Linear(num_features, 1)
            self.backbone = densenet

        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_accuracy = torchmetrics.Accuracy(task='binary')

        self.train_accuracy_pw = torchmetrics.Accuracy(task='binary')
        self.val_accuracy_pw = torchmetrics.Accuracy(task='binary')
        self.test_accuracy_pw = torchmetrics.Accuracy(task='binary')

        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')

        self.train_f1_pw = torchmetrics.F1Score(task='binary')
        self.val_f1_pw = torchmetrics.F1Score(task='binary')
        self.test_f1_pw = torchmetrics.F1Score(task='binary')

        self.train_precision_calculate = torchmetrics.Precision(task='binary')
        self.val_precision_calculate = torchmetrics.Precision(task='binary')
        self.test_precision_calculate = torchmetrics.Precision(task='binary')

        self.train_precision_calculate_pw = torchmetrics.Precision(task='binary')
        self.val_precision_calculate_pw = torchmetrics.Precision(task='binary')
        self.test_precision_calculate_pw = torchmetrics.Precision(task='binary')
        
    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch):

        x, y, patient_ids, patient_slice_nums = batch
        logits = self(x)

        logits = logits.squeeze()
        loss = focal_loss(logits, y)
        probs = nn.Sigmoid()(logits)
        preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))

        acc = self.train_accuracy(preds, y)
        prec = self.train_precision_calculate(preds, y)
        f1 = self.train_f1(preds, y)

        current_learningrate = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_learningrate, prog_bar=True, sync_dist=True)

        return {'loss': loss, 
                'probs': probs, 
                'patient_ids': patient_ids, 
                'patient_slice_nums': patient_slice_nums,
                'labels': y}

    def validation_step(self, batch):
        x, y, patient_ids, patient_slice_nums = batch
        logits = self(x)
        logits = logits.squeeze()
        loss = focal_loss(logits, y)
        probs = nn.Sigmoid()(logits)
        preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))

        acc = self.val_accuracy(preds, y)
        prec = self.val_precision_calculate(preds, y)
        f1 = self.val_f1(preds, y)
        
        return {'loss': loss, 
                'probs': probs, 
                'patient_ids': patient_ids, 
                'patient_slice_nums': patient_slice_nums,
                'labels': y}
    
    def test_step(self, batch):
        x, y, patient_id, patient_slice_nums = batch
        logits = self(x)

        logits = logits.squeeze()
        loss = focal_loss(logits, y)
        probs = nn.Sigmoid()(logits)
        preds = torch.where(probs > 0.5, torch.tensor(1), torch.tensor(0))

        acc = self.test_accuracy(preds, y)
        prec = self.test_precision_calculate(preds, y)
        f1 = self.test_f1(preds, y)

        self.log('test_loss', loss,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, 
                'probs': probs, 
                'labels': y, 
                'patient_id': patient_id,
                'patient_slice_nums': patient_slice_nums}

    def training_epoch_end(self, outputs):
        self.patient_probs = {} 
        self.patient_labels = {} 
        self.patient_slice_nums = {}

        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_ids'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]
            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_patient_slice_nums = torch.cat(patient_slice_nums, dim=0)
            concat_single_probs = concat_probs

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            for i in range(len(concat_patient_slice_nums)):
                patient_id = concat_ids[i].item()
                patient_slice_num = concat_patient_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [patient_slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(patient_slice_num)

            # Calculate the average probability for each patient.
            for patient_id in self.patient_probs:
                if self.pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for pw_based")

            # Assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            self.patient_probs_dict = {k: [v, self.patient_labels[k], self.patient_slice_nums[k]] for k, v in self.patient_probs.items()}

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            # Calculate the epoch metrics for patient level
            acc_pw = self.train_accuracy_pw(y_pred, y_true)
            acc_pw= self.train_accuracy_pw.compute()
            prec_pw = self.train_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.train_precision_calculate_pw.compute()
            f1_pw = self.train_f1_pw(y_pred, y_true)
            f1_pw = self.train_f1_pw.compute()

            self.log("train_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs]
            stacked = torch.stack(losses)
            avg_loss = torch.mean(stacked)
            self.log('train_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.train_accuracy.compute()
        prec = self.train_precision_calculate.compute()
        f1 = self.train_f1.compute()

        self.log('train_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        self.patient_probs = {} 
        self.patient_labels = {} 
        self.patient_slice_nums = {}

        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_ids'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]
            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_slice_nums = torch.cat(patient_slice_nums, dim=0)
            concat_single_probs = concat_probs

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                slice_num = concat_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(slice_num)

            # Calculate the average probability for each patient.
            for patient_id in self.patient_probs:
                if self.pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for pw_based")

            # assign 1, if one of the slice labels is 1, otherwise 0
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities, labels and slice numbers
            self.patient_probs_dict = {k: [v, self.patient_labels[k], self.patient_slice_nums[k]] for k, v in self.patient_probs.items()}
            concat_labels = concat_labels.cpu()
            concat_single_probs = concat_single_probs.cpu()

            # for each individual slice calculate the metrics
            y_pred_iw = torch.tensor([1 if x > self.threshold  else 0 for x in concat_single_probs])
            y_true_iw = concat_labels

            sens_iw, spec_iw = compute_sensitivity_specificity(y_pred_iw, y_true_iw)
            fpr_iw, tpr_iw, _ = roc_curve(y_true_iw.cpu().numpy(), y_pred_iw.cpu().numpy())
            roc_auc_iw = auc(fpr_iw, tpr_iw)

            self.log('val_sens_iw', sens_iw, prog_bar=True, sync_dist=True) # validation sensitivity score
            self.log('val_spec_iw', spec_iw, prog_bar=True, sync_dist=True) # validation specificity score
            self.log('val_roc_auc_iw', roc_auc_iw, prog_bar=True, sync_dist=True) # validation auroc score

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold  else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            sens, spec = compute_sensitivity_specificity(y_pred, y_true)
            fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_pred.cpu().numpy())
            roc_auc = auc(fpr, tpr)
            self.log('val_roc_auc', roc_auc, prog_bar=True, sync_dist=True) # validation auroc score

            # Calculate the epoch metrics for patient level
            acc_pw = self.val_accuracy_pw(y_pred, y_true)
            acc_pw= self.val_accuracy_pw.compute()
            prec_pw = self.val_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.val_precision_calculate_pw.compute()
            f1_pw = self.val_f1_pw(y_pred, y_true)
            f1_pw = self.val_f1_pw.compute()
            
            self.log("val_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_sens_pw", sens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_spec_pw", spec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs] 
            stacked = torch.stack(losses)
            # takes the average of the losses in one epoch
            avg_loss = torch.mean(stacked)
            self.log('val_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        acc = self.val_accuracy.compute()
        prec = self.val_precision_calculate.compute()
        f1 = self.val_f1.compute()
        self.log('val_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def test_epoch_end(self, outputs):
        self.patient_probs = {} 
        self.patient_labels = {} 
        self.patient_slice_nums = {} 
        self.patient_probs_with_slices = {} 
    
        if outputs:
            probs = [x['probs'] for x in outputs]
            patient_ids = [x['patient_id'] for x in outputs]
            labels = [x['labels'] for x in outputs]
            patient_slice_nums = [x['patient_slice_nums'] for x in outputs]

            concat_probs = torch.cat(probs, dim=0)
            concat_ids = torch.cat(patient_ids, dim=0)
            concat_labels = torch.cat(labels, dim=0)
            concat_patient_slice_nums = torch.cat(patient_slice_nums, dim=0)
            concat_patient_slice_nums = concat_patient_slice_nums + 1 # add 1 to the slice numbers to make them start from 1 instead of 0
            concat_single_probs = concat_probs

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                prob = concat_single_probs[i].item()

                if patient_id not in self.patient_probs:
                    self.patient_probs[patient_id] = [prob]
                else:
                    self.patient_probs[patient_id].append(prob)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                labels = concat_labels[i].item()

                if patient_id not in self.patient_labels:
                    self.patient_labels[patient_id] = [labels]
                else:
                    self.patient_labels[patient_id].append(labels)

            for i in range(len(concat_ids)):
                patient_id = concat_ids[i].item()
                patient_slice_num = concat_patient_slice_nums[i].item()

                if patient_id not in self.patient_slice_nums:
                    self.patient_slice_nums[patient_id] = [patient_slice_num]
                else:
                    self.patient_slice_nums[patient_id].append(patient_slice_num)

            # create a nested dictionary with patient id as the first key and slice number as the second key and probability as the value
            self.patient_probs_with_slices = {k: {k2: v2 for k2, v2 in zip(self.patient_slice_nums[k], v)} for k, v in self.patient_probs.items()}
            # sort the nested dictionary by slice number
            self.patient_probs_with_slices = {k: dict(sorted(v.items())) for k, v in self.patient_probs_with_slices.items()} 
            print("self.patient_probs_with_slices after sorting: ", self.patient_probs_with_slices)
            self.patient_probs_with_slices_df = pd.DataFrame(columns=["patient_id", "avg_prob", "label"])
            for patient_id in self.patient_probs_with_slices:
               self.patient_probs_with_slices_df = self.patient_probs_with_slices_df.append({"patient_id": patient_id,
                                                                           "avg_prob": np.mean(list(self.patient_probs_with_slices[patient_id].values())),
                                                                          "label": self.patient_labels[patient_id]}, ignore_index=True)
            # create a folder to keep all the test results inside 
            experiments_folder_path =  repo_path + f"/plots/{self.name_experiment}"
            os.makedirs(experiments_folder_path, exist_ok=True)

            #save the nested dictionary as a txt file
            if self.save_predictions:
                with open(experiments_folder_path + "/" + 'patient_probs_with_slices.txt', 'w') as file:
                    file.write(json.dumps(self.patient_probs_with_slices))
                    self.patient_probs_with_slices_df.to_csv(experiments_folder_path + f"/{self.back}_{self.fold}_patient_probs_with_slices_df.csv")

            concat_labels = concat_labels.cpu()
            concat_single_probs = concat_single_probs.cpu()

            # calculate the average probability for each patient
            for patient_id in self.patient_probs:
                if self.test_pw_based == "mean":
                    self.patient_probs[patient_id] = np.mean(self.patient_probs[patient_id])
                elif self.test_pw_based == "max":
                    self.patient_probs[patient_id] = np.max(self.patient_probs[patient_id])
                elif self.test_pw_based == "middle_slice":
                    self.patient_probs[patient_id] = self.patient_probs[patient_id][len(self.patient_probs[patient_id]) // 2] # take the middle slice's probability as the patient's probability.
                else:
                    raise ValueError("Invalid value for test_pw_based")
                
            for patient_id in self.patient_labels:
                self.patient_labels[patient_id] = np.max(self.patient_labels[patient_id])

            # combine the probabilities and labels
            self.patient_probs_dict = {k: [v, self.patient_labels[k]] for k, v in self.patient_probs.items()}

            # calculate teh y_pred. if the probability is higher than 0.5, assign 1, otherwise 0
            y_pred = torch.tensor([1 if x[0] > self.threshold  else 0 for x in self.patient_probs_dict.values()])
            y_true = torch.tensor([x[1] for x in self.patient_probs_dict.values()])

            # for each individual slice calculate the metrics
            y_pred_iw = torch.tensor([1 if x > self.threshold  else 0 for x in concat_single_probs])
            y_true_iw = concat_labels

            sens_iw, spec_iw = compute_sensitivity_specificity(y_pred_iw, y_true_iw)
            fpr_iw, tpr_iw, _ = roc_curve(y_true_iw.cpu().numpy(), y_pred_iw.cpu().numpy())
            roc_auc_iw = auc(fpr_iw, tpr_iw)

            # Calculate the epoch metrics for patient level
            acc_pw = self.test_accuracy_pw(y_pred, y_true)
            acc_pw= self.test_accuracy_pw.compute()
            prec_pw = self.test_precision_calculate_pw(y_pred, y_true)
            prec_pw = self.test_precision_calculate_pw.compute()
            f1_pw = self.test_f1_pw(y_pred, y_true)
            f1_pw = self.test_f1_pw.compute()

            sens, spec = compute_sensitivity_specificity(y_pred, y_true)
            fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_pred.cpu().numpy())

            score = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())

            self.log('test_roc_auc_score_pw', score, prog_bar=True, sync_dist=True)
            self.log('test_roc_auc_score_iw', roc_auc_iw, prog_bar=True, sync_dist=True)
            self.log("test_acc_pw_epoch", acc_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_prec_pw_epoch", prec_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_f1_pw_epoch", f1_pw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_sens_pw_epoch", sens, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_sens_iw_epoch", sens_iw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_spec_iw_epoch", spec_iw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_spec_pw_epoch", spec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            losses = [x['loss'] for x in outputs]
            stacked = torch.stack(losses)
            avg_loss = torch.mean(stacked)
            self.log('test_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.save_predictions:
            if not os.path.exists(repo_path + f'/predictions'):
                os.makedirs(repo_path + f'/predictions')
            df = pd.DataFrame({'label': y_true.cpu().numpy(), 'pred': y_pred.cpu().numpy()})
            df.to_csv( repo_path + f'/predictions/{self.name_experiment}_{self.back}_{self.fold}_preds.csv', index=False)

        acc = self.test_accuracy.compute()
        prec = self.test_precision_calculate.compute()
        f1 = self.test_f1.compute()

        self.log('test_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_prec_epoch', prec, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        if self.optimizer=='adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer=='sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif self.optimizer == "adamw":
            self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError("Invalid optimizer choice")
        
        if self.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma, verbose=True)
        elif self.lr_scheduler == "reduceonplateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=1, verbose=True)
            return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler, "monitor": "val_loss_epoch"}
        else:
            self.lr_scheduler = None
            return self.optimizer
        
        return [self.optimizer], [self.lr_scheduler]
