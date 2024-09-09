import os
import sys; sys.path.insert(0, os.path.abspath("../"))
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import os, os.path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import sys;
import torchvision.transforms.functional as T
import random
import copy
import random
from lightning.fabric import seed_everything

notebooks_path = Path.cwd()
repo_path = notebooks_path.parent
data_path = repo_path / 'data' / 'HematomaTruetaBasal' 
sys.path.insert(0, repo_path)
seed = 42 
seed_everything(seed)


################### Helper functions. ###################
def get_slices_from_subset(dataset_subset, return_type=None): # only for the image slices not mask or label. 
    """ 3d to 2d. The function takes a datadataset_subset and returns a list of slices of the given set pateints.
    The slices are in 2d image format (512,512) and they are ansembled in a list.

    Args:
        dataset_subset (data dataset_subset): data dataset_subset of the set of patients.

    Returns:
        list: image slices and the corresponding labels of the given set of patients.
              or image masks and the corresponding labels of the given set of patients.
    """
    # Brings all the slices of the test set patients in to together in 2d image format (512,512)
    image_slices = []
    labels = [] # labels per each slice
    masks = [] # masks per each slice
    patient_ids = []
    patient_slice_numbers = []
    p_id = dataset_subset.patient_id
    count = 0

    for batch in dataset_subset:
        image, mask, label = batch
        patient_id = p_id[count]
        for idx in range(image.shape[0]): #image.size(1) is the number of slices in the image.
            slice = image[idx, :, :] # concatinating the slices of the same patient.
            mask_slice = mask[idx, :, :] # concatinating the masks of the same patient.
            patient_ids.append(patient_id)
            image_slices.append(slice)
            labels.append(label)  # for the same patient's slices, the label is the same.
            masks.append(mask_slice) 
            patient_slice_numbers.append(idx) # for each slice of the patient, the slice number is saved.
        
        count += 1

    # convert to tensor
    image_slices = torch.stack([torch.from_numpy(arr) for arr in image_slices])
    masks = np.array(masks, dtype=np.float32)
    masks = torch.stack([torch.from_numpy(arr) for arr in masks])
    # Convert patient_ids to a supported type, such as int or float
    patient_ids = np.array(patient_ids, dtype=np.int64)
    # Convert the modified patient_ids and patient_slice_numbers to PyTorch Tensors
    patient_ids = torch.from_numpy(patient_ids)
    patient_slice_numbers = torch.from_numpy(np.array(patient_slice_numbers))
    labels = torch.from_numpy(np.array(labels).astype(np.int16))

    if return_type == 'image':
        return image_slices, labels, patient_ids, patient_slice_numbers
    elif return_type == 'mask':
        return masks, labels, patient_ids, patient_slice_numbers

def find_normalization_parameters(image):
    """
    It takes an image and returns the mean and standard deviation of the image.
    
    :param image: the image to be normalized
    :return: The mean and standard deviation of the image. For 3d images.
    """
    norm_img = copy.deepcopy(image)
    norm_parms = (np.nanmin(norm_img, axis=(-3, -2, -1), keepdims=True), 
                   np.nanmax(norm_img, axis=(-3, -2, -1), keepdims=True))

    return norm_parms 

def normalize_image(image_patch, parameters):
    """
    The function takes an image patch and a list of parameters as input, and returns the normalized
    image patch.
    
    :param image_patch: the image patch that we want to normalize
    :param parameters: [mean, std]
    :return: The normalized image patch.
    """
    if len(image_patch.shape) == 3: # 2D case
        parameters = (np.squeeze(parameters[0], axis=-1),
                      np.squeeze(parameters[1], axis=-1))

    minmax = (image_patch-parameters[0]) / (parameters[1]-parameters[0]) # [0,1]
    return minmax

################### Datasets and DataModules. ###################

class PredictionDataset(Dataset):
    """
    Dataset for prediction of the model.
    Args:
        md_path (Path): Path to the metadata file.
        
        Returns:
            image (torch.tensor): Image of the patient.
            mask (torch.tensor): Mask of the patient.
            label (torch.tensor): Label of the patient.
    """
    def __init__(self, md_path: Path = None):
        
        self.md_path = md_path 
        self.md_df = pd.read_csv(self.md_path)
        self.labels = self.md_df['label'].values 
        self.patient_id= self.md_df['patient_id'].values 
        self.mask_paths = self.md_df['mask_path'].values
        self.image_paths = self.md_df['ct_ss_path'].values
        self.index = self.md_df['index'].values
        
    def __len__(self):
        return len(self.md_df)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)
        image = np.array(image)

        mask_path = self.mask_paths[idx]
        mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask)
        mask = np.array(mask)

        label = self.labels[idx]
        label = np.array(label)

        return image, mask, label

class DataAugmentation(object):
    """
    For each slice in the 3d image, it applies the transformations.
    """
    def __init__(self, apply_hflip, apply_affine, apply_gaussian_blur, degree, translate, scale, shear, hflip_p, affine_p ):
        self.apply_hflip = apply_hflip
        self.apply_affine = apply_affine
        self.apply_gaussian_blur = apply_gaussian_blur
        self.degree = degree
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.percentage_hflip = hflip_p
        self.percentage_affine = affine_p

    def __call__(self, img, mask):
        if self.apply_hflip:
            if random.random() < self.percentage_hflip:
                img = T.hflip(img)
                mask = T.hflip(mask)

        if self.apply_gaussian_blur:
            if random.random() < 0.30:
                img = T.gaussian_blur(img, kernel_size=3, sigma=(0.1, 0.9))

        if self.apply_affine:
            if random.random() <  self.percentage_affine:
                img = T.affine(img, angle=self.degree, translate=self.translate, scale=self.scale, shear=self.shear)
                mask = T.affine(mask, angle=self.degree, translate=self.translate, scale=self.scale, shear=self.shear)
                
        return img, mask
    

class PredictionDataset2D(Dataset):
    def __init__(self, slices, masks, image_size, patient_ids,  slice_number, test_type, apply_hflip = False, apply_affine = False,
                 apply_gaussian_blur = False, affine_degree = 10, affine_translate = 10, affine_scale = 1.2, affine_shear = 0, labels=None, 
                 slices_fu=None, masks_fu= None, transform=False, three_channel_mask=False, hflip_p=0.5, affine_p=0.5):
        
        self.slices = slices
        self.test_type = test_type
        self.labels = labels
        self.masks = masks
        self.patient_ids = patient_ids
        self.slice_number = slice_number
        self.transform = transform
        self.three_channel_mask = three_channel_mask
        self.slices_fu = slices_fu
        self.masks_fu = masks_fu
        self.image_size = image_size
        self.apply_hflip = apply_hflip
        self.apply_affine = apply_affine
        self.apply_gaussian_blur = apply_gaussian_blur
        self.affine_degree = affine_degree
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale
        self.affine_shear = affine_shear
        self.hflip_p = hflip_p
        self.affine_p = affine_p

    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        id_value = self.patient_ids[idx].item()
        slice_number = self.slice_number[idx].item()
        combined_number = str(id_value) + str(slice_number)
        updated_slice = self.slices[idx].unsqueeze(0)
        maskbasal = self.masks[idx].unsqueeze(0)

        if self.transform:
            if self.test_type == 't1' or self.test_type == 't3': # Original data (no DA, no synthesis) or Original data + synthesis (no DA)
                image = updated_slice
                mask = maskbasal

            elif self.test_type == 't2' or self.test_type == 't5': # Original data + DA (no synthesis) or Original data + DA + synthesis
                image, mask = DataAugmentation(apply_hflip= self.apply_hflip, apply_affine= self.apply_affine,
                                               apply_gaussian_blur= self.apply_gaussian_blur, degree = self.affine_degree,
                                               translate = (self.affine_translate, self.affine_translate),
                                               scale = self.affine_scale, shear = self.affine_shear, hflip_p=self.hflip_p, 
                                               affine_p=self.affine_p)(img=updated_slice, mask=maskbasal, image_index= combined_number)
            else: 
                print("Please define test type t1, t2, t3, t5")
            
            # Normalize each channel separately.
            norm_params1 = find_normalization_parameters(image)
            image = normalize_image(image, norm_params1)
            mask2 = mask.cpu().numpy()
            mask2 = np.squeeze(mask2)
            mask2 = mask2.astype(np.uint8)
            mask2 = torch.from_numpy(mask2)
            changed_dim = image.expand(2,*image.shape[1:])
            mask2.unsqueeze_(0)
            three_channel = torch.cat((changed_dim, mask2), 0)
            return three_channel, self.labels[idx], self.patient_ids[idx], self.slice_number[idx]

        else: # validation and test set. 
            norm_params1 = find_normalization_parameters(updated_slice)
            updated_slice = normalize_image(updated_slice, norm_params1)
            mask = maskbasal.cpu().numpy()
            mask = np.squeeze(mask)
            mask = mask.astype(np.uint8)
            mask = torch.from_numpy(mask)
            mask.unsqueeze_(0)
            changed_dim = updated_slice.expand(2,*updated_slice.shape[1:])
            three_channel = torch.cat((changed_dim, mask), 0)
            return three_channel, self.labels[idx], self.patient_ids[idx], self.slice_number[idx]

class HEPredDataModule(pl.LightningDataModule):

    """
    The data module. It takes the metadata file and the mask file and returns the dataloader for the training, validation and test set.
    Args:

        md_path (Path): Path to the metadata file.
        test_type (str): The type of the test. It can be 't1', 't2', 't3', 't4', 't5'.
        image_size (int): The size of the image.
        basal_path (Path): Path to the basal images.
        batch_size (int): The batch size.
        split_indexes (tuple): The indexes of the train, validation and test set.
        num_workers (int): The number of workers.
        filter_slices (bool): If True, the dataloader will return the 2D that contains the lesion.
        threshold (int): The threshold used to filter the slices that are segmented in the mask. (# of pixels of bleeding)
        apply_hflip (bool): If True, the dataloader will apply horizontal flip.
        apply_affine (bool): If True, the dataloader will apply affine transformation.
        affine_degree (int): The degree of the affine transformation.
        affine_translate (int): The translation of the affine transformation.
        affine_scale (int): The scale of the affine transformation.
        affine_shear (int): The shear of the affine transformation.
        apply_gaussian_blur (bool): If True, the dataloader will apply gaussian blur.
        hflip_p (float): The probability of the horizontal flip.
        affine_p (float): The probability of the affine transformation.
    """

    def __init__(self, split_indexes, batch_size, image_size, test_type, apply_hflip,
                 apply_affine, affine_degree, affine_translate , affine_scale, affine_shear,
                 apply_gaussian_blur, hflip_p, affine_p,
                 filter_slices: bool = False,
                 num_workers: int = 8, 
                 md_path: Path = None,
                 basal_path: Path = None,
                 threshold: int = 100,
                 ):
        
        super().__init__()
        self.md_path = md_path
        self.test_type = test_type
        self.image_size = image_size
        self.basal_path = basal_path
        self.batch_size = batch_size
        self.split_indexes = split_indexes
        self.num_workers = num_workers
        self.filter_slices = filter_slices 
        self.threshold = threshold
        self.apply_hflip = apply_hflip
        self.apply_affine = apply_affine
        self.affine_degree = affine_degree
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale
        self.affine_shear = affine_shear
        self.apply_gaussian_blur = apply_gaussian_blur
        self.hflip_p = hflip_p
        self.affine_p = affine_p

    def setup(self, stage=None):
        self.dataset = PredictionDataset(md_path=self.md_path)
        X_train, X_val, X_test, _, _, _ = self.split_indexes
        self.train = torch.utils.data.Subset(self.dataset, indices=X_train)
        self.val = torch.utils.data.Subset(self.dataset, indices= X_val)
        self.test = torch.utils.data.Subset(self.dataset, indices= X_test)
        self.train.patient_id = self.dataset.patient_id[X_train] 
        self.val.patient_id = self.dataset.patient_id[X_val] 
        self.test.patient_id = self.dataset.patient_id[X_test]
        self.train.labels = self.dataset.labels[X_train]
        self.val.labels = self.dataset.labels[X_val]
        self.test.labels = self.dataset.labels[X_test]

    def filter_segmented_slices(self, slices, mask, patient_ids,  slice_numbers, labels=None, threshold: int = 100):
        """
        This function filters the slices that are containing lesion in the mask. 
        """
        
        # Find the slices where the lesion is available
        lesion_slices = []
        for i in range(len(slices)):
            if mask[i].sum() > threshold: # take if there are at least 2 pixels with value 1 
                lesion_slices.append(i)

        # Select only the slices where the lesion is available
        filtered_slices = slices[lesion_slices]
        filtered_masks = mask[lesion_slices]
        filtered_labels = labels[lesion_slices]
        filtered_ids = patient_ids[lesion_slices]
        filtered_slice_numbers = slice_numbers[lesion_slices]

        # convert the slices to tensor
        filtered_slices= filtered_slices.clone().detach()
        filtered_masks= filtered_masks.clone().detach()
        filtered_labels= filtered_labels.clone().detach()
        filtered_ids= filtered_ids.clone().detach()
        filtered_slice_numbers= filtered_slice_numbers.clone().detach()
        
        return filtered_slices, filtered_masks, filtered_labels, filtered_ids, filtered_slice_numbers

    def train_dataloader(self):

        # Return only the slices where the lesion is available.
        if self.filter_slices==True: 
            slices, labels, patient_ids, patient_slice_numbers = get_slices_from_subset(self.train, return_type="image")
            mask_slices, labels, patient_ids, patient_slice_numbers = get_slices_from_subset(self.train, return_type='mask')
            filtered_slices, filtered_masks, \
            filtered_labels, filtered_ids, filtered_slice_numbers = self.filter_segmented_slices(slices=slices, mask=mask_slices, 
                                                                                                patient_ids=patient_ids, labels=labels,
                                                                                                slice_numbers=patient_slice_numbers,
                                                                                                threshold=self.threshold)
            self.dataset_2d = PredictionDataset2D(slices=filtered_slices, masks=filtered_masks, patient_ids=filtered_ids, labels=filtered_labels,
            slice_number=filtered_slice_numbers, transform=True, image_size=self.image_size, test_type = self.test_type, apply_hflip=self.apply_hflip,
            apply_affine=self.apply_affine, apply_gaussian_blur=self.apply_gaussian_blur, affine_degree=self.affine_degree, affine_translate=self.affine_translate,
            affine_scale=self.affine_scale, affine_shear=self.affine_shear, hflip_p=self.hflip_p, affine_p = self.affine_p)
            print("The number of slices in the train set: ", len(self.dataset_2d))
            return DataLoader(self.dataset_2d, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True,drop_last=True)
        
        # Return all the slices (the whole image volume).
        else: 
            self.dataset_2d = PredictionDataset2D(slices=filtered_slices, labels=filtered_labels, patient_ids=filtered_ids, transform=True)
            return DataLoader(self.dataset_2d, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True)

    def val_dataloader(self):        
        if self.filter_slices==True:
            slices, labels, patient_ids, patient_slice_numbers = get_slices_from_subset(self.val, return_type="image")
            mask_slices, labels, patient_ids, patient_slice_numbers = get_slices_from_subset(self.val, return_type='mask')
            filtered_slices, filtered_masks, \
            filtered_labels, filtered_ids, filtered_slice_numbers = self.filter_segmented_slices(slices=slices, mask=mask_slices, 
                                                                                                patient_ids=patient_ids,
                                                                                                labels=labels,
                                                                                                slice_numbers=patient_slice_numbers,
                                                                                                threshold=self.threshold)
            
            self.dataset_2d  = PredictionDataset2D(slices=filtered_slices, masks=filtered_masks, patient_ids=filtered_ids, 
                                                    labels=filtered_labels, slice_number=filtered_slice_numbers, transform=False,
                                                    image_size=self.image_size, lesion=self.lesion, test_type = self.test_type)
            print("The number of slices in validation set:", len(filtered_labels))
            return DataLoader(self.dataset_2d, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False,drop_last=True)
            
        else: 
            self.dataset_2d = PredictionDataset2D(slices, labels, transform=False)
            return DataLoader(self.dataset_2d, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True)

    
    def test_dataloader(self):
        if self.filter_slices==True:
            slices, labels, patient_ids, patient_slice_numbers = get_slices_from_subset(self.test, return_type="image")
            mask_slices, labels, patient_ids, patient_slice_numbers = get_slices_from_subset(self.test, return_type='mask')
            filtered_slices, filtered_masks, \
            filtered_labels, filtered_ids, filtered_slice_numbers = self.filter_segmented_slices(slices=slices, mask=mask_slices, 
                                                                                                patient_ids=patient_ids,
                                                                                                labels=labels,
                                                                                                slice_numbers=patient_slice_numbers,
                                                                                                threshold=self.threshold)
            self.dataset_2d  = PredictionDataset2D(slices=filtered_slices, masks=filtered_masks, patient_ids=filtered_ids, 
                                                   labels=filtered_labels, slice_number=filtered_slice_numbers, transform=False,
                                                   image_size=self.image_size, lesion=self.lesion, test_type = self.test_type)
            
            print("The number of slices in the test set: ", len(self.dataset_2d))
            return DataLoader(self.dataset_2d, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False,drop_last=True)
            
        else:
            self.dataset_2d = PredictionDataset2D(slices, labels, transform=False)
            return DataLoader(self.dataset_2d, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False)

        