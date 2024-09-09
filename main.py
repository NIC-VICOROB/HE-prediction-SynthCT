import os
from sklearn.utils.class_weight import compute_class_weight
import sys; sys.path.insert(0, os.path.abspath("../"))
from dataset import *
from model import *
import torch.utils.data as data
import argparse
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
import time

# Helper functions.
def add_sythetic_cases(org_set, sy_set):
    org_set = org_set.copy()
    org_set.loc[:, 'patient_id'] = org_set['patient_id'].astype(str)
    sy_set.loc[:, 'patient_id'] = sy_set['patient_id'].astype(str)
    for patient in org_set['patient_id']:
        synthetic_cases = sy_set[sy_set['patient_id'].str.startswith(patient + "_")]
        org_set = pd.concat([org_set, synthetic_cases], ignore_index=True)
    return org_set.copy() 

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

# Initializations.    
seed = pl.seed_everything(42, workers=True)
torch.manual_seed(seed)
repo_path = os.getcwd()
CONFIG_PATH = repo_path + '/configs'
parser = argparse.ArgumentParser(description='HEPred')
parser.add_argument('--config', type=str, default='config.yaml', help='The config file path')
args = parser.parse_args()
config = load_config(args.config)
name_config = args.config
gpu = config["GPU"]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
print("Config file: ", name_config)

# Main CV loop. 
number_of_synthetic_imgs = 5
train_total_time = 0
if __name__ == '__main__':  
    start_time_train = time.time()

    # Evaluation metrics. 
    val_losses = []
    val_accs = []
    train_losses = []
    train_accs = []
    val_precisions = []
    val_f1s = []
    train_precisions = []
    train_f1s = []
    train_acc_pws = []
    val_acc_pws = []
    train_prec_pws = []
    val_prec_pws = []
    train_f1_pws = []
    val_f1_pws = []
    test_acc_pws = []
    test_prec_pws = []
    test_f1_pws = []
    test_sens_pws = []
    test_spec_pws = []
    test_sens_iws = []
    test_spec_iws = []
    test_auc_iws = []
    test_auc_pws = []
    test_acc_iws = []
    test_prec_iws = []
    test_f1_iws = []
    val_sens_pws = []
    val_spec_pws = []
    val_roc_aucs = []
    val_sens_iws = []
    val_spec_iws = []
    val_roc_aucs_iws = []

    # Export config parameters.
    threshold_name = config["threshold_name"]
    experiment_sample= config["experiment_sample"]
    threshold = config["threshold_percentage"]
    print("experiment sample: ", experiment_sample)
    PATIENCE = config["PATIENCE_EARLYSTOP"]
    MODEL = config["MODEL"]
    BACKBONE = config["BACKBONE"]
    no_ivh = config["NO_IVH"] # Eliminate ivh cases from the dataset
    task = config["TASK"]
    test_type = config["TEST_TYPE"]
    pw_based = config["PW_BASED"]
    print("Current backbone is: ", BACKBONE)

    # Fixed parameters. 
    NUM_WORKERS=8
    MAX_EPOCHS=20
    MASK=True
    FILTER_SEGMENTED = True
    n_splits_cv = 5
    use_class_weights=False
    model = f"{MODEL}"
    gradient_accumulation_steps =  1 
    test_pw_based = config["PW_BASED"]

    print("The experiment type is selected as:", test_type)

    # Reading the data using metadata files. 
    md_path = repo_path + "/data/metadata.csv" # contains all the 205 cases
    dataset = PredictionDataset(md_path=md_path)
    df = pd.DataFrame({'patient_id': dataset.patient_id, 'index': dataset.index})
    metadata = pd.read_csv(repo_path + "/data/data_70_cases.csv") # imbalanced dataset.
    data = metadata.copy()
    sythetic_path = repo_path + "/data/metadata_synthetic_with_original.csv"
    synthetic_dataset = pd.read_csv(sythetic_path)
    dataset_sy = PredictionDataset(md_path=sythetic_path)
    df_sy = pd.DataFrame({'patient_id': dataset_sy.patient_id, 'index': dataset_sy.index})

    skf = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(skf.split(data, data['label'])):
        print("Fold: ", i)
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        # Further split the 'train_data' into training and validation sets
        skf_train = StratifiedKFold(n_splits=n_splits_cv-1, shuffle=True, random_state=42)

        for train_idx, val_idx in skf_train.split(train_data, train_data['label']):
            train_subset = train_data.iloc[train_idx]
            val_subset = train_data.iloc[val_idx]

        print("Train, val and test sets are created")
        print("The length of train, val and test sets originally are:", len(train_subset), len(val_subset), len(test_data))

        if test_type == "t3" or test_type == "t5":
            print("Th experiment type is either t3 or t5, so synthetic data is added to the train set.")
            train_subset = add_sythetic_cases(train_subset, synthetic_dataset)
            train_subset.to_csv(repo_path + f"/data/stratified_kfold/train_fold_{i}_{number_of_synthetic_imgs}sy.csv", index=False)
            val_subset.to_csv(repo_path + f"/data/stratified_kfold/val_fold_{i}.csv", index=False)
            test_data.to_csv(repo_path + f"/data/stratified_kfold/test_fold_{i}.csv", index=False)
            train_set_copy = pd.read_csv(repo_path + f'/data/stratified_kfold/train_fold_{i}_{number_of_synthetic_imgs}sy.csv')
            val_set_copy = pd.read_csv(repo_path + f'/data/stratified_kfold/val_fold_{i}.csv')
            test_set_copy = pd.read_csv(repo_path + f'/data/stratified_kfold/test_fold_{i}.csv')

            df_sy.loc[:, 'patient_id'] = df_sy['patient_id'].astype(str)
            train_set_copy.loc[:, 'patient_id'] = train_set_copy['patient_id'].astype(str)
            val_set_copy.loc[:, 'patient_id'] = val_set_copy['patient_id'].astype(str)
            test_set_copy.loc[:, 'patient_id'] = test_set_copy['patient_id'].astype(str)

            train_df = df_sy[df_sy['patient_id'].isin(train_set_copy['patient_id'])]
            val_df = df_sy[df_sy['patient_id'].isin(val_set_copy['patient_id'])]
            test_df = df_sy[df_sy['patient_id'].isin(test_set_copy['patient_id'])]

            print("The lenght of training set after adding synthetic cases: ", len(train_df['index'].values))
                    
            X_train, X_val= dataset.index[train_df['index'].values], dataset.index[val_df['index'].values]
            y_train, y_val = dataset.labels[train_df['index'].values], dataset.labels[val_df['index'].values]
            X_test, y_test = dataset.index[test_df['index'].values], dataset.labels[test_df['index'].values]
            dataset_to_use = sythetic_path
        
        else:
            print("The experiment type is either t1 or t2, so no synthetic data is added to the train set.")
            train_subset.to_csv(repo_path + f"/data/stratified_kfold/train_fold_{i}.csv", index=False)
            val_subset.to_csv(repo_path + f"/data/stratified_kfold/val_fold_{i}.csv", index=False)
            test_data.to_csv(repo_path + f"/data/stratified_kfold/test_fold_{i}.csv", index=False)
            train_set_copy = pd.read_csv(repo_path + f'/data/stratified_kfold/train_fold_{i}.csv')
            val_set_copy = pd.read_csv(repo_path + f'/data/stratified_kfold/val_fold_{i}.csv')
            test_set_copy = pd.read_csv(repo_path + f'/data/stratified_kfold/test_fold_{i}.csv')

            train_df = df[df['patient_id'].isin(train_set_copy['patient_id'])]
            val_df = df[df['patient_id'].isin(val_set_copy['patient_id'])]
            test_df = df[df['patient_id'].isin(test_set_copy['patient_id'])]
                    
            X_train, X_val= dataset.index[train_df['index'].values], dataset.index[val_df['index'].values]
            y_train, y_val = dataset.labels[train_df['index'].values], dataset.labels[val_df['index'].values]
            X_test, y_test = dataset.index[test_df['index'].values], dataset.labels[test_df['index'].values]
            dataset_to_use = md_path

        indexes = X_train, X_val, X_test, y_train, y_val, y_test
        experiments_path_logger = repo_path + f"/lightning_logs/{threshold_name}/{experiment_sample}/"
        os.makedirs(experiments_path_logger, exist_ok=True)

        logger = TensorBoardLogger(experiments_path_logger, name=BACKBONE)
        
        # Call the data module using teh idexes that are splitted in the CV loop.
        dm = HEPredDataModule(split_indexes=indexes, filter_slices=FILTER_SEGMENTED,
                                mask=MASK, batch_size=config["BATCH_SIZE"], 
                                num_workers=NUM_WORKERS, threshold=config["THRESHOLD"],
                                md_path=dataset_to_use, image_size = config["INPUT_SIZE"],
                                test_type = test_type, apply_hflip = config["APPLY_HFLIP"],
                                apply_affine = config["APPLY_AFFINE"],
                                apply_gaussian_blur=config["APPLY_GAUSSIAN_BLUR"],
                                affine_degree=config["AFFINE_DEGREE"],
                                affine_translate=config["AFFINE_TRANSLATE"],
                                affine_scale=config["AFFINE_SCALE"],
                                affine_shear=config["AFFINE_SHEAR"], 
                                hflip_p = config["HFLIP_P"], affine_p = config["AFFINE_P"],)
        dm.setup()

        checkpoint_path = repo_path + f"/checkpoints/{threshold_name}/{experiment_sample}/{i}"
        os.makedirs(checkpoint_path, exist_ok=True)

        early_stopping_callback = EarlyStopping(monitor='val_loss_epoch', patience=PATIENCE, mode="min")
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(checkpoint_path),
            filename='last_model',
            save_top_k=1,)
        
        name_experiment = f"{threshold_name}_{experiment_sample}_{BACKBONE}_f{i}_basalfu_reg"
        print("THE MODEL IS: ", model)
        classifier = ImageClassifier(back=BACKBONE, name_experiment=name_experiment, model=model, num_classes=1,
                                    learning_rate=config["LR"], weight_decay = config["WEIGHT_DECAY"],
                                    optimizer=config["OPTIMIZER"], momentum = config["MOMENTUM"],
                                    lr_scheduler= config["LR_SCHEDULER"], step_size = config["STEP_SIZE"],
                                    gamma = config["GAMMA"], class_weights=torch.tensor([1,1]),
                                    pw_based=pw_based, test_pw_based = test_pw_based, threshold= threshold,
                                    loss= config["LOSS"], fold=i, task = config["T"], save_predictions=True,)  

        trainer = pl.Trainer(max_epochs=MAX_EPOCHS, 
                            logger=logger, 
                            accelerator="gpu", 
                            callbacks=[early_stopping_callback,checkpoint_callback],
                            fast_dev_run=False,
                            accumulate_grad_batches=gradient_accumulation_steps,)
        
        # Training phase.
        trainer.fit(model = classifier, datamodule=dm)

        val_loss_epoch = trainer.callback_metrics['val_loss_epoch']
        val_acc_epoch = trainer.callback_metrics['val_acc_epoch']
        train_loss_epoch = trainer.callback_metrics['train_loss_epoch']
        train_acc_epoch = trainer.callback_metrics['train_acc_epoch']
        val_precision_epoch = trainer.callback_metrics['val_prec_epoch']
        val_f1_epoch = trainer.callback_metrics['val_f1_epoch']
        train_precision_epoch = trainer.callback_metrics['train_prec_epoch']
        train_f1_epoch = trainer.callback_metrics['train_f1_epoch']
        train_acc_pw = trainer.callback_metrics['train_acc_pw_epoch']
        val_acc_pw = trainer.callback_metrics['val_acc_pw_epoch']
        train_f1_pw = trainer.callback_metrics['train_f1_pw_epoch']
        val_f1_pw = trainer.callback_metrics['val_f1_pw_epoch']
        train_precision_pw = trainer.callback_metrics['train_prec_pw_epoch']
        val_precision_pw = trainer.callback_metrics['val_prec_pw_epoch']
        val_sensitivity_pw = trainer.callback_metrics['val_sens_pw']
        val_specificity_pw = trainer.callback_metrics['val_spec_pw']
        val_rocauc = trainer.callback_metrics['val_roc_auc']
        val_sensitivity_iw = trainer.callback_metrics['val_sens_iw']
        val_specificity_iw = trainer.callback_metrics['val_spec_iw']
        val_rocauc_iw = trainer.callback_metrics['val_roc_auc_iw']

        val_losses.append(val_loss_epoch)
        val_accs.append(val_acc_epoch)
        train_losses.append(train_loss_epoch)
        train_accs.append(train_acc_epoch)
        val_precisions.append(val_precision_epoch)
        val_f1s.append(val_f1_epoch)
        train_precisions.append(train_precision_epoch)
        train_f1s.append(train_f1_epoch)
        train_acc_pws.append(train_acc_pw)
        val_acc_pws.append(val_acc_pw)
        train_f1_pws.append(train_f1_pw)
        val_f1_pws.append(val_f1_pw)
        train_prec_pws.append(train_precision_pw)
        val_prec_pws.append(val_precision_pw)
        val_sens_pws.append(val_sensitivity_pw)
        val_spec_pws.append(val_specificity_pw)
        val_roc_aucs.append(val_rocauc)
        val_sens_iws.append(val_sensitivity_iw)
        val_spec_iws.append(val_specificity_iw)
        val_roc_aucs_iws.append(val_rocauc_iw)

        end_time_train = time.time()
        elapsed_time_train = end_time_train - start_time_train
        train_total_time += elapsed_time_train

        # Test phase.
        start_time_test = time.time()
        trainer.test(model = classifier, datamodule=dm)

        test_auc_pws.append(trainer.callback_metrics['test_roc_auc_score_pw'])
        test_auc_iws.append(trainer.callback_metrics['test_roc_auc_score_iw'])
        test_acc_iws.append(trainer.callback_metrics['test_acc_epoch'])
        test_prec_iws.append(trainer.callback_metrics['test_prec_epoch'])
        test_f1_iws.append(trainer.callback_metrics['test_f1_epoch'])
        test_prec_pws.append(trainer.callback_metrics['test_prec_pw_epoch'])
        test_f1_pws.append(trainer.callback_metrics['test_f1_pw_epoch'])
        test_acc_pws.append(trainer.callback_metrics['test_acc_pw_epoch'])
        test_sens_pws.append(trainer.callback_metrics['test_sens_pw_epoch'])
        test_spec_pws.append(trainer.callback_metrics['test_spec_pw_epoch'])
        test_sens_iws.append(trainer.callback_metrics['test_sens_iw_epoch'])
        test_spec_iws.append(trainer.callback_metrics['test_spec_iw_epoch'])

        results_folder_path = repo_path + "/results"
        os.makedirs(results_folder_path, exist_ok=True)

        with open(f"{results_folder_path}/{threshold_name}_{experiment_sample}_fold{i}.txt", "a") as f:
            f.write(f"Fold {i}:\n")
            f.write(f"val acc pw {val_acc_pw}\n")
            f.write(f"train acc pw {train_acc_pw}\n")
            f.write(f"val sens pw {val_sensitivity_pw}\n")
            f.write(f"val spec pw {val_specificity_pw}\n")
            f.write(f"val roc auc pw {val_rocauc}\n")
            f.write(f"test acc pw {trainer.callback_metrics['test_acc_pw_epoch']}\n")
            f.write(f"test sens pw {trainer.callback_metrics['test_sens_pw_epoch']}\n")
            f.write(f"test spec pw {trainer.callback_metrics['test_spec_pw_epoch']}\n")
            f.write(f"test roc auc pw {trainer.callback_metrics['test_roc_auc_score_pw']}\n")

        end_time_test = time.time()
        elapsed_time_test = end_time_test - start_time_test

    # Save the results in a txt file.
    with open(f"{results_folder_path}/{threshold_name}_{experiment_sample}_results.txt", "w") as f:
        f.write(f"Average train accuracy iw: {np.mean(train_accs)}\n")
        f.write(f"Average train loss iw: {np.mean(train_losses)}\n")
        f.write(f"Average train precision iw: {np.mean(train_precisions)}\n")
        f.write(f"Average train f1 iw: {np.mean(train_f1s)}\n")
        f.write(f"Average train accuracy pw: {np.mean(train_acc_pws)}\n")
        f.write(f"Average train precision pw: {np.mean(train_prec_pws)}\n")
        f.write(f"Average train f1 pw: {np.mean(train_f1_pws)}\n")

        f.write(f"Average validation accuracy iw: {np.mean(val_accs)}\n")
        f.write(f"Average validation loss iw: {np.mean(val_losses)}\n")
        f.write(f"Average validation precision iw: {np.mean(val_precisions)}\n")
        f.write(f"Average validation f1 iw: {np.mean(val_f1s)}\n") 
        f.write(f"Average validation accuracy pw: {np.mean(val_acc_pws)}\n")
        f.write(f"Average validation precision pw: {np.mean(val_prec_pws)}\n")      
        f.write(f"Average validation f1 pw: {np.mean(val_f1_pws)}\n")        
        f.write(f"Average validation sensitivity pw: {np.mean(val_sens_pws)}\n")
        f.write(f"Average validation specificity pw: {np.mean(val_spec_pws)}\n")
        f.write(f"Average validation sensitivity iw: {np.mean(val_sens_iws)}\n")
        f.write(f"Average validation specificity iw: {np.mean(val_spec_iws)}\n")
        f.write(f"Average validation roc auc pw: {np.mean(val_roc_aucs)}\n")
        f.write(f"Average validation roc auc iw: {np.mean(val_roc_aucs_iws)}\n")         
        
        f.write(f"Average test accuracy iw: {np.mean(test_acc_iws)}\n")
        f.write(f"Average test f1 iw: {np.mean(test_f1_iws)}\n")
        f.write(f"Average test accuracy pw: {np.mean(test_acc_pws)}\n")
        f.write(f"Average test precision pw: {np.mean(test_prec_pws)}\n")
        f.write(f"Average test f1 pw: {np.mean(test_f1_pws)}\n")
        f.write(f"Average test precision iw: {np.mean(test_prec_iws)}\n")

        f.write(f"Average test sensitivity pw: {np.mean(test_sens_pws)}\n")
        f.write(f"Average test specificity pw: {np.mean(test_spec_pws)}\n")
        f.write(f"Average test sensitivity iw: {np.mean(test_sens_iws)}\n")
        f.write(f"Average test specificity iw: {np.mean(test_spec_iws)}\n")
        f.write(f"Average test auc pw: {np.mean(test_auc_pws)}\n")
        f.write(f"Average test auc iw: {np.mean(test_auc_iws)}\n")

        f.write(f"Std validation  roc auc pw: {np.std(val_roc_aucs)}\n")
        f.write(f"Std validation  roc auc iw: {np.std(val_roc_aucs_iws)}\n")
        f.write(f"Std validation  accuracy pw: {np.std(val_acc_pws)}\n")
        f.write(f"Std validation  accuracy iw: {np.std(val_accs)}\n")
        f.write(f"Std validation  precision pw: {np.std(val_prec_pws)}\n")
        f.write(f"Std validation  precision iw: {np.std(val_precisions)}\n")
        f.write(f"Std validation  f1 pw: {np.std(val_f1_pws)}\n")
        f.write(f"Std validation  f1 iw: {np.std(val_f1s)}\n")
        f.write(f"Std validation  sensitivity pw: {np.std(val_sens_pws)}\n")
        f.write(f"Std validation  sensitivity iw: {np.std(val_sens_iws)}\n")
        f.write(f"Std validation  specificity pw: {np.std(val_spec_pws)}\n")
        f.write(f"Std validation  specificity iw: {np.std(val_spec_iws)}\n")

        f.write(f"Std test  roc auc pw: {np.std(test_auc_pws)}\n")
        f.write(f"Std test  roc auc iw: {np.std(test_auc_iws)}\n")
        f.write(f"Std test  accuracy pw: {np.std(test_acc_pws)}\n")
        f.write(f"Std test  accuracy iw: {np.std(test_acc_iws)}\n")
        f.write(f"Std test  precision pw: {np.std(test_prec_pws)}\n")
        f.write(f"Std test  precision iw: {np.std(test_prec_iws)}\n")
        f.write(f"Std test  f1 pw: {np.std(test_f1_pws)}\n")
        f.write(f"Std test  f1 iw: {np.std(test_f1_iws)}\n")
        f.write(f"Std test  sensitivity pw: {np.std(test_sens_pws)}\n")
        f.write(f"Std test  sensitivity iw: {np.std(test_sens_iws)}\n")
        f.write(f"Std test  specificity pw: {np.std(test_spec_pws)}\n")
        f.write(f"Std test  specificity iw: {np.std(test_spec_iws)}\n")


    log_message = f"Train execution time: {train_total_time} and test time {elapsed_time_test} seconds\n"
    with open(f"execution_times_log_{experiment_sample}.txt", "a") as file:
        file.write(log_message)