import numpy as np
import random
import torch
import dgl

import os
# SKLearn
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, StratifiedKFold

def set_random_seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = False # type: ignore
    dgl.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load(file):
    if type(file) == str:
        file=open(file,"rb")
    header = file.read(128)
    if not header:
        return None
    descr = str(header[19:25], 'utf-8').replace("'","").replace(" ","")
    shape = tuple(int(num) for num in str(header[60:120], 'utf-8').replace(', }', '').replace('(', '').replace(')', '').split(','))
    datasize = np.lib.format.descr_to_dtype(descr).itemsize
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))

def generate_splits(labels, train_ratio=0.8, val_ratio=0.1, n_splits=3):
    seeds = [20+i for i in range(n_splits)]
    splits = []
    pos_indexs = np.where(labels == 1)[0]
    neg_indexs = np.where(labels == 0)[0]
    for seed in seeds:
        np.random.seed(seed)
        np.random.shuffle(pos_indexs)
        
        shuffled_indices = np.random.permutation(len(pos_indexs))
        
        total_length = len(pos_indexs)
        
        train_size = int(total_length * train_ratio)
        val_size = int(total_length * val_ratio)
        
        # Split the shuffled indices
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:train_size + val_size]
        test_indices = shuffled_indices[train_size + val_size:]
        
        # Create train, validation, and test sets for positive class
        train_pos_data = pos_indexs[train_indices]
        val_pos_data = pos_indexs[val_indices]
        test_pos_data = pos_indexs[test_indices]
        
        # Sample negative indices for train, validation, and test sets
        train_neg_indices = np.random.choice(neg_indexs, size=int(len(neg_indexs)*0.8), replace=False)
        val_neg_indices = np.random.choice(np.setdiff1d(neg_indexs, train_neg_indices), size=int(len(neg_indexs)*0.1), replace=False)
        test_neg_indices = np.setdiff1d(np.setdiff1d(neg_indexs, train_neg_indices), val_neg_indices)
        
        # Combine positive and negative indices for each split
        train_indices = np.concatenate((train_pos_data, train_neg_indices))
        val_indices = np.concatenate((val_pos_data, val_neg_indices))
        test_indices = np.concatenate((test_pos_data, test_neg_indices))
        
        # Shuffle the combined indices
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        
        splits.append((train_indices, val_indices, test_indices))
    
    return splits

def collate_fn_ma(batch):
    feats = [sample['feat'] for sample in batch]
    ancestrys = [sample['ancestry'] for sample in batch]
    labels = [sample['label'] for sample in batch]
    feats = torch.from_numpy(np.stack(feats)).to(torch.float32)
    ancestrys = torch.LongTensor(ancestrys)
    labels = torch.FloatTensor(labels).to(torch.float32).reshape(-1,1)
    return feats, ancestrys, labels

def collate_fn(batch):
    feats = [sample['feat'] for sample in batch]
    labels = [sample['label'] for sample in batch]
    sample_ids = [sample['sample_id'] for sample in batch]
    covariates = [sample['covariate'] for sample in batch]
    feats = torch.from_numpy(np.stack(feats)).to(torch.float32)
    labels = torch.FloatTensor(labels).to(torch.float32).reshape(-1,1)
    covariates = torch.from_numpy(np.stack(covariates)).to(torch.float32)
    return feats, labels, sample_ids, covariates

def ancestry_encoding(ancestries):
    encoded_ancestries = []
    for ancestry in ancestries:
        if ancestry == 'EUR':
            encoded_ancestries.append(0)
        if ancestry == 'SAS':
            encoded_ancestries.append(1)
        if ancestry == 'AFR':
            encoded_ancestries.append(2)
        else:
            encoded_ancestries.append(3)
    return encoded_ancestries

## UTILS FOR VALIDATION
def create_dir_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def logo_splits(labels, groups, val_ratio=0.1, random_state=42):
    """
    Generate train, validation, and test splits using Leave-One-Group-Out strategy.

    Args:
    - labels (array-like): Target labels.
    - groups (array-like): Group labels defining the splits.
    - val_ratio (float): Proportion of the dataset to include in the validation split.

    Returns:
    - splits (list of tuples): List containing train, validation, and test indices for each split.
    """
    logo = LeaveOneGroupOut()
    splits = []

    for train_val_idx, test_idx in logo.split(np.zeros(len(labels)), labels, groups):
        test_group = np.unique(groups[test_idx])
        # Within each LOGO split, further split the training set into training and validation sets
        if val_ratio > 0:
            # Ensure stratification by labels within the training set
            train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio, stratify=labels[train_val_idx], random_state=random_state)
            train_groups = np.unique(groups[train_idx])
            val_groups = np.unique(groups[val_idx])
            splits.append((train_idx, val_idx, test_idx, train_groups, val_groups, test_group))
        else:
            # No validation split, use all training data as is
            train_groups = np.unique(groups[train_val_idx])
            splits.append((train_val_idx, [], test_idx, train_groups, [], test_group))

    return splits

def sk_splits(labels, test_ratio=0.2, val_ratio=0.2, n_splits=3, random_state=42):
    splits = []
    indices = np.arange(len(labels))
    for rs in range(n_splits):
        # Initial split into training and testing
        train_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, stratify=labels, random_state=(random_state*rs))

        # Further split the training set to create a validation set
        final_train_indices, val_indices = train_test_split(
            train_indices, test_size=val_ratio, stratify=labels[train_indices], random_state=(random_state*rs))

        splits.append((final_train_indices, val_indices, test_indices, f'tts', 'tts', 'tts'))

    return splits

def stratified_k_fold_splits(labels, test_ratio=0.2, n_splits=3, random_state=42, test_indices=None, test_group='skf'):
    splits = []
    indices = np.arange(len(labels))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if test_indices is None:
        train_indices, test_indices = train_test_split(
                indices, test_size=test_ratio, stratify=labels, random_state=(random_state))
    else:
        train_indices = np.setdiff1d(indices, test_indices)
    # Generate folds
    for train_index, val_index in skf.split(train_indices, labels[train_indices]):
        # Here, you get indices for training and test in each fold
        splits.append((train_indices[train_index], train_indices[val_index], test_indices, 'skf', 'skf', test_group))

    return splits

def validation_split(labels, ratio=0.2, random_state=42, test_indices=[], test_group='tts', empty_test=False):
    indices = np.arange(len(labels))
    train_indices = []
    val_indices = []

    if empty_test:
        train_indices, val_indices = train_test_split(
            indices, test_size=ratio, stratify=labels, random_state=random_state)
        return [(train_indices, val_indices, test_indices, 'only_val', 'only_val', 'only_val')]

    if len(test_indices) == 0:
        # If no test indices are provided, split into train+val and test
        train_val_indices, test_indices = train_test_split(
            indices, test_size=ratio, stratify=labels, random_state=random_state)
        
        # Further split train+val into train and val
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=ratio / (1 - ratio), 
            stratify=labels[train_val_indices], random_state=random_state)
    else:
        # If test indices are provided, split the remaining indices into train and val
        train_val_indices = np.setdiff1d(indices, test_indices)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=ratio,
            stratify=labels[train_val_indices], random_state=random_state)
    return [(train_indices, val_indices, test_indices, 'tts', 'tts', test_group)]
    

def skf_validation_split(labels, n_splits=3, random_state=42):
    splits = []
    indices = np.arange(len(labels))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_index, val_index in skf.split(indices, labels[indices]):
        # Here, you get indices for training and test in each fold
        splits.append((indices[train_index], indices[val_index], None, 'skf_val', 'skf_val', 'skf_val'))

    return splits