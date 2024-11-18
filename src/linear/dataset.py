import torch
import random
import numpy as np
from .utils import load



class Dataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(self, data_path, dataset, sample_ids, labels, covariates, balanced_sampling=False, ancestries=None, multiple_ancestries=False, shuffle_labels=False, rescaler=None):
        self.sample_ids = sample_ids
        self.labels = labels
        if shuffle_labels:
            shuffled_indices = torch.randperm(len(labels))
            self.labels = labels[shuffled_indices]
        self.ancestries = ancestries
        self.data_path = data_path
        self.dataset = dataset
        self.multiple_ancestries = multiple_ancestries
        self.balanced_sampling = balanced_sampling
        self.covariates = covariates

        if balanced_sampling == True:
            self.classes = torch.unique(self.labels)
            self.class_indices = {cls.item(): torch.where(self.labels == cls)[0] for cls in self.classes}
            self.n_samples = 10000000
        else:
            self.n_samples = len(self.labels)
        self.n_acutal_samples = len(self.labels)
        self.rescaler = rescaler

        print(f"Dataset initialized with {self.n_samples} samples")
    def __len__(self):
        return self.n_samples

    def __balanced_sampling__(self):
        cls = random.choice(self.classes)
        class_indices = self.class_indices[cls.item()]
        index = random.choice(class_indices)
        return index

    def __getitem__(self, index):
        if self.balanced_sampling:
            index = self.__balanced_sampling__()
        sample_id = self.sample_ids[index]
        label = self.labels[index]
        covariate = self.covariates[index]
        feat = np.load(f'{self.data_path}/{self.dataset}/feats/{sample_id}.npy') # type: ignore
        if self.rescaler is not None:
            feat = self.rescaler.transform(feat.reshape(1, -1)).reshape(-1)
        if self.multiple_ancestries:
            return {"feat":feat, "ancestry":torch.FloatTensor([self.ancestries[index]]), "label":label} # type: ignore
        else:
            return {"feat":feat, "label":label, "sample_id": sample_id, "covariate": covariate}
        
    def fit_scaler(self, scaler):
        """Fits a StandardScaler on all features in the dataset and sets it as self.rescaler."""
        all_feats = []
        for i in range(self.n_acutal_samples):
            item = self.__getitem__(i)
            feat = item['feat']
            feat = feat.reshape(1, -1)
            all_feats.append(feat)

        all_feats = np.concatenate(all_feats, axis=0)  # Shape: (total_samples, num_features)

        # Fit scaler on all features and set as the dataset's scaler
        scaler.fit(all_feats)
        self.rescaler = scaler
        print("Normalized all training features")
        return scaler