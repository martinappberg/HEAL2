import torch

class Rescaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, loader):
        all_feats = []
        for batch in loader:
            feats, _, _ = batch
            all_feats.append(feats)
        all_feats = torch.cat(all_feats, dim=0)
        self.mean = all_feats.mean(dim=0, keepdim=True)
        self.std = all_feats.std(dim=0, keepdim=True)
        print(f"NORMALIZED")

    def transform(self, feats):
        return (feats - self.mean) / (self.std + 1e-5)

    def fit_transform(self, loader):
        self.fit(loader)
        transformed_feats = []
        for batch in loader:
            feats, _, _ = batch
            transformed_feats.append(self.transform(feats))
        return torch.cat(transformed_feats, dim=0)