import torch

class Rescaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, loader):
        sum_feats = 0
        sum_square_feats = 0
        num_samples = 0
        
        for batch in loader:
            feats, _, _ = batch
            sum_feats += torch.sum(feats, dim=0)
            sum_square_feats += torch.sum(feats ** 2, dim=0)
            num_samples += feats.size(0)
        
        self.mean = sum_feats / num_samples
        self.std = torch.sqrt(sum_square_feats / num_samples - self.mean ** 2)
        print(f"Fitted: Mean - {self.mean}, Std - {self.std}")

    def transform(self, feats):
        return (feats - self.mean) / (self.std + 1e-5)

    def fit_transform(self, loader):
        self.fit(loader)
        transformed_feats = []
        
        for batch in loader:
            feats, _, _ = batch
            transformed_feats.append(self.transform(feats))
        
        return torch.cat(transformed_feats, dim=0)
    
class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, loader):
        first_batch = True
        for batch in loader:
            feats, _, _ = batch
            if first_batch:
                self.min = torch.min(feats, dim=0)[0]
                self.max = torch.max(feats, dim=0)[0]
                first_batch = False
            else:
                self.min = torch.min(self.min, torch.min(feats, dim=0)[0])
                self.max = torch.max(self.max, torch.max(feats, dim=0)[0])
        print(f"Fitted: Min - {self.min}, Max - {self.max}")

    def transform(self, feats):
        return (feats - self.min) / (self.max - self.min + 1e-5)

    def fit_transform(self, loader):
        self.fit(loader)
        transformed_feats = []
        for batch in loader:
            feats, _, _ = batch
            transformed_feats.append(self.transform(feats))
        return torch.cat(transformed_feats, dim=0)
