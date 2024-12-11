import torch
import torch.nn as nn

class GeneScoreModel(nn.Module):
    def __init__(self, num_features, num_genes, covariate_dim, l1_feats=0.0, l1_genes=0.0, only_covariates=False):
        super(GeneScoreModel, self).__init__()
        # Linear layer for transforming features per gene into a single score per gene
        self.fc_feats = nn.Linear(num_features, 1)
        # Linear layer for combining gene scores and covariates into a single prediction
        self.fc_genes = nn.Linear(num_genes, 1)
        # Linear layer for covariate adjustment
        self.fc_covariates = nn.Linear(covariate_dim, 1)

        self.batch_norm_genes = nn.BatchNorm1d(num_genes)

        # Store lambda for L1 regularization
        self.l1_feats = l1_feats
        self.l1_genes = l1_genes
        self.num_genes = num_genes
        self.num_features = num_features

        self.only_covariates = only_covariates
        self.covariate_dim = covariate_dim

    def forward(self, feats, covariates):
        gene_output = 0
        if not self.only_covariates:
            batch_size = feats.size(0)
            feats = feats.view(batch_size, self.num_genes, self.num_features)
            # Transform each gene's features into a single score for that gene
            gene_scores = self.fc_feats(feats).squeeze(-1)  # (batch_size, num_genes)
            # Concatenate gene scores with covariates for final prediction
            gene_scores = self.batch_norm_genes(gene_scores)
            # Final prediction combining gene scores and covariates
            gene_output = self.fc_genes(gene_scores)  # (batch_size, 1)
        
        covariate_output = 0
        if self.covariate_dim > 0:
            covariate_output = self.fc_covariates(covariates)

        final_output = gene_output + covariate_output
        # Compute L1 regularization (on the weights that connect features to gene scores and on the genes themselves)
        l1_loss_feats = self.l1_feats * torch.sum(torch.abs(self.fc_feats.weight))
        l1_loss_genes = self.l1_genes * torch.sum(torch.abs(self.fc_genes.weight))
        l1_loss = l1_loss_feats + l1_loss_genes

        return final_output, l1_loss