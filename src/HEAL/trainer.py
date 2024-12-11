import torch
from torch import nn
import numpy as np
class Trainer:
    def __init__(self, device, eval_interval=100, n_steps=20000, n_early_stop=50, log_interval=20):
        self.device=device
        self.eval_interval = eval_interval
        self.n_steps = n_steps
        self.n_early_stop = n_early_stop
        self.log_interval = log_interval
    def forward_batch(self, model, batch):
        feats, labels, sample_ids, covariates = batch
        # Move data to the correct device (GPU or CPU)
        feats, labels, covariates = feats.to(self.device), labels.to(self.device), covariates.to(self.device)
        # Forward pass through the model
        outputs, l1_loss = model(feats, covariates)
        return labels, outputs, sample_ids, l1_loss
    def train_and_test(self, model, loss_fn, optimizer, metric_funcs, train_loader, val_loader, test_loader=None, evaltrain_loader=None):
        best_val_scores, best_test_scores, best_train_scores = {name: 0 for name in metric_funcs}, {name: 0 for name in metric_funcs}, {name: 0 for name in metric_funcs}
        best_model_state = None
        running_loss = []
        cur_early_stop = 0
        best_val_predictions = {}
        test_predictions = {}

        best_model_gene_weights = None
        best_model_feat_weights = None

        data_iter = iter(train_loader)
        next_batch = next(data_iter)
        feats, labels, sample_ids, covariates = next_batch
        feats = feats.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        covariates = covariates.cuda(non_blocking=True)
        next_batch = (feats, labels, sample_ids, covariates)
        print("----------------Training----------------", flush=True)
        for cur_step in range(len(train_loader)):
            ## Forward pass
            model.train()
            optimizer.zero_grad()
            batch = next_batch 
            if cur_step + 1 != len(train_loader): 
                next_batch = next(data_iter)
                feats, labels, sample_ids, covariates = next_batch
                feats = feats.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                covariates = covariates.cuda(non_blocking=True)
                next_batch = (feats, labels, sample_ids, covariates)

            labels, preds, sample_ids, l1_loss = self.forward_batch(model, batch)

            loss = loss_fn(preds, labels)
            total_loss = loss + l1_loss

            # Backward pass and optimization
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss.append(loss.detach().cpu().numpy())
            if (cur_step+1) % self.log_interval == 0:
                print(f"[{cur_step+1}] loss: {np.mean(running_loss):.3f}", flush=True)
                running_loss = []
            if (cur_step+1) % self.eval_interval == 0:
                running_loss = []
                print("----------------Validating----------------", flush=True)
                val_scores, val_predictions = self.evaluate(model, val_loader, metric_funcs)
                if val_scores['auroc'] > best_val_scores['auroc']:
                    best_val_scores = val_scores
                    best_val_predictions = val_predictions
                    # Store best model
                    best_model_state = model.state_dict()
                    best_model_feat_weights = model.fc_feats.weight.flatten()
                    best_model_gene_weights = model.fc_genes.weight.flatten()

                    ## Best model, add the attentions
                    ## VAL
                    if test_loader is not None:
                        best_test_scores, test_predictions = self.evaluate(model, test_loader, metric_funcs)
                    if evaltrain_loader is not None:
                        best_train_scores, train_predictions = self.evaluate(model, evaltrain_loader, metric_funcs)
                    cur_early_stop = 0
                else:
                    cur_early_stop += 1
                    print(f"Early stop {cur_early_stop}/{self.n_early_stop}")
                    if cur_early_stop == self.n_early_stop: break
                print(f"[{cur_step+1}] cur_val_score: {val_scores}, best_val_score: {best_val_scores}, test_score: {best_test_scores}", flush=True)
                print("----------------Training----------------", flush=True)
            if cur_step == self.n_steps: break
        return best_val_scores, best_test_scores, best_model_state, best_val_predictions, test_predictions, best_train_scores, train_predictions, best_model_gene_weights.detach().cpu().numpy(), best_model_feat_weights.detach().cpu().numpy()
        
    def evaluate(self, model, test_loader, metric_funcs):
        model.eval()
        preds_list, labels_list = [], []
        sample_id_dict = {}

        # Wrap the evaluation in torch.no_grad to save memory where gradients are not needed
        with torch.no_grad():
            for batch in test_loader:
                # Use forward_batch to handle the forward pass logic
                labels, preds, sample_ids, _ = self.forward_batch(model, batch)
                
                # Detach tensors to avoid gradient tracking
                preds_list.append(preds.detach())
                labels_list.append(labels.detach())
                
                # Process attention and z_sae scores
                for i, sample in enumerate(sample_ids):
                    sample_id_dict[sample] = {
                        'pred': torch.sigmoid(preds[i]).detach().cpu().numpy().item(),
                        'label': labels[i].detach().cpu().numpy().item()
                    }

        # Concatenate predictions and labels for metric calculations
        preds = torch.cat(preds_list).reshape(-1).detach()
        labels = torch.cat(labels_list).reshape(-1).detach()

        # Compute metrics (ensure detachment)
        metric_results = {name: func(preds, labels.int()).item() for name, func in metric_funcs.items()}

        return metric_results, sample_id_dict