import torch
from torch import nn
import dgl
import numpy as np
class Trainer:
    def __init__(self, device, eval_interval=100, n_steps=20000, n_early_stop=50, log_interval=20):
        self.device=device
        self.eval_interval = eval_interval
        self.n_steps = n_steps
        self.n_early_stop = n_early_stop
        self.log_interval = log_interval
    def forward_batch(self, model, ggi_graph, batch):
        feats, labels, sample_ids, covariates = batch
        batched_graph = dgl.batch([ggi_graph]*len(labels)).to(self.device)
        feats, labels, covariates = feats.to(self.device), labels.to(self.device), covariates.to(self.device)
        outputs, attn_scores, sae_loss, z_sae = model(batched_graph, feats, covariates)
        attn_scores = attn_scores.view(batched_graph.batch_size, ggi_graph.number_of_nodes())
        z_sae = z_sae.view(batched_graph.batch_size, ggi_graph.number_of_nodes(), 64)
        z_sae = torch.sum(torch.abs(z_sae), dim=2)
        return labels, outputs, attn_scores, sample_ids, sae_loss, z_sae
    def train_and_test(self, model, ggi_graph, loss_fn, optimizer, metric_funcs, train_loader, val_loader, test_loader=None, evaltrain_loader=None, calculate_feature_importance=False):
        best_val_scores, best_test_scores, best_train_scores = {name: 0 for name in metric_funcs}, {name: 0 for name in metric_funcs}, {name: 0 for name in metric_funcs}
        best_model_state = None
        running_loss = []
        cur_early_stop = 0
        best_train_attn_list = {}
        best_val_attn_list = {}
        best_test_attn_list = {}
        best_val_predictions = {}
        test_predictions = {}

        # Z_sae
        train_z_sae_list = {}
        best_val_z_sae_list = {}
        test_z_sae_list = {}

        # Feature importance
        train_feature_importance = {}
        best_val_feature_importance = {}
        test_feature_importance = {}

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
            
            labels, preds, attn_scores, sample_ids, sae_loss, _ = self.forward_batch(model, ggi_graph, batch)

            loss = loss_fn(preds, labels)
            total_loss = loss + sae_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            running_loss.append(loss.detach().cpu().numpy())
            if (cur_step+1) % self.log_interval == 0:
                print(f"[{cur_step+1}] loss: {np.mean(running_loss):.3f}", flush=True)
                running_loss = []
            if (cur_step+1) % self.eval_interval == 0:
                running_loss = []
                print("----------------Validating----------------", flush=True)
                val_scores, val_attn_list, val_predictions, val_z_sae_list, val_feature_importance = self.evaluate(model, ggi_graph, val_loader, metric_funcs, calculate_feature_importance)
                if val_scores['auroc'] > best_val_scores['auroc']:
                    best_val_scores = val_scores
                    best_val_predictions = val_predictions
                    best_val_z_sae_list = val_z_sae_list
                    best_val_feature_importance = val_feature_importance
                    # Store best model
                    best_model_state = model.state_dict()
                    ## Best model, add the attentions
                    ## VAL
                    best_val_attn_list = val_attn_list
                    if test_loader is not None:
                        best_test_scores, best_test_attn_list, test_predictions, test_z_sae_list, test_feature_importance = self.evaluate(model, ggi_graph, test_loader, metric_funcs, calculate_feature_importance)
                    if evaltrain_loader is not None:
                        best_train_scores, best_train_attn_list, train_predictions, train_z_sae_list, train_feature_importance = self.evaluate(model, ggi_graph, evaltrain_loader, metric_funcs, calculate_feature_importance)
                    cur_early_stop = 0
                else:
                    cur_early_stop += 1
                    print(f"Early stop {cur_early_stop}/{self.n_early_stop}")
                    if cur_early_stop == self.n_early_stop: break
                print(f"[{cur_step+1}] cur_val_score: {val_scores}, best_val_score: {best_val_scores}, test_score: {best_test_scores}", flush=True)
                print("----------------Training----------------", flush=True)
            if cur_step == self.n_steps: break
        return best_val_scores, best_test_scores, best_train_attn_list, best_val_attn_list, best_test_attn_list, best_model_state, best_val_predictions, test_predictions, best_train_scores, train_predictions, train_z_sae_list, best_val_z_sae_list, test_z_sae_list, train_feature_importance, best_val_feature_importance, test_feature_importance
        
    def evaluate(self, model, ggi_graph, test_loader, metric_funcs, calculate_feature_importance):
        model.eval()
        preds_list, labels_list = [], []
        attn_list = {}
        z_sae_list = {}
        sample_id_dict = {}
        feature_importance_per_sample = {}

        # Wrap the evaluation in torch.no_grad to save memory where gradients are not needed
        with torch.no_grad():
            for batch in test_loader:
                # Use forward_batch to handle the forward pass logic
                labels, preds, attn_scores, sample_ids, _, z_sae = self.forward_batch(model, ggi_graph, batch)
                
                # Detach tensors to avoid gradient tracking
                preds_list.append(preds.detach())
                labels_list.append(labels.detach())
                
                # Detach and move attn_scores and z_sae to CPU
                attn_scores = attn_scores.detach().cpu().numpy()
                z_sae = z_sae.detach().cpu().numpy()
                
                # Process attention and z_sae scores
                for i, sample in enumerate(sample_ids):
                    attn_list[sample] = attn_scores[i, :]
                    z_sae_list[sample] = z_sae[i, :]
                    sample_id_dict[sample] = {
                        'pred': torch.sigmoid(preds[i]).detach().cpu().numpy().item(),
                        'label': labels[i].detach().cpu().numpy().item()
                    }
        if calculate_feature_importance:
            # Now, calculate gradients only for feature importance
            for batch in test_loader:
                feats, labels, sample_ids, _ = batch
                batch_size = feats.size(0)  # Get batch size
                num_nodes = ggi_graph.number_of_nodes()  # Number of nodes

                # Enable gradients for features to compute feature importance
                feats.requires_grad_(True)

                # Use forward_batch for the forward pass with gradients enabled
                labels, preds, attn_scores, sample_ids, _, _ = self.forward_batch(model, ggi_graph, (feats, labels, sample_ids, _))

                # Now, calculate feature importance for each sample in the batch
                for i in range(batch_size):
                    # Zero out gradients from the previous iteration
                    if feats.grad is not None:
                        feats.grad.zero_()

                    # Compute gradient of the prediction w.r.t. the input features
                    pred = preds[i]

                    # Compute gradients
                    pred.backward(retain_graph=True)
                    sample_grads = feats.grad[i]  # Shape: [num_nodes * num_features]

                    # Compute absolute gradients to get feature importance
                    abs_gradients = sample_grads.abs()  # Shape: [num_nodes * num_features]
                    abs_gradients = abs_gradients.view(num_nodes, -1)

                    # Sum over nodes to get feature importance per feature
                    feature_importance = abs_gradients.sum(dim=0)  # Shape: [num_features]

                    # Store feature importance for the current sample
                    sample_id = sample_ids[i]
                    feature_importance_per_sample[sample_id] = feature_importance.detach().cpu().numpy()

                # Detach feats to prevent further gradient tracking and clear memory
                feats = feats.detach()

        # Concatenate predictions and labels for metric calculations
        preds = torch.cat(preds_list).reshape(-1).detach()
        labels = torch.cat(labels_list).reshape(-1).detach()

        # Compute metrics (ensure detachment)
        metric_results = {name: func(preds, labels.int()).item() for name, func in metric_funcs.items()}

        return metric_results, attn_list, sample_id_dict, z_sae_list, feature_importance_per_sample
