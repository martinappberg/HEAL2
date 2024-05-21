import torch
from torch import nn
import dgl
import numpy as np
class Trainer:
    def __init__(self, device, eval_interval=100, n_steps=20000, n_early_stop=50, log_interval=20, multiple_ancestries=False):
        self.device=device
        self.eval_interval = eval_interval
        self.n_steps = n_steps
        self.n_early_stop = n_early_stop
        self.log_interval = log_interval
        self.multiple_ancestries = multiple_ancestries
    def forward_batch(self, model, ggi_graph, batch):
        feats, labels, sample_ids = batch
        batched_graph = dgl.batch([ggi_graph]*len(labels)).to(self.device)
        feats, labels = feats.to(self.device), labels.to(self.device)
        outputs, attn_scores = model(batched_graph, feats)
        attn_scores = attn_scores.view(len(labels), ggi_graph.number_of_nodes())
        return labels, outputs, attn_scores, sample_ids
    def forward_batch_ma(self, model, ggi_graph, batch):
        feats, ancestries, labels = batch
        batched_graph = dgl.batch([ggi_graph]*len(labels)).to(self.device)
        feats, ancestries, labels = feats.to(self.device), ancestries.to(self.device), labels.to(self.device)
        outputs, ph_attn_scores, anc_attn_scores = model(batched_graph, feats, ancestries)
        return labels, outputs, ph_attn_scores, anc_attn_scores
    def train_and_test(self, model, ggi_graph, loss_fn, optimizer, metric_funcs, train_loader, val_loader, test_loader):
        best_val_scores, best_test_scores = {name: 0 for name in metric_funcs}, {name: 0 for name in metric_funcs}
        best_model_state = None
        running_loss = []
        cur_early_stop = 0
        best_train_attn_list = {}
        best_val_attn_list = {}
        best_test_attn_list = {}
        best_val_predictions = {}
        test_predictions = {}

        data_iter = iter(train_loader)
        next_batch = next(data_iter)
        feats, labels, sample_ids = next_batch
        feats = feats.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        next_batch = (feats, labels, sample_ids)
        print("----------------Training----------------", flush=True)
        for cur_step in range(len(train_loader)):
            ## Forward pass
            model.train()
            optimizer.zero_grad()
            batch = next_batch 
            if cur_step + 1 != len(train_loader): 
                next_batch = next(data_iter)
                feats, labels, sample_ids = next_batch
                feats = feats.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                next_batch = (feats, labels, sample_ids)
            if self.multiple_ancestries:
                labels, preds, ph_attn_scores, anc_attn_scores = self.forward_batch_ma(model, ggi_graph, batch)
            else:
                labels, preds, attn_scores, sample_ids = self.forward_batch(model, ggi_graph, batch)
                ## TRAIN
                attn_scores = attn_scores.detach().cpu().numpy()
                for i, sample in enumerate(sample_ids):
                    best_train_attn_list[sample] = attn_scores[i, :]

            loss = loss_fn(preds, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss.append(loss.detach().cpu().numpy())
            if (cur_step+1) % self.log_interval == 0:
                print(f"[{cur_step+1}] loss: {np.mean(running_loss):.3f}", flush=True)
                running_loss = []
            if (cur_step+1) % self.eval_interval == 0:
                running_loss = []
                print("----------------Validating----------------", flush=True)
                val_scores, val_attn_list, val_predictions = self.evaluate(model, ggi_graph, val_loader, metric_funcs)
                if val_scores['auroc'] > best_val_scores['auroc']:
                    best_val_scores = val_scores
                    best_val_predictions = val_predictions
                    # Store best model
                    best_model_state = model.state_dict()
                    ## Best model, add the attentions
                    ## VAL
                    best_val_attn_list = val_attn_list
                    if test_loader is not None:
                        best_test_scores, best_test_attn_list, test_predictions = self.evaluate(model, ggi_graph, test_loader, metric_funcs)
                    cur_early_stop = 0
                else:
                    cur_early_stop += 1
                    print(f"Early stop {cur_early_stop}/{self.n_early_stop}")
                    if cur_early_stop == self.n_early_stop: break
                print(f"[{cur_step+1}] cur_val_score: {val_scores}, best_val_score: {best_val_scores}, test_score: {best_test_scores}", flush=True)
                print("----------------Training----------------", flush=True)
            if cur_step == self.n_steps: break
        return best_val_scores, best_test_scores, best_train_attn_list, best_val_attn_list, best_test_attn_list, best_model_state, best_val_predictions, test_predictions
        
    def evaluate(self, model, ggi_graph, test_loader, metric_funcs):
        with torch.no_grad():
            model.eval()
            preds_list, labels_list, ancestrys_list = [], [], []
            attn_list = {}
            sample_id_dict = {}
            for batch in test_loader:
                if self.multiple_ancestries:
                    labels, preds, ph_attn_scores, anc_attn_scores = self.forward_batch_ma(model, ggi_graph, batch)
                else:
                    labels, preds, attn_scores, sample_ids = self.forward_batch(model, ggi_graph, batch)
                preds_list.append(preds.detach())
                labels_list.append(labels.detach())
                attn_scores = attn_scores.detach().cpu().numpy()
                for i, sample in enumerate(sample_ids):
                    attn_list[sample] = attn_scores[i, :]
                    sample_id_dict[sample] = {
                        'pred': torch.sigmoid(preds[i]).detach().cpu().numpy().item(),
                        'label': labels[i].detach().cpu().numpy().item()
                    }

            preds = torch.cat(preds_list).reshape(-1)
            labels = torch.cat(labels_list).reshape(-1)

            metric_results = {name: func(preds, labels.int()).item() for name, func in metric_funcs.items()}

            return metric_results, attn_list, sample_id_dict