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
        feats, labels = batch
        batched_graph = dgl.batch([ggi_graph]*len(labels)).to(self.device)
        feats, labels = feats.to(self.device), labels.to(self.device)
        outputs, attn_scores = model(batched_graph, feats)
        return labels, outputs, attn_scores
    def forward_batch_ma(self, model, ggi_graph, batch):
        feats, ancestries, labels = batch
        batched_graph = dgl.batch([ggi_graph]*len(labels)).to(self.device)
        feats, ancestries, labels = feats.to(self.device), ancestries.to(self.device), labels.to(self.device)
        outputs, ph_attn_scores, anc_attn_scores = model(batched_graph, feats, ancestries)
        return labels, outputs, ph_attn_scores, anc_attn_scores
    def train_and_test(self, model, ggi_graph, loss_fn, optimizer, metric, train_loader, val_loader, test_loader):
        best_val_score, best_test_score = 0, 0
        running_loss = []
        cur_early_stop = 0

        data_iter = iter(train_loader)
        next_batch = next(data_iter)
        next_batch = [ _.cuda(non_blocking=True) for _ in next_batch ]
        print("----------------Training----------------", flush=True)
        for cur_step in range(len(train_loader)):
            ## Forward pass
            model.train()
            optimizer.zero_grad()
            batch = next_batch 
            if cur_step + 1 != len(train_loader): 
                next_batch = next(data_iter)
                next_batch = [ _.cuda(non_blocking=True) for _ in next_batch]
            if self.multiple_ancestries:
                labels, preds, ph_attn_scores, anc_attn_scores = self.forward_batch_ma(model, ggi_graph, batch)
            else:
                labels, preds, attn_scores = self.forward_batch(model, ggi_graph, batch)

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
                val_score, val_attn_scores = self.evaluate(model, ggi_graph, val_loader, metric)
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_test_score, test_attn_scores = self.evaluate(model, ggi_graph, test_loader, metric)
                    cur_early_stop = 0
                else:
                    cur_early_stop += 1
                    print(f"Early stop {cur_early_stop}/{self.n_early_stop}")
                    if cur_early_stop == self.n_early_stop: break
                print(f"[{cur_step+1}] val_score: {best_val_score:.3f}, test_score: {best_test_score: .3f}", flush=True)
                print("----------------Training----------------", flush=True)
            if cur_step == self.n_steps: break
        return best_val_score, best_test_score, test_attn_scores
        
    def evaluate(self, model, ggi_graph, test_loader, metric):
        with torch.no_grad():
            model.eval()
            preds_list, labels_list, ancestrys_list = [], [], []
            for batch in test_loader:
                if self.multiple_ancestries:
                    labels, preds, ph_attn_scores, anc_attn_scores = self.forward_batch_ma(model, ggi_graph, batch)
                else:
                    labels, preds, attn_scores = self.forward_batch(model, ggi_graph, batch)
                preds_list.append(preds.detach())
                labels_list.append(labels.detach())
            preds = torch.cat(preds_list).reshape(-1)
            labels = torch.cat(labels_list).reshape(-1)
            return metric(preds, labels).item(), attn_scores
        
    def train_full_dataset(self, model, ggi_graph, loss_fn, optimizer, full_loader):
        print("----------------Training on Full Dataset----------------", flush=True)
        moving_avg_loss = []
        patience = 10
        min_delta = 0.001
        patience_counter = 0
        prev_avg_loss = None

        data_iter = iter(full_loader)
        next_batch = next(data_iter)
        next_batch = [ _.cuda(non_blocking=True) for _ in next_batch ]

        for cur_step in range(len(full_loader)):
            model.train()
            optimizer.zero_grad()
            batch = next_batch 

            if cur_step + 1 != len(full_loader): 
                next_batch = next(data_iter)
                next_batch = [ _.cuda(non_blocking=True) for _ in next_batch]

            labels, preds, attn_scores = self.forward_batch(model, ggi_graph, batch)

            loss = loss_fn(preds, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update moving average of the loss
            if len(moving_avg_loss) < 100:
                moving_avg_loss.append(loss.item())
            else:
                moving_avg_loss.pop(0)
                moving_avg_loss.append(loss.item())

            if (cur_step+1) % self.log_interval == 0:
                current_avg_loss = np.mean(moving_avg_loss)
                print(f"[{cur_step+1}] loss: {current_avg_loss:.3f}", flush=True)
                
                # Check for convergence
                if prev_avg_loss is not None and cur_step != 0 and abs(prev_avg_loss - current_avg_loss) < min_delta:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Loss convergence achieved, stopping training.")
                        break
                else:
                    patience_counter = 0
                
                prev_avg_loss = current_avg_loss

        return attn_scores