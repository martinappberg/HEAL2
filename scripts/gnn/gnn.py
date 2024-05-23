import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import argparse
import warnings
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics import AUROC, AveragePrecision
import dgl
import pandas as pd
import numpy as np
import json
warnings.filterwarnings("ignore")

from src.gnn.dataset import Dataset
from src.gnn.utils import seed_worker, collate_fn, ancestry_encoding, set_random_seed
from src.gnn.utils import logo_splits, sk_splits, stratified_k_fold_splits, create_dir_if_not_exists
from src.gnn.model import PRSNet
from src.gnn.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training GNN")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--af", type=float, default=1.0)
    parser.add_argument("--exclude", type=str, default=None)
    parser.add_argument("--cohort", type=str, default="full")
    parser.add_argument("--logo", action="store_true")
    parser.add_argument("--stratified_kfold", action="store_true")
    parser.add_argument("--test_group", type=str, default=None)
    parser.add_argument("--shuffle_controls", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("-rs", "--random_state", type=int, default=42)
    parser.add_argument("-o", "--output", type=str, default=".")
    parser.add_argument('--pop', type=str, default=None, help='Which population to filter for (eg. EUR, EAS, etc.)')
    parser.add_argument('--pop_file', type=str, default=None, help='Population file')
    parser.add_argument('--pop_threshold', type=float, help='Population threshold', default=0.85)
    args = parser.parse_args()
    return args

def store_predictions(predictions_dict, output_dir, file_name):
    # Create a DataFrame from the predictions dictionary
    df = pd.DataFrame.from_dict(predictions_dict, orient='index')
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(output_dir, file_name))

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(22)
    ## Data loading and splits generating
    # Load the DataFrame
    info_df = pd.read_csv(f'{args.data_path}/{args.dataset}/info.csv')
    non_pop_samples = None

    ## POP filter if we should
    if args.pop is not None:
        assert args.pop_file is not None, "Error: --pop_file must be specified if --pop is provided."
        print(f"Will begin pop-filter of {info_df.shape[0]} samples for {args.pop} ≥ {args.pop_threshold}")
        pop = pd.read_csv(args.pop_file)
        pop = pop[pop[args.pop].astype(float) >= args.pop_threshold].copy()
        pop.loc[:, 'IID'] = pop['FID'].astype(str).str.cat(pop['SID'].astype(str), sep='_')
        pop = pop.set_index('IID', drop=True)
        non_pop_samples = info_df[~info_df['sample_id'].isin(pop.index)]
        info_df = info_df[info_df['sample_id'].isin(pop.index)].reset_index(drop=True)
        print(f"Population filtered -> {info_df.shape[0]} samples remain, {non_pop_samples.shape[0]} filtered out")

    # Read gene to index
    json_data = open(f'{args.data_path}/gene_to_index_{args.af}.json', 'r')
    gene_to_index = json.load(json_data)
    gti_arr = [a for a in gene_to_index.values()]

    # Set the 'group' column based on the first character of 'sample_id'
    info_df['group'] = info_df['sample_id'].str[0]

    # Reassign 'I' cases to 'E'
    info_df.loc[(info_df['group'] == 'I') & (info_df['label'] == 1), 'group'] = 'E'

    if args.exclude is not None:
        exclude_drop = info_df[info_df['group'] == args.exclude]
        info_df = info_df.drop(exclude_drop.index)
        print(f"Excluded {exclude_drop.shape[0]} samples from cohort {args.exclude}")

    if args.shuffle_controls:
        # Remove control samples from the original DataFrame to avoid duplication
        case_df = info_df[info_df['label'] == 1]
        case_cohorts = case_df['group'].unique().tolist()

        # Shuffle and handle controls separately
        controls_df = info_df[info_df['label'] == 0].sample(frac=1, random_state=args.random_state).reset_index(drop=True)

        # Calculate the total number of cases in E, F, U for proportion
        total_cases = case_df['group'].isin(case_cohorts).sum()
        total_controls = controls_df.shape[0]

        # Assign controls to case cohorts E, F, U proportionally
        for cohort in case_cohorts:
            cohort_cases = case_df[case_df['group'] == cohort].shape[0]
            proportion_controls = int(total_controls * (cohort_cases / total_cases))
            assigned_controls = controls_df[:proportion_controls]
            controls_df = controls_df[proportion_controls:]
            assigned_controls['group'] = cohort
            case_df = pd.concat([case_df, assigned_controls])

        # Handle any remaining controls by assigning them to the groups with the highest case counts
        while not controls_df.empty:
            for cohort in case_cohorts:
                if controls_df.empty:
                    break
                control = controls_df.iloc[0:1]
                control['group'] = cohort
                case_df = pd.concat([case_df, control])
                controls_df = controls_df.iloc[1:].reset_index(drop=True)

        info_df = case_df
    
    if args.cohort is not "full":
        info_df = info_df[~((info_df['group'] != args.cohort) & (info_df['label'] == 1))]
        if args.bootstrap and not args.logo:
            bootstrap_controls = info_df[info_df['label'] == 0].sample(frac=1, random_state=args.random_state).reset_index(drop=True)
            info_df = info_df[info_df['label'] == 1]
            info_df = pd.concat([info_df, bootstrap_controls.head(info_df.shape[0])])

    assert info_df['sample_id'].nunique() == len(info_df), "Duplicated sample IDs found."
    

    sample_ids = info_df['sample_id'].values

    labels = torch.from_numpy(info_df['label'].values)
    groups = info_df['group'].values
    ancestries = ancestry_encoding(info_df['ancestry'].values)
    #splits = generate_splits(labels)

    if args.logo:
        splits = logo_splits(labels, groups, random_state=args.random_state)
    elif args.stratified_kfold is not None:
        if args.test_group is not None:
            test_indices = info_df[info_df['group'] == args.test_group].index.values
            splits = stratified_k_fold_splits(labels, random_state=args.random_state, test_indices=test_indices, test_group=args.test_group)
        else:
            splits = stratified_k_fold_splits(labels, random_state=args.random_state)
    else:
        splits = sk_splits(labels, random_state=args.random_state)
    

    ggi_graph = dgl.load_graphs(f'{args.data_path}/ggi_graph_{args.af}.bin')[0][0]
    num_nodes = ggi_graph.number_of_nodes()
    print("Number of nodes:", num_nodes)

    ## Device
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    trainer = Trainer(device=device, log_interval=20, n_early_stop=20)

    filename = f"{args.output}/prsnet_output_{args.cohort}cohort_{args.shuffle_controls}shufflecontrols_{args.bootstrap}bootstrap_{args.logo}logo_{args.af}af_exc{args.exclude}.csv"

    case_control_counts = info_df.groupby(['group', 'label']).size().unstack(fill_value=0)
    print("\nCase and Control counts per group:")
    print(case_control_counts)

    train_size = 16
    learning_rate = 1e-4 * (train_size / 256)
    weight_decay = 1e-6
    features = 92
    n_layers = 1

    ## Validation
    for split_id, (train_ids, val_ids, test_ids, train_groups, val_groups, test_groups) in enumerate(splits):

        assert len(set(train_ids) & set(val_ids)) == 0, "Overlap found between training and validation sets"
        assert len(set(train_ids) & set(test_ids)) == 0, "Overlap found between training and test sets"
        assert len(set(val_ids) & set(test_ids)) == 0, "Overlap found between validation and test sets"

        print(f"\n\nSplit {split_id} --> Training groups: {train_groups} | Validation groups: {val_groups} | Test groups: {test_groups}")
        train_set = Dataset(args.data_path, args.dataset, sample_ids=sample_ids[train_ids],labels=labels[train_ids], balanced_sampling=True)
        val_set = Dataset(args.data_path, args.dataset, sample_ids=sample_ids[val_ids],labels=labels[val_ids], balanced_sampling=False)
        test_set = Dataset(args.data_path, args.dataset, sample_ids=sample_ids[test_ids],labels=labels[test_ids], balanced_sampling=False)

        train_loader = DataLoader(train_set, batch_size=int(train_size), shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, drop_last=False, pin_memory=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=int(train_size), shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker, drop_last=False, pin_memory=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=int(train_size), shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker, drop_last=False, pin_memory=True, collate_fn=collate_fn)

        model = PRSNet(n_genes=num_nodes, n_layers=n_layers, d_input=features).to(device)
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        metric_auroc = AUROC(task='binary')
        metric_auprc = AveragePrecision(task='binary')
        metric_funcs = {
            'auroc': metric_auroc,
            'auprc': metric_auprc,
        }
        
        best_val_scores, best_test_scores, _, _, _, _, val_predictions, test_predictions = trainer.train_and_test(model, ggi_graph, loss_fn, optimizer, metric_funcs, train_loader, val_loader, test_loader)
        print(f"----------------Split {split_id} final result----------------", flush=True)
        print(f"Training groups: {train_groups} | Validation groups: {val_groups} | Test groups: {test_groups}")
        print(f"best_val_score: {best_val_scores}, best_test_score: {best_test_scores}")

        if not args.silent:
            ## Store everything in a results_df and attention scores
            ## Create output directory
            create_dir_if_not_exists(args.output)
            create_dir_if_not_exists(f"{args.output}/val_predictions")
            create_dir_if_not_exists(f"{args.output}/test_predictions")

            # Store validation predictions
            store_predictions(val_predictions, f"{args.output}/val_predictions", f"{args.random_state}rs_split{split_id}_val_predictions.csv")

            # Store test predictions
            store_predictions(test_predictions, f"{args.output}/test_predictions", f"{args.random_state}rs_split{split_id}_test_predictions.csv")

            # Combine additional split details
            results_data = {
                "Split ID": [split_id],
                "Random State": [args.random_state],
                "Train groups": [str(train_groups)],
                "Validation groups": [str(val_groups)],
                "Test groups": [str(test_groups)],
            }
            # Add best validation scores to results data
            for metric_name, score in best_val_scores.items():
                results_data[f"Validation ({metric_name})"] = [score]

            # Add best test scores to results data
            for metric_name, score in best_test_scores.items():
                results_data[f"Test ({metric_name})"] = [score]

            results_df = pd.DataFrame(results_data)

            # Append to CSV with header written only once
            results_df.to_csv(filename, mode='a', header=False, index=False)

        
