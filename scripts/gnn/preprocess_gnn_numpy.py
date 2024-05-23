import pandas as pd
import numpy as np

import dgl
import json

import torch

import os
import argparse
import glob



def main():
    parser = argparse.ArgumentParser(description='Additionally Preprocess data for GNN')
    parser.add_argument('--af', type=float, default=0.05, help='Allele frequency threshold')
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.path.dirname(__file__), '../../gnn_data'), help='Output directory')
    parser.add_argument('--ggi', type=str, default=os.path.join(os.path.dirname(__file__), '../../files/ggi_no_score.csv'), help='Gene-gene interaction file')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input directory')
    parser.add_argument('--gene_list', type=str, required=True, help='Path to gene list')
    parser.add_argument('--pheno', type=str, required=True, help='Phenotype file')

    args = parser.parse_args()
    af = args.af
    output = args.output

    os.makedirs(f'{output}/af_{af}', exist_ok=True)
    os.makedirs(f'{output}/af_{af}/feats', exist_ok=True)

    ggi = pd.read_csv(args.ggi, sep='\t', header=None)

    ## Read the gene list
    gene_list = pd.read_csv(args.gene_list, header=None).squeeze().values

    # Remove rows in the ggi dataframe that are not in the gene_list
    print(ggi.shape)
    ggi = ggi[(ggi[0].isin(gene_list)) & (ggi[1].isin(gene_list))]
    print(ggi.shape)

    # Create a list of all the edges from the ggi dataframe with the score as the weight
    # Index of the gene in the gene_list is used as the node number
    # Create a mapping from gene names to indices
    gene_to_index = {gene: idx for idx, gene in enumerate(gene_list)}

    # Use the mapping to get indices for the edges
    edges = np.array([[gene_to_index[row[1]], gene_to_index[row[2]]] for row in ggi.itertuples()])
    self_loops = np.array([[idx, idx] for idx, _ in enumerate(gene_list)])
    edges = np.concatenate([self_loops, edges], axis=0)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    g = dgl.graph((edge_index[0], edge_index[1]))

    index_to_gene = {idx: gene for gene, idx in gene_to_index.items()}
    with open(f'{output}/gene_to_index_{af}.json', 'w') as f:
        json.dump(index_to_gene, f)
    dgl.save_graphs(f'{output}/ggi_graph_{af}.bin', g)

    ## READ FILES
    sample_list_file = glob.glob(f"{args.input}/*.txt")[0]
    scores_file = glob.glob(f"{args.input}/*.npy")[0]

    # Create the info.csv file
    samples = pd.read_csv(sample_list_file, header=None, index_col=0, names=['sample_id'])
    phenos = pd.read_csv(args.pheno, sep='\t')
    phenos = phenos.set_index('FID', drop=True)
    phenos = phenos.iloc[:,-1:]
    phenos = phenos.rename(columns={'mecfs': 'label'})
    merged = pd.merge(samples, phenos, left_on='sample_id', right_index=True)
    infocsv = merged[merged['label'] != 0]
    infocsv.loc[:, 'label'] = infocsv['label'].replace({1: 0, 2: 1})
    infocsv.reset_index(inplace=True)
    infocsv['ancestry'] = 'EUR'

    # Load the scores
    scores = np.load(scores_file)
    for idx, s in infocsv['sample_id'].iteritems():
        arr = scores[idx, :, :].flatten()
        arr = arr.astype('float32')
        np.save(f'{output}/af_{af}/feats/{s}.npy', arr)
    infocsv.to_csv(f'{output}/af_{af}/info.csv', index=False)

if __name__ == '__main__':
    main()
