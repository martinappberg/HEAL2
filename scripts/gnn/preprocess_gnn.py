import pandas as pd
import numpy as np

import dgl
import json

import torch

import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Additionally Preprocess data for GNN')
    parser.add_argument('--af', type=float, default=0.05, help='Allele frequency threshold')
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.path.dirname(__file__), '../../gnn_data'), help='Output directory')
    parser.add_argument('--ggi', type=str, default=os.path.join(os.path.dirname(__file__), '../files/ggi_no_score.csv'), help='Gene-gene interaction file')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input file')
    args = parser.parse_args()
    af = args.af
    output = args.output

    os.makedirs(f'{output}/af_{af}', exist_ok=True)
    os.makedirs(f'{output}/af_{af}/feats', exist_ok=True)

    ggi = pd.read_csv(args.ggi, sep='\t', header=None)
    mutational_burden = pd.read_csv(args.input, sep='\t', index_col=[0, 1])

    gene_list = mutational_burden.index.get_level_values(level=1).unique().tolist()

    # Remove rows in the ggi dataframe that are not in the gene_list
    print(ggi.shape)
    ggi = ggi[(ggi[0].isin(gene_list)) & (ggi[1].isin(gene_list))]
    print(ggi.shape)


    # Create a list of all the edges from the ggi dataframe with the score as the weight
    # Index of the gene in the gene_list is used as the node number
    # Create a mapping from gene names to indices
    gene_to_index = {gene: idx for idx, gene in enumerate(gene_list)}

    # Use the mapping to get indices for the edges
    edges = [[gene_to_index[row[1]], gene_to_index[row[2]]] for row in ggi.itertuples()]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()


    g = dgl.graph((edge_index[0], edge_index[1]))
    g = g.add_self_loop()


    index_to_gene = {idx: gene for gene, idx in gene_to_index.items()}
    with open(f'{output}/gene_to_index_{af}.json', 'w') as f:
        json.dump(index_to_gene, f)
    dgl.save_graphs(f'{output}/ggi_graph_{af}.bin', g)


    # Create the info.csv file
    infocsv = mutational_burden.iloc[:, -1:].copy()
    infocsv = infocsv.reset_index()
    infocsv = infocsv.rename(columns={'index': 'sample_id', 'case': 'label'})
    infocsv['ancestry'] = 'EUR'


    # store the data
    for i, row in mutational_burden.iterrows():
        arr = row[:-1].values
        arr = arr.astype('float32')
        np.save(f'{output}/af_{af}/feats/{i}.npy', np.array([arr]))
    infocsv.to_csv(f'{output}/af_{af}/info.csv', index=False)


if __name__ == '__main__':
    main()

