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

    # Remove rows in ggi that are not in gene_list
    ggi = ggi[(ggi[0].isin(gene_list)) & (ggi[1].isin(gene_list))]

    # Identify genes in ggi
    genes_in_ggi = set(ggi[0]).union(set(ggi[1]))
    print("Number of genes in filtered ggi:", len(genes_in_ggi))

    # Filter gene_list and get indices of genes not in ggi
    indices_not_in_ggi = [i for i, gene in enumerate(gene_list) if gene not in genes_in_ggi]
    print("Indices of genes not in ggi:", len(indices_not_in_ggi))

    # Create a mapping from the full gene list
    gene_to_index = {gene: idx for idx, gene in enumerate(gene_list)}
    print("Gene to index mapping size:", len(gene_to_index))

    # Use the mapping to get indices for the edges from the ggi dataframe
    edges = np.array([[gene_to_index[row[1]], gene_to_index[row[2]]] for row in ggi.itertuples()])
    print("Edges shape before self-loops:", edges.shape)

    # Create self-loops for all nodes in gene_list
    self_loops = np.array([[idx, idx] for idx, _ in enumerate(gene_list)])
    print("Self-loops shape:", self_loops.shape)

    # Concatenate self-loops with edges
    edges = np.concatenate([self_loops, edges], axis=0)
    print("Edges shape after adding self-loops:", edges.shape)

    # Create edge_index tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print("Edge index shape:", edge_index.shape)
    print("Unique nodes in edge index:", edge_index.unique().shape)

    # Create the graph with all nodes from gene_list
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=len(gene_list))
    print("Graph nodes:", g.num_nodes())
    print("Graph edges:", g.num_edges())

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
