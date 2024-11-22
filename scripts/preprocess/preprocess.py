import pandas as pd
import numpy as np
import argparse
import os

gnomad_constraints = pd.read_csv(os.path.join(os.path.dirname(__file__), '../files/gnomad_pli.tsv'), sep='\t')
gnomad_constraints.drop_duplicates(subset='gene', keep='first', inplace=True)
lofs = ['stopgain', 'stoploss', 'startloss', 'startgain', 'frameshift_insertion', 'frameshift_deletion']
def lof_score(feat):
    score = 1.0
    filtered = gnomad_constraints[gnomad_constraints['gene'] == feat]
    if not filtered.empty:  # Check if the filtered DataFrame is not empty
        pLI = filtered.iloc[0]['lof.pLI']
        if pLI >= 0:
            score = pLI  # Set score if a match is found
    else:
        print(f"No PLI for {feat}")
    return score

def haplotype_to_numeric(hap, equal_allele_weights):
    if hap == '1/1':
        if equal_allele_weights:
            return 1
        return 2
    elif hap in ['0/1', '1/0']:
        return 1
    elif hap == '0/0':
        return 0
    else:
        #print(f"unexpected: {hap} will return 0")
        return 0

def main():

    # Create the parser
    parser = argparse.ArgumentParser(description='Preprocess Exonic ANNOVAR for HEAL')
    # Add the float argument
    parser.add_argument('--AF', type=float, required=True, help='Allele Frequency as a float. Variants above this threshold in gnomAD exome + genome will be excluded.')
    # Add the boolean argument. This will be False by default, and True if --indels is specified on the command line
    parser.add_argument('--indels', action='store_true', help='Flag to indicate if indels are included or not (default False)')

    # Input file
    parser.add_argument('-f', '--file', type=str, required=True, help='Input file to preprocess')
    parser.add_argument('-n', '--n_samples', type=int, required=True, help='Number of samples')

    # Output dir
    parser.add_argument('-o', '--output', type=str, required=True, help='Where to put the output')

    # gnomAD_filter
    parser.add_argument('--gnomad_ancestry', type=str, required=False, default=None, help='gnomAD ancestry to filter (default standard)')

    # special filter
    parser.add_argument('--rare_gnomad_common_controls', action='store_true', help='Rare in gnomAD but common in controls')
    parser.add_argument('--common_gnomad_rare_cases', action='store_true', help='Common in gnomAD, rare in cases')
    parser.add_argument('--filter_cohort_U', type=float, default=0.0, help='Cutoff for variants in U cohort')

    # Additional arguments
    parser.add_argument('--equal_allele_weights', action='store_true', help='Equal alelle weights for heterozygous and homozygous variants')
    parser.add_argument('--multidimensional', action='store_true', help='Include all scores, both sum and max for GNN processing')

    # Parse the arguments
    args = parser.parse_args()

    gnomAD_AF = args.AF
    indels = args.indels

    n_samples = args.n_samples

    path = args.file
    output_dir = args.output

    combined = pd.read_csv(path, index_col=0, low_memory=False)
    start_variants = combined.shape[0]
    print(f"Starting with a total of {start_variants} variants..")

    # Fix those that have ;
    combined['Gene.refGene'] = combined['Gene.refGene'].str.replace('\\x3b', ';')
    combined['Gene.refGene'] = combined['Gene.refGene'].str.split(';').str[0]

    # Explode variants that are ,
    combined['Gene.refGene'] = combined['Gene.refGene'].str.split(',')
    combined = combined.explode('Gene.refGene')

    if indels:
        if not args.multidimensional:
            # Add LoF scores
            combined.loc[:, 'REVEL_score'] = combined.apply(lambda x: lof_score(x['Gene.refGene']) if (x['REVEL_score'] == '.' and x['ExonicFunc.refGene'] in lofs) else x['REVEL_score'], axis=1)
        else:
            # Add LoF scores to everything
            combined['temp_pLI_score'] = combined.apply(lambda x: lof_score(x['Gene.refGene']) if x['ExonicFunc.refGene'] in lofs else 0.0, axis=1)
            insert_index = combined.shape[1] - n_samples - 1
            combined.insert(loc=insert_index, column='pLI_score', value=combined['temp_pLI_score'])
            combined.drop(columns=['temp_pLI_score'], inplace=True)
    else:
        combined = combined[combined['ExonicFunc.refGene'] == 'nonsynonymous_SNV']

    only_revel = combined if args.multidimensional else combined[combined['REVEL_score'] != '.']

    print(f"Filtered out {start_variants - only_revel.shape[0]} variants that did not have REVEL or pLI score")

    gnomad_group_exome = 'gnomad40_exome_AF'
    gnomad_group_genome = 'gnomad40_genome_AF'
    if args.gnomad_ancestry is not None:
        gnomad_group_exome += f'_{args.gnomad_ancestry}'
        gnomad_group_genome += f'_{args.gnomad_ancestry}'

    only_revel[gnomad_group_genome] = pd.to_numeric(only_revel[gnomad_group_genome], errors='coerce').fillna(0.0)
    only_revel[gnomad_group_exome] = pd.to_numeric(only_revel[gnomad_group_exome], errors='coerce').fillna(0.0)
    only_revel[gnomad_group_exome] = np.where(
        (only_revel[gnomad_group_exome] == 0) & (only_revel[gnomad_group_genome] != 0),
        only_revel[gnomad_group_genome],
        only_revel[gnomad_group_exome]
    )
    if gnomAD_AF != 0.0 and gnomAD_AF != 1.0:
        only_revel_rare = only_revel[only_revel[gnomad_group_exome] < gnomAD_AF]
    else:
        non_zero = only_revel[only_revel[gnomad_group_exome] > gnomAD_AF]
        only_revel_rare = only_revel.drop(non_zero.index)

    if args.rare_gnomad_common_controls:
        rare_gnomad_common_controls = only_revel_rare[(only_revel_rare[gnomad_group_exome] < 0.01) & (only_revel_rare['sample_AF_controls'] >= 0.01)]
        only_revel_rare.drop(rare_gnomad_common_controls.index, inplace=True)
        print(f"Filtered out {rare_gnomad_common_controls.shape[0]} rare gnomad common controls")

    if args.common_gnomad_rare_cases:
        common_gnomad_rare_cases = only_revel_rare[(only_revel_rare[gnomad_group_exome] >= 0.01) & (only_revel_rare['sample_AF_cases'] < 0.01)]
        only_revel_rare.drop(common_gnomad_rare_cases.index, inplace=True)
        print(f"Filtered out {common_gnomad_rare_cases.shape[0]} common gnomad rare cases")

    if args.filter_cohort_U > 0.0:
        too_common_in_U = only_revel_rare[only_revel_rare['cohort_U_af'] >= args.filter_cohort_U]
        only_revel_rare.drop(too_common_in_U.index, inplace=True)
        print(f"Filtered out {too_common_in_U.shape[0]} variants with AF ≥{args.filter_cohort_U} in U")


    only_revel_rare = only_revel_rare.reset_index()

    print(f"Filtered out {only_revel.shape[0] - only_revel_rare.shape[0]} variants with an AF ≥ {gnomAD_AF}")

    # Make to mutmatrix
    anno_complete = only_revel_rare.iloc[:,:-n_samples]
    geno_complete = only_revel_rare.iloc[:,-n_samples:]

    geno_complete = geno_complete.apply(lambda x: x.map(lambda y: haplotype_to_numeric(y, args.equal_allele_weights)))
    geno_complete_numeric = geno_complete.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    
    mutmatrix = None
    if args.multidimensional:
        gnn_scores = ['SIFT_converted_rankscore', 'SIFT4G_converted_rankscore', 'Polyphen2_HDIV_rankscore',
                      'Polyphen2_HVAR_rankscore', 'LRT_converted_rankscore', 'MutationTaster_converted_rankscore',
                      'MutationAssessor_rankscore', 'FATHMM_converted_rankscore', 'PROVEAN_converted_rankscore',
                      'VEST4_rankscore', 'MetaSVM_rankscore', 'MetaLR_rankscore', 'MetaRNN_rankscore',
                      'M-CAP_rankscore', 'REVEL_rankscore', 'MutPred_rankscore', 'MVP_rankscore', 'MPC_rankscore',
                        'PrimateAI_rankscore', 'DEOGEN2_rankscore', 'BayesDel_addAF_rankscore', 'BayesDel_noAF_rankscore',
                        'ClinPred_rankscore', 'LIST-S2_rankscore', 'CADD_raw_rankscore', 'DANN_rankscore', 'fathmm-MKL_coding_rankscore',
                        'fathmm-XF_coding_rankscore', 'Eigen-raw_coding_rankscore', 'Eigen-PC-raw_coding_rankscore', 'GenoCanyon_rankscore',
                        'integrated_fitCons_rankscore', 'LINSIGHT_rankscore', 'GERP++_RS_rankscore', 'phyloP100way_vertebrate_rankscore',
                        'phyloP30way_mammalian_rankscore', 'phastCons100way_vertebrate_rankscore', 'phastCons30way_mammalian_rankscore', 'pLI_score']
        
        unique_genes = anno_complete['Gene.refGene'].unique()
        score_columns = [f"{score}_sum" for score in gnn_scores] + [f"{score}_max" for score in gnn_scores]
        mutmatrix = pd.DataFrame(index=pd.MultiIndex.from_product([geno_complete.columns, score_columns], names=['sample', 'scores']), columns=unique_genes, dtype=np.float32, data=0.0)
        print(f"Allocated big mutmatrix of shape: {mutmatrix.shape}")
        completed_samples = 0
        for sample in geno_complete.T.iterrows():
            #indices = sample[1][sample[1] != 0].index
            nonzero_variants = sample[1][sample[1] != 0].index
            product = sample[1].loc[nonzero_variants].values[:, np.newaxis] * pd.to_numeric(anno_complete.loc[nonzero_variants,gnn_scores].stack(), errors='coerce').unstack(fill_value=0.0).fillna(0.0)
            product['Gene.refGene'] = anno_complete['Gene.refGene']
            nonzeros = product[product[gnn_scores].sum(axis=1) != 0]
            sums = nonzeros.groupby('Gene.refGene').sum()
            maxs = nonzeros.groupby('Gene.refGene').max()
            sums.columns = [f'{col}_sum' for col in sums.columns]
            maxs.columns = [f'{col}_max' for col in maxs.columns]
            concatenated = pd.concat([sums, maxs], axis=1)
            nonzero_genes = mutmatrix.columns.intersection(concatenated.index)
            mutmatrix.loc[(sample[0], slice(None)), nonzero_genes] = concatenated.T[nonzero_genes].values.astype(np.float32)
            completed_samples += 1
            print(f"Completed samples: {completed_samples} / {geno_complete.shape[1]}")
    else:
        pivot_table = anno_complete.pivot_table(index=anno_complete.index, columns='Gene.refGene', values='REVEL_score', aggfunc='first')
        pivot_table_numeric = pivot_table.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
        mutmatrix = np.dot(geno_complete_numeric.T, pivot_table_numeric)
        mutmatrix = pd.DataFrame(mutmatrix, 
                                        index=geno_complete.columns, 
                                        columns=pivot_table.columns)
    mutmatrix.index.name = 'sample'
    mutmatrix.columns.name = 'scores'
        
    mutmatrix = mutmatrix.astype(pd.SparseDtype("float", 0.0))
    
    phenos = pd.read_csv('mecfs.pheno.new.txt', sep='\t')
    phenos = phenos.set_index('FID', drop=True)
    phenos = phenos.iloc[:,-1:]
    phenos = phenos.rename(columns={'mecfs': 'case'})
    merged = pd.merge(mutmatrix, phenos, left_on='sample', right_index=True)
    # Drop samples that do not have phenotype information
    merged_filtered = merged[merged["case"] != 0]
    print(f"Excluding {merged.shape[0] - merged_filtered.shape[0]} samples without phenotype information")

    # Redo scoring
    merged_filtered.loc[:, 'case'] = merged_filtered['case'].replace({1: 0, 2: 1})
    file_name = path.split('/')[-1]  # Get the last component of the path
    base_name = file_name.split('.')[0]  # Remove the file extension
    if args.gnomad_ancestry is not None:
        base_name += f'_{args.gnomad_ancestry}'
    if args.rare_gnomad_common_controls:
        base_name += '_raregnomadcommoncontrols'
    if args.common_gnomad_rare_cases:
        base_name += '_commongnomadrarecases'
    if args.filter_cohort_U > 0.0:
        base_name += f'_filtercohortU{args.filter_cohort_U}'
    if args.equal_allele_weights:
        base_name += '_equalalleleweights'
    if args.multidimensional:
        base_name += '_multidimensional'
    merged_filtered.to_csv(f'{output_dir}/{base_name}_REVEL_{gnomAD_AF}_gnomad_{indels}_indels.csv', sep='\t')

    print("Done")


if __name__ == '__main__':
    main()
