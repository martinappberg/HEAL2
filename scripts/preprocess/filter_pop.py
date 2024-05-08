import pandas as pd
import numpy as np

import argparse
import os


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Preprocess Exonic ANNOVAR for HEAL')
    # Add the float argument
    parser.add_argument('--pop', type=str, required=True, help='Which population to filter for (eg. EUR, EAS, etc.)')
    parser.add_argument('-f', '--file', type=str, required=True, help='Population file')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory of mutation matrices to fix')
    parser.add_argument('--threshold', type=float, required=False, help='Population threshold', default=0.85)

    # Parse the arguments
    args = parser.parse_args()

    # Read population file
    pop = pd.read_csv(args.file)
    pop = pop[pop[args.pop].astype(float) >= args.threshold].copy()

    # Convert the desired rows to strings before concatenation
    pop.loc[:, 'IID'] = pop['FID'].astype(str).str.cat(pop['SID'].astype(str), sep='_')

    # Display the modified DataFrame
    pop = pop.set_index('IID', drop=True)

    # Ensure the output directory exists
    output_dir = os.path.join(args.directory, 'pop_filtered')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all CSV files in the directory
    for file in os.listdir(args.directory):
        if file.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(args.directory, file)
            
            # Read the DataFrame
            df = pd.read_csv(file_path, index_col=0, sep='\t')

            # Filter the DataFrame
            filtered_df = df.loc[df.index.intersection(pop.index)]

            file_name, file_extension = os.path.splitext(file)  # Split the file name and the extension
            filtered_file_name = f"{file_name}_{args.threshold}{args.pop}filtered{file_extension}"  # Append '_filtered' before the extension
            filtered_file_path = os.path.join(output_dir, filtered_file_name)
            filtered_df.to_csv(filtered_file_path, index_label=["sample", "gene"], sep='\t')
            
            print(f"{file_name}\n{filtered_df.shape[0]} samples remain after excluding {df.shape[0] - filtered_df.shape[0]} samples from original {df.shape[0]} samples")

if __name__ == "__main__":
    main()