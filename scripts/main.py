import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Project Command Line Interface")
    subparsers = parser.add_subparsers(dest='script', help='Choose a script to run')

    # GNN related parsers
    gnn_parser = subparsers.add_parser('gnn', help='Train GNN model')
    gnn_parser.add_argument("--data_path", type=str, required=True)
    gnn_parser.add_argument("--dataset", type=str, required=True)
    gnn_parser.add_argument("--num_workers", type=int, default=4)
    gnn_parser.add_argument("--af", type=float, default=1.0)
    gnn_parser.add_argument("--exclude", type=str, default=None)
    gnn_parser.add_argument("--cohort", type=str, default="full")
    gnn_parser.add_argument("--logo", action="store_true")
    gnn_parser.add_argument("--shuffle_controls", action="store_true")
    gnn_parser.add_argument("--bootstrap", action="store_true")
    gnn_parser.add_argument("--silent", action="store_true")
    gnn_parser.add_argument("-rs", "--random_state", type=int, default=42)

    preprocess_gnn_parser = subparsers.add_parser('preprocess_gnn', help='Preprocess data for GNN')
    preprocess_gnn_parser.add_argument('--af', type=float, default=0.05)
    preprocess_gnn_parser.add_argument('--output', '-o', type=str, required=True)
    preprocess_gnn_parser.add_argument('--ggi', type=str, required=True)
    preprocess_gnn_parser.add_argument('--input', type=str, required=True)

    # Linear model parsers
    heal_parser = subparsers.add_parser('heal', help='Run HEAL model')
    heal_parser.add_argument('--file_path', type=str, required=True)
    heal_parser.add_argument('--output', type=str, default='')
    heal_parser.add_argument('--splits', type=int, default=5)
    heal_parser.add_argument('--trials', type=int, default=2)
    heal_parser.add_argument('--l1', type=float, default=1.0)
    heal_parser.add_argument('--l2', type=float, default=40.0)
    heal_parser.add_argument('--lfidelity', type=int, default=5)
    heal_parser.add_argument('--cohort', type=str, default='full')
    heal_parser.add_argument('--bootstrap', action='store_true')
    heal_parser.add_argument('--allcontrols', action='store_true')
    heal_parser.add_argument('-exc', '--exclude', type=str, default=None)
    heal_parser.add_argument('--tts', action='store_true')

    # General preprocessing parsers
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data from ANNOVAR')
    preprocess_parser.add_argument('--AF', type=float, required=True)
    preprocess_parser.add_argument('--indels', action='store_true')
    preprocess_parser.add_argument('-f', '--file', type=str, required=True)
    preprocess_parser.add_argument('-n', '--n_samples', type=int, required=True)
    preprocess_parser.add_argument('-o', '--output', type=str, required=True)
    preprocess_parser.add_argument('--gnomad_ancestry', type=str, default=None)
    preprocess_parser.add_argument('--rare_gnomad_common_controls', action='store_true')
    preprocess_parser.add_argument('--common_gnomad_rare_cases', action='store_true')
    preprocess_parser.add_argument('--filter_cohort_U', type=float, default=0.0)
    preprocess_parser.add_argument('--equal_allele_weights', action='store_true')
    preprocess_parser.add_argument('--gnn', action='store_true')

    filter_pop_parser = subparsers.add_parser('filter_pop', help='Filter preprocessed data for population')
    filter_pop_parser.add_argument('--pop', type=str, required=True)
    filter_pop_parser.add_argument('-f', '--file', type=str, required=True)
    filter_pop_parser.add_argument('-d', '--directory', type=str, required=True)
    filter_pop_parser.add_argument('--threshold', type=float, default=0.85)

    args = parser.parse_args()

    # Run the appropriate script with the provided arguments
    if args.script == 'gnn':
        subprocess.run(['python', 'scripts/gnn/gnn.py'] + sys.argv[2:])
    elif args.script == 'preprocess_gnn':
        subprocess.run(['python', 'scripts/gnn/preprocess_gnn.py'] + sys.argv[2:])
    elif args.script == 'heal':
        subprocess.run(['python', 'scripts/linear_model/heal.py'] + sys.argv[2:])
    elif args.script == 'preprocess':
        subprocess.run(['python', 'scripts/preprocess/preprocess.py'] + sys.argv[2:])
    elif args.script == 'filter_pop':
        subprocess.run(['python', 'scripts/preprocess/filter_pop.py'] + sys.argv[2:])
    else:
        parser.print_help()

if __name__ == "__main__":
    main()