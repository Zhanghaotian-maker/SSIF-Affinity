import argparse
from SSF_Affinity.features import (
    process_node_features, 
    process_edge_features,
    process_sequence_features
)


def main():
    parser = argparse.ArgumentParser(description='Generate features for SSF-Affinity')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Node features parser
    node_parser = subparsers.add_parser('node', help='Generate node features')
    node_parser.add_argument('--input_dir', required=True, help='Input directory for CSV files')
    node_parser.add_argument('--output_dir', required=True, help='Output directory for node features')

    # Edge features parser
    edge_parser = subparsers.add_parser('edge', help='Generate edge features')
    edge_parser.add_argument('--input_dir', required=True, help='Input directory for JSON files')
    edge_parser.add_argument('--output_dir', required=True, help='Output directory for edge features')
    edge_parser.add_argument('--max_neighbors', type=int, default=10, help='Maximum number of neighbors')

    # Sequence features parser
    seq_parser = subparsers.add_parser('sequence', help='Generate sequence features')
    seq_parser.add_argument('--input_dir', required=True, help='Input directory for sequence TXT files')
    seq_parser.add_argument('--output_dir', required=True, help='Output directory for sequence features')

    args = parser.parse_args()

    if args.command == 'node':
        print("Generating node features...")
        process_node_features(args.input_dir, args.output_dir)
    elif args.command == 'edge':
        print("Generating edge features...")
        process_edge_features(
            input_dir=args.input_dir,
            output_dir=args.output_dir,  # 传递新参数
            max_neighbors=args.max_neighbors
        )
    elif args.command == 'sequence':
        print("Generating sequence features...")
        process_sequence_features(args.input_dir, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
    