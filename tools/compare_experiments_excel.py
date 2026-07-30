"""Aggregate per-trajectory segment metrics across experiments into CSV/Excel."""
import os
import json
import pandas as pd
import numpy as np
import argparse

def get_all_metrics(base_dir, experiments=None):
    """
    Finds all segment_metrics_summary.json files and extracts their data.
    """
    if not experiments:
        experiments = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    all_data = []

    for exp in experiments:
        exp_path = os.path.join(base_dir, exp)
        if not os.path.exists(exp_path):
            continue

        for root, _, files in os.walk(exp_path):
            if 'segment_metrics_summary.json' in files:
                json_path = os.path.join(root, 'segment_metrics_summary.json')
                
                # Identify the trajectory name (bag name)
                # The structure is usually exp_result/transformer/EXP_NAME/BAG_NAME/network_input.npz/
                parts = root.split(os.sep)
                # Walk back to find the part after the experiment name
                try:
                    exp_idx = parts.index(exp)
                    traj_name = parts[exp_idx + 1]
                except (ValueError, IndexError):
                    traj_name = "unknown"

                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        segments = data.get('segment_metrics', [])
                        if not segments:
                            continue
                        
                        # Calculate mean for this specific bag in this specific experiment
                        metrics = {
                            'Trajectory': traj_name,
                            'Experiment': exp,
                            'ATE': np.mean([s.get('ate', 0) for s in segments]),
                            'T_RTE': np.mean([s.get('t_rte', 0) for s in segments]),
                            'D_RTE': np.mean([s.get('d_rte', 0) for s in segments]),
                            'AVE': np.mean([s.get('AVE', 0) for s in segments]),
                            'X_AVE': np.mean([s.get('X_AVE', 0) for s in segments]),
                            'Y_AVE': np.mean([s.get('Y_AVE', 0) for s in segments]),
                            'Z_AVE': np.mean([s.get('Z_AVE', 0) for s in segments]),
                        }
                        all_data.append(metrics)
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")

    return pd.DataFrame(all_data)

def main():
    """CLI entry: collect metrics, write CSV/Excel, and optionally log to WandB."""
    parser = argparse.ArgumentParser(description='Compare experiment results per trajectory.')
    parser.add_argument('experiments', nargs='*', help='List of experiment names.')
    parser.add_argument('--base_dir', default='./exp_result/transformer', help='Base directory')
    parser.add_argument('--output', default='per_trajectory_comparison', help='Output filename')
    parser.add_argument('--wandb', action='store_true', help='Upload to WandB')
    parser.add_argument('--project', default='humanoid-transformer-per-traj', help='WandB project')
    
    args = parser.parse_args()

    df = get_all_metrics(args.base_dir, args.experiments)

    if df.empty:
        print("No results found.")
        return

    # Sort by Trajectory then Experiment for easy side-by-side comparison
    df = df.sort_values(by=['Trajectory', 'Experiment'])

    # Save local files
    csv_path = os.path.join(args.base_dir, f"{args.output}.csv")
    excel_path = os.path.join(args.base_dir, f"{args.output}.xlsx")
    
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(excel_path, index=False)
        print(f"✅ Excel and CSV generated: {args.base_dir}/{args.output}")
    except Exception:
        print(f"✅ CSV generated: {csv_path}")

    # Print a snippet to terminal
    print("\n" + "="*100)
    print("PER-TRAJECTORY COMPARISON (Grouped by Bag)")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    if args.wandb:
        import wandb
        run = wandb.init(project=args.project, job_type="comparison", name="per_traj_summary")
        run.log({"per_trajectory_table": wandb.Table(dataframe=df)})
        run.finish()
        print(f"✅ Uploaded to WandB Project: {args.project}")

if __name__ == "__main__":
    main()
