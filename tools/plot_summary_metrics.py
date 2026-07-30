"""Plot per-segment performance metrics from a segment_metrics_summary.json."""
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def plot_performance_summary(json_path, output_dir=None):
    """Render position/velocity/per-axis metric bar charts to a summary PNG."""
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    segments = data.get('segment_metrics', [])
    if not segments:
        print("No segment metrics found in JSON")
        return

    # Extract metrics
    segment_ids = [s['segment_id'] for s in segments]
    ate = [s.get('ate', 0) for s in segments]
    t_rte = [s.get('t_rte', 0) for s in segments]
    d_rte = [s.get('d_rte', 0) for s in segments]
    ave = [s.get('AVE', 0) for s in segments]
    v_rmse = [s.get('V_RMSE', 0) for s in segments]
    x_ave = [s.get('X_AVE', 0) for s in segments]
    y_ave = [s.get('Y_AVE', 0) for s in segments]
    z_ave = [s.get('Z_AVE', 0) for s in segments]

    # Calculate averages
    metrics = {
        'ATE': ate, 'T_RTE': t_rte, 'D_RTE': d_rte,
        'AVE': ave, 'V_RMSE': v_rmse,
        'X_AVE': x_ave, 'Y_AVE': y_ave, 'Z_AVE': z_ave
    }
    averages = {k: np.mean(v) for k, v in metrics.items()}

    # Setup Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    plt.subplots_adjust(hspace=0.4)
    x = np.arange(len(segment_ids))
    width = 0.25

    # Panel 1: Position Metrics
    axes[0].bar(x - width, ate, width, label='ATE (m)', color='skyblue')
    axes[0].bar(x, t_rte, width, label='T_RTE (m/min)', color='salmon')
    axes[0].bar(x + width, d_rte, width, label='D_RTE (m/10m)', color='lightgreen')
    axes[0].set_title('Position Drift Metrics per Segment', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Error Value')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Seg {i}' for i in segment_ids])
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add average text
    avg_text = f"AVERAGES:\nATE: {averages['ATE']:.3f}m\nT_RTE: {averages['T_RTE']:.3f}m/min\nD_RTE: {averages['D_RTE']:.3f}m/10m"
    axes[0].text(1.02, 0.5, avg_text, transform=axes[0].transAxes, verticalalignment='center', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Panel 2: Overall Velocity Metrics
    axes[1].bar(x - width/2, ave, width, label='AVE (m/s)', color='orchid')
    axes[1].bar(x + width/2, v_rmse, width, label='V_RMSE (m/s)', color='gold')
    axes[1].set_title('Overall Body Velocity Error per Segment', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Velocity Error (m/s)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Seg {i}' for i in segment_ids])
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    avg_text_v = f"AVERAGES:\nAVE: {averages['AVE']:.3f} m/s\nV_RMSE: {averages['V_RMSE']:.3f} m/s"
    axes[1].text(1.02, 0.5, avg_text_v, transform=axes[1].transAxes, verticalalignment='center', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Panel 3: Axis-Specific Velocity Errors
    axes[2].bar(x - width, x_ave, width, label='X_AVE (Forward)', color='blue', alpha=0.6)
    axes[2].bar(x, y_ave, width, label='Y_AVE (Lateral)', color='red', alpha=0.6)
    axes[2].bar(x + width, z_ave, width, label='Z_AVE (Vertical)', color='green', alpha=0.6)
    axes[2].set_title('Axis-Specific Body Velocity Error (m/s)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Velocity Error (m/s)')
    axes[2].set_xlabel('Segment ID')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'Seg {i}' for i in segment_ids])
    axes[2].legend()
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    avg_text_axes = f"AVERAGES (m/s):\nX: {averages['X_AVE']:.3f}\nY: {averages['Y_AVE']:.3f}\nZ: {averages['Z_AVE']:.3f}"
    axes[2].text(1.02, 0.5, avg_text_axes, transform=axes[2].transAxes, verticalalignment='center', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Overall Figure setup
    traj_name = data.get('trajectory', 'Unknown Trajectory')
    fig.suptitle(f'Performance Summary: {traj_name}', fontsize=16, fontweight='bold', y=0.95)
    
    if output_dir is None:
        output_dir = os.path.dirname(json_path)
    
    save_path = os.path.join(output_dir, 'performance_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Summary figure saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python tools/plot_summary_metrics.py <segment_metrics_summary.json>"
        )
    plot_performance_summary(sys.argv[1])




