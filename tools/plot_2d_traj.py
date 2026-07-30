"""Plot the X-Y ground-truth trajectory from a network_input.npz file."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

def plot_2d_traj(npz_path, out_dir):
    """Save a 2D (X-Y) ground-truth trajectory plot to out_dir."""
    data = np.load(npz_path, allow_pickle=True)
    gt_pos = data['gt_translation']
    
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 10))
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'b-', label='GT Trajectory')
    plt.scatter(gt_pos[0, 0], gt_pos[0, 1], color='g', label='Start', s=100)
    plt.scatter(gt_pos[-1, 0], gt_pos[-1, 1], color='r', label='End', s=100)
    
    plt.title(f"2D Trajectory (X-Y Plane): {os.path.basename(os.path.dirname(npz_path))}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(out_dir, "2d_trajectory.png"))
    print(f"Saved 2D trajectory to {os.path.join(out_dir, '2d_trajectory.png')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python tools/plot_2d_traj.py <trajectory.npz> [out_dir]")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "debug_plots/analysis"
    plot_2d_traj(sys.argv[1], out_dir)




