"""Visualize IMU vs ground-truth coordinate alignment (gyro, accel, gravity)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
import sys

def visualize_alignment(npz_path, out_dir):
    """Plot gyro, accel-vs-gravity, and world-frame accel comparisons to out_dir."""
    print(f"Analyzing {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    os.makedirs(out_dir, exist_ok=True)
    
    z_pelvis = data['z_pelvis']
    gt_pos = data['gt_translation']
    gt_ori = data['gt_orientation'] # xyzw
    ts = data['time']
    
    dt = np.diff(ts)
    
    acc = z_pelvis[:, 0:3]
    gyro = z_pelvis[:, 3:6]
    
    # 1. Calculate GT Angular Velocity in Body Frame
    r_gt = R.from_quat(gt_ori)
    dq = r_gt[:-1].inv() * r_gt[1:]
    gt_gyro = dq.as_rotvec() / dt[:, np.newaxis]
    
    # 2. Project Gravity into Body Frame (Assuming Z-up world)
    # If the SLAM world is Z-up, gravity is [0, 0, -9.81]
    # Accelerometer measures (a - g). Stationary: acc = -g = [0, 0, 9.81] in world.
    g_world = np.array([0, 0, 9.8105])
    exp_acc_body = r_gt.inv().apply(g_world)
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    
    # Subplot 1: Gyroscope Comparison
    axes[0].plot(ts[:-1], gt_gyro[:, 0], 'r', label='GT Gyro X', alpha=0.8)
    axes[0].plot(ts[:-1], gt_gyro[:, 1], 'g', label='GT Gyro Y', alpha=0.8)
    axes[0].plot(ts[:-1], gt_gyro[:, 2], 'b', label='GT Gyro Z', alpha=0.8)
    
    axes[0].plot(ts, gyro[:, 0], 'r--', label='IMU Gyro X', alpha=0.4)
    axes[0].plot(ts, gyro[:, 1], 'g--', label='IMU Gyro Y', alpha=0.4)
    axes[0].plot(ts, gyro[:, 2], 'b--', label='IMU Gyro Z', alpha=0.4)
    
    axes[0].set_title("Gyroscope Comparison (Solid: GT, Dashed: IMU Raw)")
    axes[0].set_ylabel("rad/s")
    axes[0].legend(ncol=2)
    axes[0].grid(True)
    
    # Subplot 2: Accelerometer Comparison (Gravity Alignment)
    axes[1].plot(ts, exp_acc_body[:, 0], 'r', label='Exp Gravity X', alpha=0.8)
    axes[1].plot(ts, exp_acc_body[:, 1], 'g', label='Exp Gravity Y', alpha=0.8)
    axes[1].plot(ts, exp_acc_body[:, 2], 'b', label='Exp Gravity Z', alpha=0.8)
    
    axes[1].plot(ts, acc[:, 0], 'r--', label='IMU Acc X', alpha=0.4)
    axes[1].plot(ts, acc[:, 1], 'g--', label='IMU Acc Y', alpha=0.4)
    axes[1].plot(ts, acc[:, 2], 'b--', label='IMU Acc Z', alpha=0.4)
    
    axes[1].set_title("Accelerometer vs Projected Gravity (Solid: Exp from GT, Dashed: IMU Raw)")
    axes[1].set_ylabel("m/s^2")
    axes[1].legend(ncol=2)
    axes[1].grid(True)
    
    # Subplot 3: World Acceleration
    # Differentiate GT position twice to get acceleration
    gt_vel = np.diff(gt_pos, axis=0) / dt[:, np.newaxis]
    gt_accel = np.diff(gt_vel, axis=0) / dt[1:, np.newaxis]
    
    # Transform IMU to world and subtract gravity (assuming Z-up)
    world_acc_imu = r_gt.apply(acc) - g_world
    
    axes[2].plot(ts[2:], gt_accel[:, 0], 'r', label='GT Accel X', alpha=0.8)
    axes[2].plot(ts[2:], gt_accel[:, 1], 'g', label='GT Accel Y', alpha=0.8)
    axes[2].plot(ts[2:], gt_accel[:, 2], 'b', label='GT Accel Z', alpha=0.8)
    
    axes[2].plot(ts, world_acc_imu[:, 0], 'r--', label='IMU World Acc X', alpha=0.3)
    axes[2].plot(ts, world_acc_imu[:, 1], 'g--', label='IMU World Acc Y', alpha=0.3)
    axes[2].plot(ts, world_acc_imu[:, 2], 'b--', label='IMU World Acc Z', alpha=0.3)
    
    axes[2].set_title("World Frame Acceleration (Solid: GT, Dashed: IMU Raw Transformed)")
    axes[2].set_ylabel("m/s^2")
    axes[2].legend(ncol=2)
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "coordinate_analysis.png"))
    print(f"Saved analysis plot to {os.path.join(out_dir, 'coordinate_analysis.png')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python tools/visualize_coord_alignment.py <trajectory.npz> [out_dir]"
        )
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "debug_plots/analysis"
    visualize_alignment(sys.argv[1], out_dir)




