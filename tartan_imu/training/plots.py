# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Standalone plotting helpers for the training pipeline.

These functions render/save the per-epoch loss curves and the online-adaptation
/ offline-finetune summary figures. They were extracted verbatim from the
``Trainer`` class in ``train.py`` (which now delegates to them); the only change
is that the three former instance attributes (cfg, optimizer, out_dir) are now
explicit parameters instead of being read off the Trainer instance.
"""

from os import path as osp

import numpy as np

from tartan_imu.utils.rich_logging import info, success, warning


def plot_epoch_loss_results(
    cfg: dict,
    optimizer,
    out_dir: str,
    test_loader,
    val_loader,
    epoch_train_loss,
    epoch_train_mse,
    epoch_val_loss,
    epoch_val_mse,
    epoch_test_loss,
    epoch_test_mse,
):
    """Save and plot per-epoch train/val/test loss and MSE curves."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Create figure with enhanced styling
    fig = plt.figure(num="Training Loss Analysis", dpi=150, figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1])

    # Main loss plot
    ax_main = fig.add_subplot(gs[0, 0])

    # Prepare data
    train_loss = np.array(epoch_train_loss)
    train_mse = np.array(epoch_train_mse)
    epochs = range(1, len(train_loss) + 1)

    # Enhanced plotting with better colors and styling
    colors = {
        "train": "#1f77b4",  # Blue
        "val": "#d62728",  # Red
        "test": "#2ca02c",  # Green
    }

    # Plot training loss
    ax_main.plot(
        epochs,
        train_loss,
        color=colors["train"],
        linewidth=2.5,
        label="Training Loss",
        marker="o",
        markersize=4,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )

    loss_all = train_loss[:, np.newaxis]
    mse_all = train_mse[:, np.newaxis]

    # Plot validation loss if available
    if val_loader is not None:
        val_loss = np.array(epoch_val_loss)
        val_mse = np.array(epoch_val_mse)
        ax_main.plot(
            epochs,
            val_loss,
            color=colors["val"],
            linewidth=2.5,
            label="Validation Loss",
            marker="s",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
        loss_all = np.concatenate((loss_all, val_loss[:, np.newaxis]), axis=1)
        mse_all = np.concatenate((mse_all, val_mse[:, np.newaxis]), axis=1)

    # Plot test loss if available
    if test_loader is not None:
        test_loss = np.array(epoch_test_loss)
        test_mse = np.array(epoch_test_mse)
        ax_main.plot(
            epochs,
            test_loss,
            color=colors["test"],
            linewidth=2.5,
            label="Test Loss",
            marker="^",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
        loss_all = np.concatenate((loss_all, test_loss[:, np.newaxis]), axis=1)
        mse_all = np.concatenate((mse_all, test_mse[:, np.newaxis]), axis=1)

    # Enhanced styling for main plot
    ax_main.set_xlabel("Epoch", fontsize=14, fontweight="bold")
    ax_main.set_ylabel("Loss", fontsize=14, fontweight="bold")
    ax_main.set_title(
        "Training Progress: Loss vs Epochs", fontsize=16, fontweight="bold", pad=20
    )
    ax_main.grid(True, alpha=0.3, linestyle="--")
    ax_main.legend(fontsize=12, framealpha=0.9, loc="upper right")

    # Add trend analysis
    if len(train_loss) > 1:
        # Calculate trend
        trend = np.polyfit(epochs, train_loss, 1)
        trend_line = np.poly1d(trend)(epochs)
        ax_main.plot(
            epochs,
            trend_line,
            "--",
            color="gray",
            alpha=0.7,
            linewidth=1.5,
            label="Trend",
        )

    # Metrics panel
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_metrics.axis("off")

    # Calculate important metrics
    final_train_loss = train_loss[-1] if len(train_loss) > 0 else 0
    min_train_loss = np.min(train_loss) if len(train_loss) > 0 else 0
    max_train_loss = np.max(train_loss) if len(train_loss) > 0 else 0
    loss_improvement = (
        ((train_loss[0] - final_train_loss) / train_loss[0] * 100)
        if len(train_loss) > 0 and train_loss[0] > 0
        else 0
    )

    if val_loader is not None:
        final_val_loss = val_loss[-1] if len(val_loss) > 0 else 0
        min_val_loss = np.min(val_loss) if len(val_loss) > 0 else 0
        overfitting_ratio = (
            final_train_loss / final_val_loss if final_val_loss > 0 else 0
        )
    else:
        final_val_loss = min_val_loss = overfitting_ratio = "N/A"

    # Create metrics text
    metrics_text = f"""📊 TRAINING METRICS

    🎯 Final Losses:
    • Training: {final_train_loss:.6f}
    • Validation: {final_val_loss if isinstance(final_val_loss, str) else f'{final_val_loss:.6f}'}

    📈 Loss Statistics:
    • Min Training Loss: {min_train_loss:.6f}
    • Max Training Loss: {max_train_loss:.6f}
    • Min Val Loss: {min_val_loss if isinstance(min_val_loss, str) else f'{min_val_loss:.6f}'}

    📉 Improvement:
    • Loss Reduction: {loss_improvement:.2f}%
    • Overfitting Ratio: {overfitting_ratio if isinstance(overfitting_ratio, str) else f'{overfitting_ratio:.3f}'}

    ⚙️ Training Info:
    • Total Epochs: {len(train_loss)}
    • Learning Rate: {optimizer.param_groups[0]['lr']:.2e}
    • Model: {cfg['model']['model_name'] if 'model_name' in cfg['model'] else 'Neural Inertial Tracking'}"""

    ax_metrics.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax_metrics.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    # MSE subplot
    ax_mse = fig.add_subplot(gs[1, :])
    ax_mse.plot(
        epochs,
        train_mse,
        color="purple",
        linewidth=2,
        label="Training MSE",
        marker="o",
        markersize=3,
    )

    if val_loader is not None:
        ax_mse.plot(
            epochs,
            val_mse,
            color="orange",
            linewidth=2,
            label="Validation MSE",
            marker="s",
            markersize=3,
        )

    if test_loader is not None:
        ax_mse.plot(
            epochs,
            test_mse,
            color="brown",
            linewidth=2,
            label="Test MSE",
            marker="^",
            markersize=3,
        )

    ax_mse.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax_mse.set_ylabel("MSE", fontsize=12, fontweight="bold")
    ax_mse.set_title("Mean Squared Error Progress", fontsize=14, fontweight="bold")
    ax_mse.grid(True, alpha=0.3, linestyle="--")
    ax_mse.legend(fontsize=10, framealpha=0.9)

    # Add final MSE metrics
    final_train_mse = train_mse[-1] if len(train_mse) > 0 else 0
    min_train_mse = np.min(train_mse) if len(train_mse) > 0 else 0
    mse_improvement = (
        ((train_mse[0] - final_train_mse) / train_mse[0] * 100)
        if len(train_mse) > 0 and train_mse[0] > 0
        else 0
    )

    mse_text = f"Final MSE: {final_train_mse:.6f} | Min MSE: {min_train_mse:.6f} | Improvement: {mse_improvement:.2f}%"
    ax_mse.text(
        0.02,
        0.98,
        mse_text,
        transform=ax_mse.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(
        osp.join(out_dir, "_epoch_loss.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Save data files
    np.savetxt(
        osp.join(out_dir, "_epoch_loss.txt"),
        loss_all,
        delimiter=",",
        header=(
            "epoch,train_loss,val_loss,test_loss"
            if val_loader is not None and test_loader is not None
            else "epoch,train_loss"
        ),
    )
    np.savetxt(
        osp.join(out_dir, "_epoch_mse.txt"),
        mse_all,
        delimiter=",",
        header=(
            "epoch,train_mse,val_mse,test_mse"
            if val_loader is not None and test_loader is not None
            else "epoch,train_mse"
        ),
    )

    # Log completion with metrics
    success(f"✅ Training complete! Final training loss: {final_train_loss:.6f}")
    info(
        f"📊 Loss improvement: {loss_improvement:.2f}% | Final MSE: {final_train_mse:.6f}"
    )
    if val_loader is not None and not isinstance(overfitting_ratio, str):
        if overfitting_ratio > 1.1:
            warning(
                f"⚠️ Potential overfitting detected (ratio: {overfitting_ratio:.3f})"
            )
        else:
            info(
                f"✅ Good generalization (overfitting ratio: {overfitting_ratio:.3f})"
            )


def plot_online_adaptation_results(
    cfg: dict,
    optimizer,
    out_dir: str,
    epoch_train_loss,
    epoch_train_mse,
    total_epochs,
):
    """Enhanced plotting for online adaptation results."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Create figure with enhanced styling
    fig = plt.figure(num="Online Adaptation Analysis", dpi=150, figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1])

    # Main loss plot
    ax_main = fig.add_subplot(gs[0, 0])

    # Prepare data
    train_loss = np.array(epoch_train_loss)
    train_mse = np.array(epoch_train_mse)
    epochs = range(1, len(train_loss) + 1)

    # Enhanced plotting
    ax_main.plot(
        epochs,
        train_loss,
        color="#1f77b4",
        linewidth=2.5,
        label="Online Adaptation Loss",
        marker="o",
        markersize=4,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )

    # Enhanced styling for main plot
    ax_main.set_xlabel("Epoch", fontsize=14, fontweight="bold")
    ax_main.set_ylabel("Loss", fontsize=14, fontweight="bold")
    ax_main.set_title(
        "Online Adaptation Progress: Loss vs Epochs",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax_main.grid(True, alpha=0.3, linestyle="--")
    ax_main.legend(fontsize=12, framealpha=0.9, loc="upper right")

    # Add trend analysis
    if len(train_loss) > 1:
        trend = np.polyfit(epochs, train_loss, 1)
        trend_line = np.poly1d(trend)(epochs)
        ax_main.plot(
            epochs,
            trend_line,
            "--",
            color="gray",
            alpha=0.7,
            linewidth=1.5,
            label="Trend",
        )

    # Metrics panel
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_metrics.axis("off")

    # Calculate important metrics
    final_train_loss = train_loss[-1] if len(train_loss) > 0 else 0
    min_train_loss = np.min(train_loss) if len(train_loss) > 0 else 0
    max_train_loss = np.max(train_loss) if len(train_loss) > 0 else 0
    loss_improvement = (
        ((train_loss[0] - final_train_loss) / train_loss[0] * 100)
        if len(train_loss) > 0 and train_loss[0] > 0
        else 0
    )

    # Create metrics text
    metrics_text = f"""📊 ONLINE ADAPTATION METRICS
    🎯 Final Results:
    • Final Loss: {final_train_loss:.6f}
    • Min Loss: {min_train_loss:.6f}
    • Max Loss: {max_train_loss:.6f}

    📈 Performance:
    • Loss Reduction: {loss_improvement:.2f}%
    • Total Epochs: {total_epochs}
    • Learning Rate: {optimizer.param_groups[0]['lr']:.2e}

    ⚙️ Adaptation Info:
    • Method: Online Adaptation
    • Model: {cfg['model']['model_name'] if 'model_name' in cfg['model'] else 'Neural Inertial Tracking'}
    • Convergence: {'✅ Converged' if loss_improvement > 0 else '❌ No Improvement'}"""

    ax_metrics.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax_metrics.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
    )

    # MSE subplot
    ax_mse = fig.add_subplot(gs[1, :])
    ax_mse.plot(
        epochs,
        train_mse,
        color="purple",
        linewidth=2,
        label="Online Adaptation MSE",
        marker="o",
        markersize=3,
    )

    ax_mse.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax_mse.set_ylabel("MSE", fontsize=12, fontweight="bold")
    ax_mse.set_title("Mean Squared Error Progress", fontsize=14, fontweight="bold")
    ax_mse.grid(True, alpha=0.3, linestyle="--")
    ax_mse.legend(fontsize=10, framealpha=0.9)

    # Add final MSE metrics
    final_train_mse = train_mse[-1] if len(train_mse) > 0 else 0
    min_train_mse = np.min(train_mse) if len(train_mse) > 0 else 0
    mse_improvement = (
        ((train_mse[0] - final_train_mse) / train_mse[0] * 100)
        if len(train_mse) > 0 and train_mse[0] > 0
        else 0
    )

    mse_text = f"Final MSE: {final_train_mse:.6f} | Min MSE: {min_train_mse:.6f} | Improvement: {mse_improvement:.2f}%"
    ax_mse.text(
        0.02,
        0.98,
        mse_text,
        transform=ax_mse.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(
        osp.join(out_dir, "_online_adaptation_loss.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Save data files
    np.savetxt(
        osp.join(out_dir, "_online_adaptation_loss.txt"),
        train_loss[:, np.newaxis],
        delimiter=",",
        header="epoch,online_adaptation_loss",
    )
    np.savetxt(
        osp.join(out_dir, "_online_adaptation_mse.txt"),
        train_mse[:, np.newaxis],
        delimiter=",",
        header="epoch,online_adaptation_mse",
    )

    success(
        f"✅ Online adaptation plotting complete! Final loss: {final_train_loss:.6f}"
    )
    info(
        f"📊 Loss improvement: {loss_improvement:.2f}% | Final MSE: {final_train_mse:.6f}"
    )


def plot_offline_finetune_results(
    cfg: dict,
    optimizer,
    out_dir: str,
    epoch_train_loss,
    epoch_val_loss,
    total_epochs,
):
    """Enhanced plotting for offline finetuning results."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Create figure with enhanced styling
    fig = plt.figure(num="Offline Finetuning Analysis", dpi=150, figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1])

    # Main loss plot
    ax_main = fig.add_subplot(gs[0, 0])

    # Prepare data
    train_loss = np.array(epoch_train_loss)
    epochs = range(1, len(train_loss) + 1)

    # Enhanced plotting
    ax_main.plot(
        epochs,
        train_loss,
        color="#1f77b4",
        linewidth=2.5,
        label="Training Loss",
        marker="o",
        markersize=4,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )

    # Plot validation loss if available
    if epoch_val_loss and len(epoch_val_loss) > 0:
        val_loss = np.array(epoch_val_loss)
        ax_main.plot(
            epochs,
            val_loss,
            color="#d62728",
            linewidth=2.5,
            label="Validation Loss",
            marker="s",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

    # Enhanced styling for main plot
    ax_main.set_xlabel("Epoch", fontsize=14, fontweight="bold")
    ax_main.set_ylabel("Loss", fontsize=14, fontweight="bold")
    ax_main.set_title(
        "Offline Finetuning Progress: Loss vs Epochs",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax_main.grid(True, alpha=0.3, linestyle="--")
    ax_main.legend(fontsize=12, framealpha=0.9, loc="upper right")

    # Add trend analysis
    if len(train_loss) > 1:
        trend = np.polyfit(epochs, train_loss, 1)
        trend_line = np.poly1d(trend)(epochs)
        ax_main.plot(
            epochs,
            trend_line,
            "--",
            color="gray",
            alpha=0.7,
            linewidth=1.5,
            label="Trend",
        )

    # Metrics panel
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_metrics.axis("off")

    # Calculate important metrics
    final_train_loss = train_loss[-1] if len(train_loss) > 0 else 0
    min_train_loss = np.min(train_loss) if len(train_loss) > 0 else 0
    loss_improvement = (
        ((train_loss[0] - final_train_loss) / train_loss[0] * 100)
        if len(train_loss) > 0 and train_loss[0] > 0
        else 0
    )

    if epoch_val_loss and len(epoch_val_loss) > 0:
        val_loss = np.array(epoch_val_loss)
        final_val_loss = val_loss[-1] if len(val_loss) > 0 else 0
        overfitting_ratio = (
            final_train_loss / final_val_loss if final_val_loss > 0 else 0
        )
    else:
        final_val_loss = overfitting_ratio = "N/A"

    # Create metrics text
    metrics_text = f"""📊 OFFLINE FINETUNING METRICS

🎯 Final Results:
• Training Loss: {final_train_loss:.6f}
• Validation Loss: {final_val_loss if isinstance(final_val_loss, str) else f'{final_val_loss:.6f}'}
• Min Training Loss: {min_train_loss:.6f}

📈 Performance:
• Loss Reduction: {loss_improvement:.2f}%
• Total Epochs: {total_epochs}
• Learning Rate: {optimizer.param_groups[0]['lr']:.2e}

⚙️ Finetuning Info:
• Method: Offline Finetuning
• Model: {cfg['model']['model_name'] if 'model_name' in cfg['model'] else 'Neural Inertial Tracking'}
• Overfitting Ratio: {overfitting_ratio if isinstance(overfitting_ratio, str) else f'{overfitting_ratio:.3f}'}"""

    ax_metrics.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax_metrics.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8),
    )

    # Learning rate subplot
    ax_lr = fig.add_subplot(gs[1, :])
    lr_values = [optimizer.param_groups[0]["lr"]] * len(
        epochs
    )  # Simplified - could track actual LR history
    ax_lr.plot(
        epochs,
        lr_values,
        color="orange",
        linewidth=2,
        label="Learning Rate",
        marker="o",
        markersize=3,
    )

    ax_lr.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax_lr.set_ylabel("Learning Rate", fontsize=12, fontweight="bold")
    ax_lr.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax_lr.grid(True, alpha=0.3, linestyle="--")
    ax_lr.legend(fontsize=10, framealpha=0.9)
    ax_lr.set_yscale("log")

    # Add final LR metrics
    final_lr = optimizer.param_groups[0]["lr"]
    lr_text = f"Final Learning Rate: {final_lr:.2e}"
    ax_lr.text(
        0.02,
        0.98,
        lr_text,
        transform=ax_lr.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
    )

    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(
        osp.join(out_dir, "_offline_finetune_loss.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Save data files
    np.savetxt(
        osp.join(out_dir, "_offline_finetune_loss.txt"),
        train_loss[:, np.newaxis],
        delimiter=",",
        header="epoch,offline_finetune_loss",
    )

    success(
        f"✅ Offline finetuning plotting complete! Final training loss: {final_train_loss:.6f}"
    )
    info(f"📊 Loss improvement: {loss_improvement:.2f}% | Final LR: {final_lr:.2e}")
    if (
        epoch_val_loss
        and len(epoch_val_loss) > 0
        and not isinstance(overfitting_ratio, str)
    ):
        if overfitting_ratio > 1.1:
            warning(
                f"⚠️ Potential overfitting detected (ratio: {overfitting_ratio:.3f})"
            )
        else:
            info(
                f"✅ Good generalization (overfitting ratio: {overfitting_ratio:.3f})"
            )
