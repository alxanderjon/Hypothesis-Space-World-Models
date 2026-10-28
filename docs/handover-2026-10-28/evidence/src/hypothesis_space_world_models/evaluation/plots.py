"""Plotting helpers for experiment outputs."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "hypothesis_space_world_models_matplotlib"),
)
import matplotlib.pyplot as plt
import torch


def plot_trajectory_comparison(
    times: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Save a simple target-vs-prediction trajectory plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(times.detach().cpu(), target.detach().cpu(), label="target")
    plt.plot(times.detach().cpu(), prediction.detach().cpu(), "--", label="prediction")
    if title is not None:
        plt.title(title)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_phase_space_comparison(
    target_positions: torch.Tensor,
    target_velocities: torch.Tensor,
    prediction_positions: torch.Tensor,
    prediction_velocities: torch.Tensor,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Save a target-vs-prediction phase-space plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.plot(
        target_positions.detach().cpu(),
        target_velocities.detach().cpu(),
        label="target",
    )
    plt.plot(
        prediction_positions.detach().cpu(),
        prediction_velocities.detach().cpu(),
        "--",
        label="prediction",
    )
    if title is not None:
        plt.title(title)
    plt.xlabel("x(t)")
    plt.ylabel("v(t)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_phase_space_side_by_side(
    comparisons: list[dict[str, torch.Tensor | str]],
    output_path: str | Path,
) -> None:
    """Save side-by-side phase-space comparisons."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(comparisons), figsize=(5 * len(comparisons), 5))
    if len(comparisons) == 1:
        axes = [axes]

    for axis, comparison in zip(axes, comparisons):
        axis.plot(
            comparison["target_positions"].detach().cpu(),
            comparison["target_velocities"].detach().cpu(),
            label="target",
        )
        axis.plot(
            comparison["prediction_positions"].detach().cpu(),
            comparison["prediction_velocities"].detach().cpu(),
            "--",
            label="prediction",
        )
        axis.set_title(str(comparison["title"]))
        axis.set_xlabel("x(t)")
        axis.set_ylabel("v(t)")
        axis.axis("equal")
        axis.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_training_loss(
    losses: list[dict[str, float | int]],
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Save train/test loss curves and their running bests."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [entry["epoch"] for entry in losses]
    train_losses = [entry["train_loss"] for entry in losses]
    test_losses = [entry["test_loss"] for entry in losses]
    best_train_losses = [entry["best_train_loss_so_far"] for entry in losses]
    best_test_losses = [entry["best_test_loss_so_far"] for entry in losses]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_losses, label="train loss", alpha=0.45)
    plt.plot(epochs, test_losses, label="test loss", alpha=0.45)
    plt.plot(epochs, best_train_losses, label="best train loss so far", linewidth=2)
    plt.plot(epochs, best_test_losses, label="best test loss so far", linewidth=2)
    if title is not None:
        plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_omega_history(
    losses: list[dict[str, float | int]],
    true_omega: float,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Save learned global omega over training."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    omega_entries = [entry for entry in losses if "omega_hat" in entry]
    epochs = [entry["epoch"] for entry in omega_entries]
    omega_hats = [entry["omega_hat"] for entry in omega_entries]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, omega_hats, label="learned omega_hat")
    plt.axhline(true_omega, linestyle="--", color="black", label="true omega")
    if title is not None:
        plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("omega")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_omega_gradient_history(
    losses: list[dict[str, float | int]],
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Save the absolute raw-omega gradient over training."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gradient_entries = [
        entry
        for entry in losses
        if entry.get("omega_grad_abs") is not None
    ]
    epochs = [entry["epoch"] for entry in gradient_entries]
    gradients = [entry["omega_grad_abs"] for entry in gradient_entries]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, gradients, label="abs(dL / d raw_omega)")
    if title is not None:
        plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("absolute gradient")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_test_loss_and_omega_history(
    losses: list[dict[str, float | int]],
    true_omega: float,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Save test loss and learned omega on a shared epoch axis."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    omega_entries = [entry for entry in losses if "omega_hat" in entry]
    epochs = [entry["epoch"] for entry in omega_entries]
    test_losses = [entry["test_loss"] for entry in omega_entries]
    omega_hats = [entry["omega_hat"] for entry in omega_entries]

    fig, loss_axis = plt.subplots(figsize=(7, 4.5))
    omega_axis = loss_axis.twinx()
    loss_line = loss_axis.plot(
        epochs,
        test_losses,
        color="tab:blue",
        label="test loss",
    )
    omega_line = omega_axis.plot(
        epochs,
        omega_hats,
        color="tab:orange",
        label="omega_hat",
    )
    true_line = omega_axis.axhline(
        true_omega,
        linestyle="--",
        color="black",
        label="true omega",
    )

    if title is not None:
        plt.title(title)
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("test loss")
    omega_axis.set_ylabel("omega")
    loss_axis.set_yscale("log")
    lines = loss_line + omega_line + [true_line]
    labels = [line.get_label() for line in lines]
    loss_axis.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_coefficient_scatter(
    true_values: torch.Tensor,
    predicted_values: torch.Tensor,
    output_path: str | Path,
    title: str,
    coefficient_name: str,
) -> None:
    """Save predicted-vs-true coefficient scatter with y=x reference."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    true_cpu = true_values.detach().cpu()
    pred_cpu = predicted_values.detach().cpu()
    min_value = min(true_cpu.min().item(), pred_cpu.min().item())
    max_value = max(true_cpu.max().item(), pred_cpu.max().item())

    plt.figure(figsize=(5, 5))
    plt.scatter(true_cpu, pred_cpu, alpha=0.65, s=18)
    plt.plot([min_value, max_value], [min_value, max_value], "--", color="black", label="y = x")
    plt.title(title)
    plt.xlabel(f"true {coefficient_name}")
    plt.ylabel(f"predicted {coefficient_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_coefficient_histograms(
    true_a: torch.Tensor,
    pred_a: torch.Tensor,
    true_b: torch.Tensor,
    pred_b: torch.Tensor,
    output_path: str | Path,
    title: str,
) -> None:
    """Save histograms for true and predicted SHO coefficients."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    entries = [
        (axes[0, 0], true_a, "true a"),
        (axes[0, 1], pred_a, "predicted a"),
        (axes[1, 0], true_b, "true b"),
        (axes[1, 1], pred_b, "predicted b"),
    ]
    for axis, values, label in entries:
        axis.hist(values.detach().cpu(), bins=32, alpha=0.85)
        axis.set_title(label)
        axis.set_xlabel("coefficient value")
        axis.set_ylabel("count")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_coefficient_norms(
    metrics: dict[str, float],
    output_path: str | Path,
    title: str,
) -> None:
    """Save a bar chart of true/predicted coefficient magnitudes."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [
        "pred |a|",
        "true |a|",
        "pred |b|",
        "true |b|",
    ]
    values = [
        metrics["mean_abs_predicted_a_test"],
        metrics["mean_abs_true_a_test"],
        metrics["mean_abs_predicted_b_test"],
        metrics["mean_abs_true_b_test"],
    ]

    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, values, color=["tab:orange", "tab:blue", "tab:orange", "tab:blue"])
    plt.title(title)
    plt.ylabel("mean absolute coefficient")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_linear_matrix_comparison(
    learned_weight: torch.Tensor,
    target_weight: torch.Tensor,
    output_path: str | Path,
    title: str,
) -> None:
    """Visualize learned W, target W, and W - W_true."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    learned = learned_weight.detach().cpu()
    target = target_weight.detach().cpu()
    error = learned - target
    matrices = [
        (learned, "learned W"),
        (target, "true W"),
        (error, "W - W_true"),
    ]
    max_abs = max(matrix.abs().max().item() for matrix, _ in matrices)
    max_abs = max(max_abs, 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for axis, (matrix, label) in zip(axes, matrices):
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
        axis.set_title(label)
        axis.set_xticks([0, 1], labels=["x0", "v0"])
        axis.set_yticks([0, 1], labels=["a", "b"])
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column].item():.3g}",
                    ha="center",
                    va="center",
                    color="black",
                )
    fig.suptitle(title)
    fig.colorbar(image, ax=axes, shrink=0.75)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_phase_space_error(
    times: torch.Tensor,
    position_error: torch.Tensor,
    velocity_error: torch.Tensor,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Save position and velocity error over time."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(times.detach().cpu(), position_error.detach().cpu(), label="x_pred - x_true")
    plt.plot(times.detach().cpu(), velocity_error.detach().cpu(), label="v_pred - v_true")
    if title is not None:
        plt.title(title)
    plt.xlabel("t")
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
