"""Training and generation utilities for the Insertion Transformer."""

import itertools
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .config import EOS, Config, default_config
from .data import InsertionBatch, DataModuleType
from .model import InsertionTransformer


@dataclass
class TrainingHistory:
    """Stores training metrics for every step."""

    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)

    # Periodic eval metrics (step -> value)
    eval_steps: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)

    def record_step(self, step: int, loss: float, grad_norm: float):
        """Record metrics for a training step."""
        self.steps.append(step)
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)

    def record_eval(
        self,
        step: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
    ):
        """Record metrics from periodic evaluation."""
        self.eval_steps.append(step)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)


def plot_training_diagnostics(history: TrainingHistory, save_path: str | None = None):
    """
    Plot training diagnostics: per-step loss and gradient norm.

    Args:
        history: TrainingHistory object with recorded metrics
        save_path: If provided, save plot to this path instead of showing
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Per-step loss
    ax1 = axes[0, 0]
    ax1.plot(history.steps, history.losses, alpha=0.7, linewidth=0.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Per-Step Training Loss")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gradient norm
    ax2 = axes[0, 1]
    ax2.plot(
        history.steps, history.grad_norms, alpha=0.7, linewidth=0.5, color="orange"
    )
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Gradient Norm")
    ax2.set_title("Per-Step Gradient Norm")
    ax2.grid(True, alpha=0.3)
    # Log scale often helps see gradient dynamics
    ax2.set_yscale("log")

    # Plot 3: Smoothed loss (rolling average)
    ax3 = axes[1, 0]
    window = min(50, len(history.losses) // 10 + 1)
    if len(history.losses) > window:
        import numpy as np

        smoothed = np.convolve(history.losses, np.ones(window) / window, mode="valid")
        smooth_steps = history.steps[window - 1 :]
        ax3.plot(
            smooth_steps, smoothed, linewidth=1.5, label=f"Smoothed (window={window})"
        )
        ax3.plot(history.steps, history.losses, alpha=0.2, linewidth=0.5, label="Raw")
        ax3.legend()
    else:
        ax3.plot(history.steps, history.losses, linewidth=1)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Loss")
    ax3.set_title("Smoothed Training Loss")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Eval metrics (train/val loss)
    ax4 = axes[1, 1]
    if history.eval_steps:
        ax4.plot(
            history.eval_steps,
            history.train_losses,
            "o-",
            label="Train Loss",
            markersize=4,
        )
        ax4.plot(
            history.eval_steps, history.val_losses, "o-", label="Val Loss", markersize=4
        )
        ax4.legend()
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Loss")
    ax4.set_title("Eval Losses (Averaged)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training diagnostics to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def compute_loss(
    model: InsertionTransformer,
    batch: InsertionBatch,
    device: str,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the insertion transformer loss.

    Key insight: We give credit for ANY valid insertion, not just the sampled one.
    This is done via logsumexp over all valid (position, token) pairs.

    Loss = -log P(any valid action)
         = -logsumexp_{(p,t) in valid} log P(insert t at p)

    For EOS steps (when hypo == ref), the valid action is finishing.
    """
    hypo = batch.hypo.to(device)
    hypo_len = batch.hypo_len.to(device)
    valid_mask = batch.valid_mask.to(device)  # [B, T+1, vocab_size]

    B, T = hypo.shape

    # Forward pass
    out = model(hypo, hypo_len)
    insert_logp = out["insert_logp"]  # [B, T+1, vocab_size]

    # Ensure valid_mask matches insert_logp shape
    if valid_mask.shape[1] < insert_logp.shape[1]:
        # Pad valid_mask
        pad_size = insert_logp.shape[1] - valid_mask.shape[1]
        valid_mask = F.pad(valid_mask, (0, 0, 0, pad_size), value=False)
    elif valid_mask.shape[1] > insert_logp.shape[1]:
        # Truncate valid_mask
        valid_mask = valid_mask[:, : insert_logp.shape[1], :]

    # Mask out invalid insertions
    masked_logp = insert_logp.masked_fill(~valid_mask, float("-inf"))

    # logsumexp over all valid (position, token) pairs
    # Flatten positions and tokens, then logsumexp
    logp_any_valid = torch.logsumexp(masked_logp.view(B, -1), dim=-1)  # [B]

    # Loss is negative log probability
    loss = -logp_any_valid.mean()

    # Compute metrics
    with torch.no_grad():
        # Accuracy: is the argmax a valid action?
        pred_flat = insert_logp.view(B, -1).argmax(dim=-1)  # [B]
        valid_flat = valid_mask.view(B, -1)  # [B, T*vocab]
        acc = valid_flat.gather(1, pred_flat.unsqueeze(1)).float().mean()

        # Perplexity-like metric
        ppl = torch.exp(loss)

    metrics = {
        "loss": loss.item(),
        "acc": acc.item(),
        "ppl": ppl.item(),
    }

    return loss, metrics


@torch.no_grad()
def estimate_loss(
    model: InsertionTransformer,
    data_module: DataModuleType,
    config: Config,
) -> dict[str, float]:
    """Estimate loss over multiple batches for more stable metrics."""
    out = {}
    model.eval()

    loaders = {"train": data_module.train_loader, "val": data_module.val_loader}

    for split, loader in loaders.items():
        losses = torch.zeros(config.eval_iters)
        accs = torch.zeros(config.eval_iters)
        num_batches = 0
        for k, batch in enumerate(loader):
            if k >= config.eval_iters:
                break
            loss, metrics = compute_loss(model, batch, config.device)
            losses[k] = metrics["loss"]
            accs[k] = metrics["acc"]
            num_batches = k + 1
        out[f"{split}_loss"] = (
            losses[:num_batches].mean().item() if num_batches > 0 else 0.0
        )
        out[f"{split}_acc"] = (
            accs[:num_batches].mean().item() if num_batches > 0 else 0.0
        )

    model.train()
    return out


@torch.no_grad()
def generate(
    model: InsertionTransformer,
    max_len: int = 100,
    temperature: float = 1.0,
    device: str = "cpu",
) -> list[int]:
    """
    Generate a sequence using the insertion transformer.

    Starts from empty sequence and iteratively inserts tokens
    until EOS is predicted or max_len is reached.
    """
    model.eval()

    # Start with empty sequence
    hypo = []

    for _ in range(max_len):
        if len(hypo) == 0:
            # Empty sequence: create dummy tensor
            hypo_tensor = torch.zeros(1, 1, dtype=torch.long, device=device)
            hypo_len = torch.tensor([0], device=device)
        else:
            hypo_tensor = torch.tensor([hypo], dtype=torch.long, device=device)
            hypo_len = torch.tensor([len(hypo)], device=device)

        # Forward pass
        out = model(hypo_tensor, hypo_len)
        insert_logp = out["insert_logp"][0]  # [T+1, vocab_size]

        # Apply temperature
        if temperature != 1.0:
            insert_logp = insert_logp / temperature

        # Sample from the distribution
        # Flatten, sample, then unflatten
        num_positions = len(hypo) + 1
        logp_flat = insert_logp[:num_positions].reshape(
            -1
        )  # [num_positions * vocab_size]
        probs = F.softmax(logp_flat, dim=0)
        idx = torch.multinomial(probs, 1).item()

        # Decode position and token
        pos = int(idx // model.vocab_size)
        tok = int(idx % model.vocab_size)

        # Check for EOS
        if tok == EOS:
            break

        # Insert token
        hypo.insert(pos, tok)

    model.train()
    return hypo


def train_model(
    model: InsertionTransformer,
    data_module: DataModuleType,
    config: Config = default_config,
    plot_diagnostics: bool = False,
    plot_save_path: str | None = None,
) -> tuple[InsertionTransformer, TrainingHistory]:
    """
    Train the insertion transformer.

    Args:
        model: The model to train
        data_module: Data module with train/val loaders
        config: Training configuration
        plot_diagnostics: If True, plot training diagnostics after training
        plot_save_path: If provided, save plot to this path

    Returns:
        Tuple of (trained model, training history)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    debug = config.debug
    history = TrainingHistory()

    if debug:
        print("[DEBUG] Debug mode enabled - timing each step")
        print()

    # Create infinite iterator over training batches (cycles through epochs)
    train_iter = iter(itertools.cycle(data_module.train_loader))

    for step in range(config.training_steps):
        step_start = time.perf_counter()

        # Periodic evaluation
        if step % config.eval_iter_period == 0 or step == config.training_steps - 1:
            eval_start = time.perf_counter()
            metrics = estimate_loss(model, data_module, config)
            eval_time = (time.perf_counter() - eval_start) * 1000

            # Record eval metrics
            history.record_eval(
                step=step,
                train_loss=metrics["train_loss"],
                val_loss=metrics["val_loss"],
                train_acc=metrics["train_acc"],
                val_acc=metrics["val_acc"],
            )

            if debug:
                print(f"[DEBUG] Step {step:05d} | estimate_loss: {eval_time:.1f}ms")

            print(
                f"Step {step:05d} | "
                f"Train loss: {metrics['train_loss']:.4f} | "
                f"Val loss: {metrics['val_loss']:.4f} | "
                f"Train acc: {metrics['train_acc']:.2%} | "
                f"Val acc: {metrics['val_acc']:.2%}"
            )

            # Generate a sample
            gen_start = time.perf_counter()
            sample = generate(
                model,
                max_len=50,
                temperature=config.temperature,
                device=config.device,
            )
            gen_time = (time.perf_counter() - gen_start) * 1000

            if debug:
                print(f"[DEBUG] Step {step:05d} | generate: {gen_time:.1f}ms")

            decoded = data_module.tokenizer.decode(sample)[:60]
            print(f"  Sample: '{decoded}'...")
            print()

        # Training step - iterate over the dataloader
        batch_start = time.perf_counter()
        batch = next(train_iter)
        batch_time = (time.perf_counter() - batch_start) * 1000

        forward_start = time.perf_counter()
        loss, _ = compute_loss(model, batch, config.device)
        forward_time = (time.perf_counter() - forward_start) * 1000

        backward_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        backward_time = (time.perf_counter() - backward_start) * 1000

        # Compute gradient norm (before optimizer step)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float("inf")
        ).item()

        # Record per-step metrics
        history.record_step(step=step, loss=loss.item(), grad_norm=grad_norm)

        optim_start = time.perf_counter()
        optimizer.step()
        optim_time = (time.perf_counter() - optim_start) * 1000

        if debug:
            total_time = (time.perf_counter() - step_start) * 1000
            print(
                f"[DEBUG] Step {step:05d} | "
                f"batch: {batch_time:.1f}ms | "
                f"forward: {forward_time:.1f}ms | "
                f"backward: {backward_time:.1f}ms | "
                f"optim: {optim_time:.1f}ms | "
                f"total: {total_time:.1f}ms | "
                f"grad_norm: {grad_norm:.2f}"
            )

    print("Training complete!")

    # Plot diagnostics if requested
    if plot_diagnostics or plot_save_path:
        plot_training_diagnostics(history, save_path=plot_save_path)

    return model, history


def save_checkpoint(model: InsertionTransformer, path: str, config: Config):
    """Save model checkpoint."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": model.vocab_size,
            "config": {
                "n_embd": config.n_embd,
                "n_heads": config.n_heads,
                "n_layers": config.n_layers,
                "dropout": config.dropout,
                "block_size": config.block_size,
            },
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str, config: Config, device: str) -> InsertionTransformer:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    # Update config from checkpoint
    for key, value in checkpoint["config"].items():
        if hasattr(config, key):
            setattr(config, key, value)

    model = InsertionTransformer(
        vocab_size=checkpoint["vocab_size"],
        config=config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Loaded checkpoint from {path}")
    return model
