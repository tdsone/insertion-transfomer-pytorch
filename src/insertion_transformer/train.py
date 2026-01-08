"""Training and generation utilities for the Insertion Transformer."""

import torch
import torch.nn.functional as F

from .config import PAD, EOS, Config, default_config
from .data import DataLoader, InsertionBatch
from .model import InsertionTransformer


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
    data_loader: DataLoader,
    config: Config,
) -> dict[str, float]:
    """Estimate loss over multiple batches for more stable metrics."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        accs = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            batch = data_loader.get_batch(split)
            loss, metrics = compute_loss(model, batch, config.device)
            losses[k] = metrics["loss"]
            accs[k] = metrics["acc"]
        out[f"{split}_loss"] = losses.mean().item()
        out[f"{split}_acc"] = accs.mean().item()
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
    data_loader: DataLoader,
    config: Config = default_config,
) -> InsertionTransformer:
    """Train the insertion transformer."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for step in range(config.training_steps):
        # Periodic evaluation
        if step % config.eval_iter_period == 0 or step == config.training_steps - 1:
            metrics = estimate_loss(model, data_loader, config)
            print(
                f"Step {step:05d} | "
                f"Train loss: {metrics['train_loss']:.4f} | "
                f"Val loss: {metrics['val_loss']:.4f} | "
                f"Train acc: {metrics['train_acc']:.2%} | "
                f"Val acc: {metrics['val_acc']:.2%}"
            )
            # Generate a sample
            sample = generate(
                model,
                max_len=50,
                temperature=config.temperature,
                device=config.device,
            )
            decoded = data_loader.tokenizer.decode(sample)[:60]
            print(f"  Sample: '{decoded}...'")
            print()

        # Training step
        batch = data_loader.get_batch("train")
        loss, _ = compute_loss(model, batch, config.device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training complete!")
    return model


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
