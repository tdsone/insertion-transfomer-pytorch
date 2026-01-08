"""CLI for the Insertion Transformer."""

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="insertion-transformer",
    help="Train and generate with the Insertion Transformer model.",
    add_completion=False,
)


@app.command()
def train(
    data_path: Annotated[
        Path, typer.Argument(help="Path to training data file")
    ] = Path("input.txt"),
    checkpoint: Annotated[
        Optional[Path],
        typer.Option("--checkpoint", "-c", help="Path to save checkpoint"),
    ] = Path("checkpoint.pt"),
    # Device
    device: Annotated[
        str, typer.Option("--device", "-d", help="Device to train on (cuda/mps/cpu)")
    ] = "mps",
    # Data params
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Batch size")
    ] = 64,
    block_size: Annotated[
        int, typer.Option("--block-size", help="Max sequence length")
    ] = 256,
    # Model params
    n_embd: Annotated[int, typer.Option("--n-embd", help="Embedding dimension")] = 384,
    n_heads: Annotated[
        int, typer.Option("--n-heads", help="Number of attention heads")
    ] = 6,
    n_layers: Annotated[
        int, typer.Option("--n-layers", help="Number of transformer layers")
    ] = 6,
    dropout: Annotated[float, typer.Option("--dropout", help="Dropout rate")] = 0.2,
    # Training params
    learning_rate: Annotated[float, typer.Option("--lr", help="Learning rate")] = 3e-4,
    training_steps: Annotated[
        int, typer.Option("--steps", "-s", help="Number of training steps")
    ] = 10_000,
    eval_period: Annotated[
        int, typer.Option("--eval-period", help="Steps between evaluations")
    ] = 500,
    eval_iters: Annotated[
        int, typer.Option("--eval-iters", help="Batches to average for eval")
    ] = 100,
):
    """Train the Insertion Transformer on character-level data."""
    import torch
    from .config import Config
    from .data import DataLoader
    from .model import InsertionTransformer
    from .train import train_model, save_checkpoint

    # Build config
    config = Config(
        device=device,
        batch_size=batch_size,
        block_size=block_size,
        n_embd=n_embd,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        training_steps=training_steps,
        eval_iter_period=eval_period,
        eval_iters=eval_iters,
    )

    # Load data
    typer.echo(f"Loading data from {data_path}...")
    data_loader = DataLoader(data_path, config)
    info = data_loader.info()
    typer.echo(f"  Vocab size: {info['vocab_size']}")
    typer.echo(f"  Train tokens: {info['train_tokens']:,}")
    typer.echo(f"  Val tokens: {info['val_tokens']:,}")

    # Create model
    model = InsertionTransformer(vocab_size=data_loader.vocab_size, config=config)
    model = model.to(config.device)
    typer.echo(f"\nModel: {model.get_num_params() / 1e6:.2f}M parameters")
    typer.echo(f"Device: {config.device}")
    typer.echo()

    # Train
    model = train_model(model, data_loader, config)

    # Save checkpoint
    if checkpoint:
        save_checkpoint(model, str(checkpoint), config)


@app.command()
def generate(
    checkpoint: Annotated[Path, typer.Argument(help="Path to model checkpoint")],
    data_path: Annotated[
        Path, typer.Option("--data", "-d", help="Path to data file (for tokenizer)")
    ] = Path("input.txt"),
    num_samples: Annotated[
        int, typer.Option("--num", "-n", help="Number of samples to generate")
    ] = 5,
    max_len: Annotated[
        int, typer.Option("--max-len", "-l", help="Maximum sequence length")
    ] = 100,
    temperature: Annotated[
        float, typer.Option("--temperature", "-t", help="Sampling temperature")
    ] = 0.8,
    device: Annotated[str, typer.Option("--device", help="Device to run on")] = "mps",
):
    """Generate text using a trained model."""
    import torch
    from .config import Config
    from .data import DataLoader
    from .train import load_checkpoint, generate as gen

    # Load data for tokenizer
    config = Config(device=device)
    data_loader = DataLoader(data_path, config)

    # Load model
    typer.echo(f"Loading model from {checkpoint}...")
    model = load_checkpoint(str(checkpoint), config, device)
    model.eval()

    # Generate samples
    typer.echo(f"\n{'=' * 40}")
    typer.echo("Generated Samples")
    typer.echo(f"{'=' * 40}\n")

    for i in range(num_samples):
        tokens = gen(model, max_len=max_len, temperature=temperature, device=device)
        text = data_loader.tokenizer.decode(tokens)
        typer.echo(f"[Sample {i + 1}]")
        typer.echo(text)
        typer.echo("-" * 40)


@app.command()
def info(
    checkpoint: Annotated[Path, typer.Argument(help="Path to model checkpoint")],
):
    """Show information about a saved checkpoint."""
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)

    typer.echo(f"Checkpoint: {checkpoint}")
    typer.echo(f"Vocab size: {ckpt['vocab_size']}")
    typer.echo("\nModel config:")
    for key, value in ckpt["config"].items():
        typer.echo(f"  {key}: {value}")

    # Count parameters
    total_params = sum(p.numel() for p in ckpt["model_state_dict"].values())
    typer.echo(f"\nTotal parameters: {total_params / 1e6:.2f}M")


if __name__ == "__main__":
    app()
