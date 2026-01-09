"""CLI for the Insertion Transformer."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from .config import default_config

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
    ] = default_config.device,
    # Data format
    dna: Annotated[
        bool,
        typer.Option("--dna", help="Parse as FASTA/DNA file (only train on A,C,G,T)"),
    ] = False,
    # Data params
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Batch size")
    ] = default_config.batch_size,
    block_size: Annotated[
        int, typer.Option("--block-size", help="Max sequence length")
    ] = default_config.block_size,
    # Model params
    n_embd: Annotated[
        int, typer.Option("--n-embd", help="Embedding dimension")
    ] = default_config.n_embd,
    n_heads: Annotated[
        int, typer.Option("--n-heads", help="Number of attention heads")
    ] = default_config.n_heads,
    n_layers: Annotated[
        int, typer.Option("--n-layers", help="Number of transformer layers")
    ] = default_config.n_layers,
    dropout: Annotated[
        float, typer.Option("--dropout", help="Dropout rate")
    ] = default_config.dropout,
    # Training params
    learning_rate: Annotated[
        float, typer.Option("--lr", help="Learning rate")
    ] = default_config.learning_rate,
    training_steps: Annotated[
        int, typer.Option("--steps", "-s", help="Number of training steps")
    ] = default_config.training_steps,
    eval_period: Annotated[
        int, typer.Option("--eval-period", help="Steps between evaluations")
    ] = default_config.eval_iter_period,
    eval_iters: Annotated[
        int, typer.Option("--eval-iters", help="Batches to average for eval")
    ] = default_config.eval_iters,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug mode with timing info")
    ] = default_config.debug,
    plot: Annotated[
        Optional[Path],
        typer.Option(
            "--plot", "-p", help="Save training diagnostics plot to this path"
        ),
    ] = None,
):
    """Train the Insertion Transformer on character-level data."""
    from .config import Config
    from .data import InsertionDataModule, DNADataModule
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
        debug=debug,
    )

    # Load data - use DNA module for FASTA files
    typer.echo(f"Loading data from {data_path}...")
    if dna:
        typer.echo("  Mode: DNA/FASTA (vocabulary: A, C, G, T)")
        data_module = DNADataModule(data_path, config)
    else:
        data_module = InsertionDataModule(data_path, config)
    info = data_module.info()
    typer.echo(f"  Vocab size: {info['vocab_size']}")
    typer.echo(f"  Train tokens: {info['train_tokens']:,}")
    typer.echo(f"  Val tokens: {info['val_tokens']:,}")

    # Create model
    model = InsertionTransformer(vocab_size=data_module.vocab_size, config=config)
    model = model.to(config.device)
    typer.echo(f"\nModel: {model.get_num_params() / 1e6:.2f}M parameters")
    typer.echo(f"Device: {config.device}")
    typer.echo()

    # Train
    model, history = train_model(
        model,
        data_module,
        config,
        plot_save_path=str(plot) if plot else None,
    )

    # Save checkpoint
    if checkpoint:
        save_checkpoint(model, str(checkpoint), config)


@app.command()
def generate(  # TODO review
    checkpoint: Annotated[Path, typer.Argument(help="Path to model checkpoint")],
    data_path: Annotated[
        Path, typer.Option("--data", "-d", help="Path to data file (for tokenizer)")
    ] = Path("input.txt"),
    dna: Annotated[
        bool,
        typer.Option("--dna", help="Use DNA tokenizer (for models trained on FASTA)"),
    ] = False,
    num_samples: Annotated[
        int, typer.Option("--num", "-n", help="Number of samples to generate")
    ] = 5,
    max_len: Annotated[
        int, typer.Option("--max-len", "-l", help="Maximum sequence length")
    ] = default_config.max_gen_len,
    temperature: Annotated[
        float, typer.Option("--temperature", "-t", help="Sampling temperature")
    ] = default_config.temperature,
    device: Annotated[
        str, typer.Option("--device", help="Device to run on")
    ] = default_config.device,
):
    """Generate text using a trained model."""
    import torch
    from .config import Config
    from .data import InsertionDataModule, DNADataModule
    from .tokenizers import DNATokenizer
    from .train import load_checkpoint, generate as gen

    # Load tokenizer
    config = Config(device=device)
    if dna:
        # DNA mode: use fixed DNA tokenizer (no data file needed)
        tokenizer = DNATokenizer()
        typer.echo("Using DNA tokenizer (vocabulary: A, C, G, T)")
    else:
        # Text mode: need data file to build tokenizer
        data_module = InsertionDataModule(data_path, config)
        tokenizer = data_module.tokenizer

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
        text = tokenizer.decode(tokens)
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
