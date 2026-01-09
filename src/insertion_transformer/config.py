"""Configuration and hyperparameters for the Insertion Transformer."""

from dataclasses import dataclass


@dataclass
class Config:
    """
    All configurable parameters for the Insertion Transformer.
    The parameters here define the default parameters in the CLI
    """

    # Device
    device: str = "mps"  # "cuda", "mps", or "cpu"

    # Data
    batch_size: int = 32
    block_size: int = 128  # max sequence length for training

    # Model architecture
    n_embd: int = 384  # embedding dimension
    n_heads: int = 6  # number of attention heads
    n_layers: int = 6  # number of transformer blocks
    dropout: float = 0.2  # dropout rate

    # Training
    learning_rate: float = 3e-4
    training_steps: int = 10_000
    eval_iter_period: int = 500
    eval_iters: int = 100  # batches to average for stable loss estimate

    # Generation
    max_gen_len: int = 100
    temperature: float = 0.8

    # Debugging
    debug: bool = False  # Print timing info for each training step


# Special token indices
PAD = 0
BOS = 1  # Beginning of sequence
EOS = 2  # End of sequence


# Default config instance
default_config = Config()
