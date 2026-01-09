"""Data loading and batch generation for the Insertion Transformer."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from .config import PAD, BOS, EOS, Config, default_config
from .tokenizers import CharTokenizer, DNATokenizer, parse_fasta


# ============ INSERTION ORACLE ============


def get_optimal_inserts(cand: list[int], ref: list[int]) -> list[set[int]]:
    """
    For a candidate sequence that is a subsequence of reference,
    compute valid insertions at each position.

    Returns: list of sets, one per position in [0, len(cand)+1)
             Each set contains tokens that can be inserted at that position
             while keeping cand as a subsequence of ref.

    Example:
        ref  = [A, B, C, D]
        cand = [B, D]

        Position 0 (before B): can insert A (to get [A,B,D] which is subseq of ref)
        Position 1 (between B and D): can insert C
        Position 2 (after D): nothing (already at end)

        Returns: [{A}, {C}, {}]
    """
    # Find where each cand token appears in ref (leftmost match)
    # starts[i] = position in ref after matching cand[:i]
    starts = [0]
    ref_iter = iter(enumerate(ref))
    for cand_item in cand:
        for ref_pos, ref_item in ref_iter:
            if ref_item == cand_item:
                starts.append(ref_pos + 1)
                break
        else:
            raise ValueError("cand must be a subsequence of ref")

    # Find rightmost matches going backwards
    # ends[i] = position in ref before matching cand[i:]
    ends = [len(ref)]
    reverse_ref_iter = iter(reversed(list(enumerate(ref))))
    for cand_item in reversed(cand):
        for ref_pos, ref_item in reverse_ref_iter:
            if ref_item == cand_item:
                ends.append(ref_pos)
                break
        else:
            raise ValueError("cand must be a subsequence of ref")
    ends = ends[::-1]

    # Valid inserts at position i are tokens in ref[starts[i]:ends[i]]
    inserts = []
    for i, j in zip(starts, ends):
        inserts.append(set(ref[i:j]))
    return inserts


# ============ TRAJECTORY GENERATION ============


@dataclass
class InsertionSample:
    """A single training sample for the insertion transformer."""

    hypo: list[int]  # Current partial sequence (input)
    ref_inserts: list[set[int]]  # Valid insertions at each position (for loss)
    chosen_pos: int  # Position where we inserted (target)
    chosen_token: int  # Token we inserted (target)


def generate_trajectory(ref: list[int], mode: str = "random") -> list[InsertionSample]:
    """
    Generate a full trajectory from empty sequence to reference.

    Args:
        ref: Target sequence (list of token indices)
        mode: "random" for random order, "l2r" for left-to-right

    Returns:
        List of InsertionSample, one per insertion step
    """
    samples = []
    hypo = []

    while True:
        inserts = get_optimal_inserts(hypo, ref)

        # Flatten to list of (position, token) pairs
        flat_inserts = [
            (pos, tok) for pos, tokens in enumerate(inserts) for tok in tokens
        ]

        if not flat_inserts:
            # Trajectory complete - hypo == ref
            break

        # Choose next insertion
        if mode == "random":
            chosen_pos, chosen_tok = random.choice(flat_inserts)
        elif mode == "l2r":
            # Left-to-right: always insert at position len(hypo), token ref[len(hypo)]
            chosen_pos, chosen_tok = len(hypo), ref[len(hypo)]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        samples.append(
            InsertionSample(
                hypo=list(hypo),
                ref_inserts=inserts,
                chosen_pos=chosen_pos,
                chosen_token=chosen_tok,
            )
        )

        # Apply the insertion
        hypo.insert(chosen_pos, chosen_tok)

    return samples


def generate_trajectory_with_eos(
    ref: list[int], mode: str = "random"
) -> list[InsertionSample]:
    """Generate trajectory including the final EOS step."""
    samples = generate_trajectory(ref, mode=mode)

    # Add final step where hypo == ref and we should predict EOS
    if ref:
        eos_inserts = [{EOS} for _ in range(len(ref) + 1)]
        chosen_pos = random.randint(0, len(ref))
        samples.append(
            InsertionSample(
                hypo=list(ref),
                ref_inserts=eos_inserts,
                chosen_pos=chosen_pos,
                chosen_token=EOS,
            )
        )

    return samples


# ============ BATCH GENERATION ============


@dataclass
class InsertionBatch:
    """A batch of training samples for the insertion transformer."""

    hypo: torch.Tensor  # [B, max_hypo_len] - padded hypotheses
    hypo_len: torch.Tensor  # [B] - actual lengths
    target_pos: torch.Tensor  # [B] - position to insert
    target_token: torch.Tensor  # [B] - token to insert
    valid_mask: torch.Tensor  # [B, max_hypo_len+1, vocab_size] - 1 where valid


def collate_samples(samples: list[InsertionSample], vocab_size: int) -> InsertionBatch:
    """Convert list of samples into a padded batch."""
    B = len(samples)
    max_len = max(len(s.hypo) for s in samples) if samples else 0

    # Pad hypotheses
    hypo = torch.full((B, max_len), PAD, dtype=torch.long)
    hypo_len = torch.zeros(B, dtype=torch.long)
    target_pos = torch.zeros(B, dtype=torch.long)
    target_token = torch.zeros(B, dtype=torch.long)

    # Valid mask: [B, max_len+1, vocab_size]
    # +1 because we can insert at positions 0 to len(hypo) inclusive
    valid_mask = torch.zeros((B, max_len + 1, vocab_size), dtype=torch.bool)

    for i, s in enumerate(samples):
        L = len(s.hypo)
        if L > 0:
            hypo[i, :L] = torch.tensor(s.hypo)
        hypo_len[i] = L
        target_pos[i] = s.chosen_pos
        target_token[i] = s.chosen_token

        # Fill valid mask
        for pos, valid_tokens in enumerate(s.ref_inserts):
            for tok in valid_tokens:
                valid_mask[i, pos, tok] = True

    return InsertionBatch(
        hypo=hypo,
        hypo_len=hypo_len,
        target_pos=target_pos,
        target_token=target_token,
        valid_mask=valid_mask,
    )


# ============ DATA LOADER ============


class InsertionDataset(Dataset):
    """
    Map-style Dataset that generates insertion samples on-the-fly.

    Each index corresponds to a starting position in the data.
    Trajectories are generated randomly, so each epoch sees different samples.
    """

    def __init__(
        self,
        data: torch.Tensor,
        vocab_size: int,
        block_size: int,
        include_eos: bool = True,
    ):
        self.data = data
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.include_eos = include_eos

    def __len__(self) -> int:
        """Number of possible starting positions."""
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> InsertionSample:
        """Get a random insertion sample from a trajectory starting at idx."""
        ref = self.data[idx : idx + self.block_size].tolist()

        # Generate trajectory and pick one random sample
        if self.include_eos:
            trajectory = generate_trajectory_with_eos(ref, mode="random")
        else:
            trajectory = generate_trajectory(ref, mode="random")

        if trajectory:
            return random.choice(trajectory)

        raise Exception("Could not create a trajectory!")


def _collate_fn(samples: list[InsertionSample], vocab_size: int) -> InsertionBatch:
    """Custom collate function that wraps collate_samples."""
    return collate_samples(samples, vocab_size)


class InsertionDataModule:
    """
    Data module that creates PyTorch DataLoaders for training.

    Exposes standard PyTorch DataLoaders via `train_loader` and `val_loader`.

    Key features:
    - num_workers: Parallel worker processes prepare batches in background
    - prefetch_factor: Workers prefetch batches ahead of training
    - pin_memory: Faster CPU->GPU memory transfer
    - persistent_workers: Avoids worker respawn overhead between epochs
    """

    def __init__(
        self,
        data_path: str | Path,
        config: Config = default_config,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        self.config = config

        # Load and tokenize text
        with open(data_path, "r") as f:
            text = f.read()

        self.tokenizer = CharTokenizer(text)
        self.vocab_size = self.tokenizer.vocab_size

        # Split data
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        train_size = int(len(data) * 0.9)
        self.train_data = data[:train_size]
        self.val_data = data[train_size:]

        # Create datasets
        self.train_dataset = InsertionDataset(
            self.train_data,
            self.vocab_size,
            config.block_size,
            include_eos=True,
        )
        self.val_dataset = InsertionDataset(
            self.val_data,
            self.vocab_size,
            config.block_size,
            include_eos=True,
        )

        # Create collate function with vocab_size bound
        def collate(samples):
            return _collate_fn(samples, self.vocab_size)

        # DataLoader settings for efficiency
        loader_kwargs = {
            "batch_size": config.batch_size,
            "collate_fn": collate,
            "shuffle": True,  # Shuffle each epoch
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
            "pin_memory": True,  # Faster CPU->GPU transfer
            "persistent_workers": num_workers > 0,  # Keep workers alive
            "drop_last": True,  # Avoid partial batches
        }

        self.train_loader = TorchDataLoader(self.train_dataset, **loader_kwargs)
        self.val_loader = TorchDataLoader(
            self.val_dataset, **{**loader_kwargs, "shuffle": False}
        )

    def info(self) -> dict:
        """Return info about the dataset."""
        return {
            "vocab_size": self.vocab_size,
            "train_tokens": len(self.train_data),
            "val_tokens": len(self.val_data),
        }


class DNADataModule:
    """
    Data module for training on DNA/FASTA files.

    Parses FASTA format, extracts only valid DNA nucleotides (A, C, G, T),
    and uses a fixed 7-token vocabulary (PAD, BOS, EOS + 4 nucleotides).
    """

    def __init__(
        self,
        data_path: str | Path,
        config: Config = default_config,
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        self.config = config

        # Parse FASTA file to get DNA sequence
        print(f"Parsing FASTA file: {data_path}")
        dna_sequence = parse_fasta(data_path)
        print(f"Extracted {len(dna_sequence):,} nucleotides")

        # Use DNA-specific tokenizer (fixed vocabulary)
        self.tokenizer = DNATokenizer()
        self.vocab_size = self.tokenizer.vocab_size

        # Tokenize the DNA sequence
        data = torch.tensor(self.tokenizer.encode(dna_sequence), dtype=torch.long)

        # Split data (90% train, 10% val)
        train_size = int(len(data) * 0.9)
        self.train_data = data[:train_size]
        self.val_data = data[train_size:]

        # Create datasets
        self.train_dataset = InsertionDataset(
            self.train_data,
            self.vocab_size,
            config.block_size,
            include_eos=True,
        )
        self.val_dataset = InsertionDataset(
            self.val_data,
            self.vocab_size,
            config.block_size,
            include_eos=True,
        )

        # Create collate function with vocab_size bound
        def collate(samples):
            return _collate_fn(samples, self.vocab_size)

        # DataLoader settings for efficiency
        loader_kwargs = {
            "batch_size": config.batch_size,
            "collate_fn": collate,
            "shuffle": True,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
            "drop_last": True,
        }

        self.train_loader = TorchDataLoader(self.train_dataset, **loader_kwargs)
        self.val_loader = TorchDataLoader(
            self.val_dataset, **{**loader_kwargs, "shuffle": False}
        )

    def info(self) -> dict:
        """Return info about the dataset."""
        return {
            "vocab_size": self.vocab_size,
            "train_tokens": len(self.train_data),
            "val_tokens": len(self.val_data),
        }


# Type alias for any data module
DataModuleType = Union[InsertionDataModule, DNADataModule]

# Backwards compatibility alias
DataLoader = InsertionDataModule
