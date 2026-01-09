"""Tokenizers for the Insertion Transformer."""

from __future__ import annotations

from pathlib import Path

from .config import PAD, BOS, EOS


class CharTokenizer:
    """Character-level tokenizer with special tokens."""

    def __init__(self, text: str):
        """Build vocabulary from text."""
        chars = sorted(set(text))
        self.vocab_size = len(chars) + 3  # +3 for PAD, BOS, EOS

        # Character mappings (offset by 3 for special tokens)
        self.char_to_idx = {ch: i + 3 for i, ch in enumerate(chars)}
        self.idx_to_char = {i + 3: ch for i, ch in enumerate(chars)}
        self.idx_to_char[PAD] = "<PAD>"
        self.idx_to_char[BOS] = "<BOS>"
        self.idx_to_char[EOS] = "<EOS>"

    def encode(self, s: str) -> list[int]:
        """Encode string to list of token indices."""
        return [self.char_to_idx[c] for c in s]

    def decode(self, tokens: list[int]) -> str:
        """Decode token indices to string, filtering special tokens."""
        return "".join(self.idx_to_char[t] for t in tokens if t >= 3)


class DNATokenizer:
    """
    DNA-specific tokenizer with fixed vocabulary.

    Vocabulary:
        0: PAD
        1: BOS
        2: EOS
        3: A
        4: C
        5: G
        6: T
    """

    # Fixed DNA vocabulary
    NUCLEOTIDES = "ACGT"

    def __init__(self):
        """Initialize DNA tokenizer with fixed vocabulary."""
        self.vocab_size = 7  # PAD, BOS, EOS + 4 nucleotides

        # Character mappings (offset by 3 for special tokens)
        self.char_to_idx = {ch: i + 3 for i, ch in enumerate(self.NUCLEOTIDES)}
        self.idx_to_char = {i + 3: ch for i, ch in enumerate(self.NUCLEOTIDES)}
        self.idx_to_char[PAD] = "<PAD>"
        self.idx_to_char[BOS] = "<BOS>"
        self.idx_to_char[EOS] = "<EOS>"

        # Set of valid nucleotides for filtering
        self._valid_chars = set(self.NUCLEOTIDES)

    def encode(self, s: str) -> list[int]:
        """Encode DNA string to list of token indices."""
        return [self.char_to_idx[c] for c in s if c in self._valid_chars]

    def decode(self, tokens: list[int]) -> str:
        """Decode token indices to DNA string, filtering special tokens."""
        return "".join(self.idx_to_char[t] for t in tokens if t >= 3)


def parse_fasta(file_path: str | Path) -> str:
    """
    Parse a FASTA file and extract DNA sequences.

    Skips header lines (starting with '>') and extracts only valid
    DNA nucleotides (A, C, G, T), converting to uppercase.

    Args:
        file_path: Path to FASTA file

    Returns:
        Concatenated DNA sequence string (uppercase, only ACGT)

    Raises:
        ValueError: If a non-header line contains invalid characters
    """

    valid_nucleotides = set("ACGTacgt")
    sequences = []

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Skip header lines
            if line.startswith(">"):
                continue
            # Check for invalid characters
            invalid_chars = set(line) - valid_nucleotides
            if invalid_chars:
                raise ValueError(
                    f"Invalid character(s) {invalid_chars} found at line {line_num}: "
                    f"'{line[:50]}{'...' if len(line) > 50 else ''}'"
                )
            # Convert to uppercase
            sequences.append(line.upper())

    return "".join(sequences)
