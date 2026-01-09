# Insertion Transformer

A PyTorch implementation of Stern's [Insertion Transformer](https://arxiv.org/abs/1902.03249) for DNA sequence generation.

## Install

```bash
uv sync
```

## Usage

```bash
insertion-transformer --help
```

### Train on DNA/FASTA data with a specific GPU

```bash
insertion-transformer train genome.fna --dna --device cuda:2
```
