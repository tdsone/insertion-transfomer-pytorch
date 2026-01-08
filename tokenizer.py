import torch

with open("input.txt", "r") as f:
    input = f.read()

chars = sorted(list(set(input)))
LEN_VOCAB = len(chars)

ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for c, i in ctoi.items()}


def encode(text: str) -> list[int]:
    return [ctoi[c] for c in text]


def decode(tokens: list[int]) -> str:
    return "".join([itoc[i] for i in tokens])


DATA = torch.tensor(encode(input), dtype=torch.long)
