import torch
import torch.nn as nn

# ===== SETTINGS =====
DEVICE = "mps"
BLOCK_SIZE = 32
BATCH_SIZE = 10
TRAINING_STEPS = 5_000
LEARNING_RATE = 3e-3
EVAL_ITER_PERIOD = 500  # Eval every 500 steps
EVAL_ITERS = 200
# ====================

"""
Definitions
x: source_canvas <-- this is the start sequence (can be empty) that we insert into
y: target_canvas <-- this is the final sequence that we want to get to

y_hat_t: hypothesis canvas as time t

C: content vocabulary (== token vocab)

L: the set of all possible insert locations
-> we just have slots between each pair of tokens and on the left of the first and on the right of the last token

Let c ∈ C a particular choice of token to fill in and l in L be a position to insert

p(c, l | x, y_hat_t) = InsertionTransformer(x, y_hat_t) <-- p is the joint distribution of c and l

Here, we focus on the factorized version p(c, l) = p(c | l) * p(l)

Permitted operations at time t are inserting into slot from l = 0 to l = |y_hat_t|. This could something like this: 

target_canvas: T h e _ b r o w n _ f o x
y_hat_t: [l=0] T [l=1] o [l=2] x [l=3] (="Tox")

For the factorization, they say:

### p(c | l) = softmax(h_l @ W) ###

where h_l is the l-th row of H and H is the matrix 
of slot representations with H.shape = (T + 1, h) where h is the size of the 
hidden state (= embedding dimension?) and T is the length of the current partial 
hypothesis (i.e. len(y_hat_t)).
-> we somehow have to produce H, pluck out h_l and multiply it with W which is the
"standard softmax projection matrix from the transformer" (?) which gives us the logits
for the vocab and thus after softmaxing the probabilities.

### p(l) = softmax(Hq) ###
q is a learnable query vector with q.shape = h and H the slot representation matrix
-> Hq gives the logits over all positions and softmax the probs

Attention: 
- we don't have an attention mask

Training data: 
- For normal transformer the labels are just the shifted inputs to predict the next token

For the IT, we would have as a training example the current hypothesis canvas tokens 
and then the next canvas with one insertion as output?

Here we focus on the balanced binary tree strategy for decoding.

Current assumption: 
- we have normal multihead attention
- we have normal Attention blocks except for the absence of the mask

"""


class InsertionTransformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, source_canvas, y_hat_t):
        pass

    def generate(self, x):
        pass


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    idx = torch.randint(
        len(data) - BLOCK_SIZE, (BATCH_SIZE,)
    )  # we need bs indices each between 0 and len(DATA) - block
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in idx])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in idx])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    model.train()
    return out


def train(model: nn.Module):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for i in range(TRAINING_STEPS):
        if i % EVAL_ITER_PERIOD == 0 or i == TRAINING_STEPS - 1:
            losses = estimate_loss()
            print(
                f"At step {i:05} - Training loss {losses['train']:.4f} | Validation loss: {losses['val']:.4f}"
            )

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


from tokenizer import decode, LEN_VOCAB, DATA

train_data_size = int(len(DATA) * 0.9)
train_data = DATA[:train_data_size]
val_data = DATA[train_data_size:]

model = InsertionTransformer()
model = model.to(DEVICE)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

print("=== TRAINING ===")
train(model=model)
print("Done training!")

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
