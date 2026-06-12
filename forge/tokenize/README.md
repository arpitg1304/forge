# Forge Tokenize

Action tokenizers turn the canonical `Frame.action` stream into discrete integer
tokens and back (`fit → encode → decode`). VLA / robot-learning models don't
consume continuous action vectors — they predict discrete action tokens, and
every lab reinvents the discretization. Forge ships the proven strategies in-box,
a registry to add your own, and a comparator that benchmarks them on *your*
dataset so you don't have to guess.

## Usage

### CLI

```bash
# List registered strategies
forge tokenize list

# Benchmark every strategy on your dataset (the headline command)
forge tokenize compare ./my_dataset --sample 20 --export report.json
forge tokenize compare pusht                       # resolves to hf://lerobot/pusht

# Fit a tokenizer and save it
forge tokenize fit ./my_dataset --strategy openvla-bins --num-bins 256 --out tok.json

# Write a LeRobot v3 dataset with an action_tokens column
forge tokenize write ./my_dataset ./tokenized --strategy openvla-bins
forge tokenize write ./my_dataset ./tokenized --tokenizer tok.json --keep-actions
```

`compare` reports reconstruction error (MAE / MSE / max-abs), tokens-per-step and
vocabulary utilization per strategy, and marks the lowest-MAE winner.

### Python API

```python
from forge.tokenize import TokenizerRegistry, TokenizerComparator, load_tokenizer

# Fit + encode + decode
tok = TokenizerRegistry.create("openvla-bins", num_bins=256).fit(corpus)  # corpus: (N, D)
tokens = tok.encode(actions)        # (T, D) float -> (T, D) int
recon = tok.decode(tokens)          # (T, D) int  -> (T, D) float

# Benchmark strategies on a dataset
report = TokenizerComparator().compare_dataset("./my_dataset", sample=200)
print(report.best_by("mae"))

# Inference-time detokenization (tokenizer saved alongside the dataset)
tok = load_tokenizer("./tokenized/meta/action_tokenizer.json")
actions = tok.decode(model_output_tokens)
```

## Built-in strategies

All four are per-step (one token per action dim per frame), numpy-only, and ship
in the base install.

| Strategy | Technique | Reference |
|---|---|---|
| `uniform-bins` | per-dim min/max → N uniform bins | RT-1 (Brohan 2022) |
| `openvla-bins` | per-dim 1st–99th percentile, uniform bins in range, clip outliers | OpenVLA (Kim 2024) |
| `quantile-bins` | equal-mass bins (each token roughly equally likely) | — |
| `mu-law` | mu-law companding + uniform quantize (fine resolution near 0) | — |

## Adding a strategy

Register a class with the decorator and it gets the CLI, comparator, and
save/load for free:

```python
from forge.tokenize import TokenizerRegistry
from forge.tokenize.base import BaseTokenizer

@TokenizerRegistry.register("my-strategy")
class MyTokenizer(BaseTokenizer):
    @property
    def vocab_size(self) -> int: ...
    def fit(self, actions): ...        # learn params from (N, D) corpus, return self
    def encode(self, actions): ...     # (T, D) float -> int tokens
    def decode(self, tokens): ...      # inverse
    def get_params(self) -> dict: ...  # JSON-serializable fitted params
    @classmethod
    def from_params(cls, params): ...  # rebuild a fitted tokenizer
```

`granularity` defaults to `"per_step"`. Chunk-granularity tokenizers (e.g. a
DCT+BPE FAST tokenizer) work through the library and `compare` API, but the v1
dataset writer is per-step only and rejects them with a clear error.

## Dataset writer

`forge tokenize write` (and `forge.tokenize.writer.tokenize_and_write`) emits a
LeRobot v3 dataset with an integer `action_tokens` column and saves the fitted
tokenizer to `<output>/meta/action_tokenizer.json`. This is additive — the
LeRobot v3 writer's token column is opt-in via
`LeRobotV3WriterConfig.tokenized_action_feature` and off by default. `--keep-actions`
retains the original float `action` column alongside the tokens.

## Scope (v1)

- Committed: the four analytical strategies, the comparator, the per-step dataset
  writer, and inference-time decode.
- Sequential write only; parallel-worker support is a follow-up.
- FAST (DCT+BPE) and k-means (BeT) strategies are dependency-gated stretch goals
  behind the `tokenize-fast` / `tokenize-learned` extras.
