# Action Tokenizer Featureset for Forge

## Context

VLA/robot-learning models don't consume continuous action vectors directly — they
predict **discrete action tokens**. Every lab reinvents the discretization (RT-1
uniform bins, OpenVLA percentile bins, Pi0-FAST DCT+BPE, BeT k-means), and picking
the right one for a given dataset is trial-and-error. This is exactly the kind of
fragmentation Forge already eliminates for *formats* — so it belongs in Forge as a
first-class featureset.

**Goal:** add a `forge/tokenize/` module that turns the canonical `Frame.action`
stream into discrete tokens and back (`fit → encode → decode`), with:
1. **Proven strategies** shipped in-box (uniform, OpenVLA-percentile, quantile, mu-law — numpy only).
2. **Extensibility** via a `Protocol` + `TokenizerRegistry` (mirrors `FormatRegistry`) so a
   user adds a strategy with one decorator and gets the CLI/compare/serialization for free.
3. **A compare tool** (`forge tokenize compare`) that benchmarks every strategy on *your*
   dataset (reconstruction error, compression, vocab utilization) — the differentiator.
4. **A tokenized-dataset writer** (`forge tokenize write`) that emits a LeRobot v3 dataset
   with an `action_tokens` column for direct model-training ingestion, plus the fitted
   tokenizer saved alongside for **inference-time detokenization**.

The module deliberately copies the layout of `forge/quality/` and `forge/segment/`
(`config.py` / `models.py` / `analyzer.py` / `__init__.py` / `README.md`) and the
strategy-registry pattern of `forge/formats/registry.py`.

---

## Design

### Tokenizer interface (`forge/tokenize/base.py`)

```python
@runtime_checkable
class ActionTokenizer(Protocol):
    name: str
    @property
    def vocab_size(self) -> int: ...        # for sizing model embedding / LLM vocab slice
    @property
    def granularity(self) -> str: ...       # "per_step" | "chunk"
    @property
    def is_fitted(self) -> bool: ...
    def fit(self, actions: NDArray) -> "ActionTokenizer": ...   # corpus (N, D)
    def encode(self, actions: NDArray) -> NDArray: ...          # (T, D) -> per_step (T,D) int / chunk (L,) int
    def decode(self, tokens: NDArray) -> NDArray: ...           # inverse -> (T, D) float
    def get_params(self) -> dict: ...                           # JSON-serializable fitted params
    @classmethod
    def from_params(cls, params: dict) -> "ActionTokenizer": ...
```

`BaseTokenizer` (ABC) implements shared `save(path)` / `load(path)` on top of
`get_params`/`from_params`, embedding the registry `name` so `load` dispatches via the
registry. Small params (bin edges, mins/maxes) serialize to JSON; large arrays
(k-means centroids) to a sidecar `.npz`. This reuses the exact `to_dict`/`from_dict`/
`to_json`/`from_json` style already in `forge/quality/models.py:119-172`.

**The per-step vs. chunk distinction is the one real design decision.** Per-step
tokenizers (binning, mu-law) emit one token per action dim per frame → map cleanly to a
per-frame `action_tokens` column. Chunk tokenizers (FAST) consume a horizon of frames
and emit a variable-length sequence → they are usable through the library/compare API
but the **dataset-writer integration is per-step-only in v1** (chunk-column layout is a
documented follow-up). `granularity` makes this explicit and lets the writer reject
chunk tokenizers with a clear error.

### Registry (`forge/tokenize/registry.py`)

Direct mirror of `FormatRegistry` (`forge/formats/registry.py:38-115`):
`@TokenizerRegistry.register("openvla-bins")`, `get(name)`, `list_strategies()`,
`create(name, config)`. Add-a-strategy-and-you're-done ergonomics.

### Built-in strategies (`forge/tokenize/strategies/`) — committed v1, numpy only

| name | technique | reference |
|---|---|---|
| `uniform-bins` | per-dim min/max → N uniform bins | RT-1 (Brohan 2022) |
| `openvla-bins` | per-dim 1st–99th percentile, uniform bins in range, clip | OpenVLA (Kim 2024) |
| `quantile-bins` | equal-mass bins (each token equally likely) | — |
| `mu-law` | mu-law companding + uniform quantize (fine resolution near 0) | — |

All four share a `_PerStepBinTokenizer` base (`fit` learns per-dim edges; `encode` =
`np.digitize`; `decode` = bin-center lookup). `vocab_size = num_bins`, `granularity =
"per_step"`.

### Optional/stretch strategies (designed + registered, behind extras)

- `fast` (`strategies/fast.py`) — DCT + BPE, Pi0-FAST (Pertsch 2025). `granularity="chunk"`.
  Optional dep guard (`scipy`, `tokenizers`) via the `_ensure_*` /
  `MissingDependencyError` pattern from `forge/segment/analyzer.py:27-40`.
- `kmeans` (`strategies/kmeans.py`) — BeT cluster tokenizer. Optional dep (`scikit-learn`).

These prove the extension path end-to-end. Implemented only after the analytical set +
compare + writer are green; gated so base install is unaffected.

### Comparator (`forge/tokenize/analyzer.py`)

`TokenizerComparator.compare_dataset(path, strategies=None, sample=0) -> ComparisonReport`.
Iterates `reader.read_episodes()` (same path-resolution + iteration as
`forge/quality/analyzer.py:184-262` / `cli.py:_resolve_dataset_path` at
`forge/cli.py:94-163`), stacks per-frame `frame.action` into an `(N, D)` corpus, fits
each strategy, then for a held sample computes:
- reconstruction **MSE / MAE / max-abs / per-dim RMSE** (encode→decode vs. original),
- **tokens-per-step** and (for chunk strategies) compression ratio,
- **vocab utilization** (fraction of codebook actually used).

`ComparisonReport` / `TokenizerStats` dataclasses in `forge/tokenize/models.py` with
`to_dict`/`to_json`/`from_json`.

### Tokenized-dataset writer

Reuses `LeRobotV3Writer` rather than subclassing. Two small, additive pieces:

1. **Writer column hook** (`forge/formats/lerobot_v3/writer.py`): add
   `tokenized_action_feature: str | None = None` to `LeRobotV3WriterConfig`. In the
   row-building loop (`write_episode`, ~writer.py:347-419) — when set and the key is
   present in `frame.extras` — emit it as an integer column and declare it in
   `self._features` (dtype `int64`, shape `[D]`); exclude it from `_STAT_FEATURES` so
   stats aren't computed on token ids. Purely opt-in; default `None` = current behavior.
2. **Write path** (`forge/tokenize/writer.py::tokenize_and_write`): resolve source →
   reader; fit (or load) the tokenizer on the action corpus; stream episodes through a
   thin Episode wrapper that sets `frame.extras["action_tokens"] = tok.encode(frame.action)`
   per frame; call `LeRobotV3Writer.write_episode` / `finalize`; then save the fitted
   tokenizer to `<output>/meta/action_tokenizer.json`. v1 runs **sequentially** (mirrors
   the converter's sequential loop in `forge/convert/converter.py:316-335`); the parallel
   path is a documented follow-up to avoid the worker-serialization complexity. `--keep-actions`
   controls whether the original float `action` column is retained alongside tokens.

Saving the tokenizer into `meta/` gives the **inference detokenize path** for free:
`ActionTokenizer.load("<dataset>/meta/action_tokenizer.json").decode(model_tokens)`.

### CLI (`forge tokenize` sub-app in `forge/cli.py`)

Mirror the `registry_app` wiring (`cli.py:20-24`): `tokenize_app = typer.Typer(...)`,
`app.add_typer(tokenize_app, name="tokenize")`. Commands:

- `forge tokenize list` — registered strategies + vocab/granularity (rich table).
- `forge tokenize compare <dataset> [--strategies a,b] [--sample N] [--export report.json]`
  — the benchmark table (recon error / compression / vocab-util). **The headline command.**
- `forge tokenize fit <dataset> --strategy openvla-bins [--num-bins 256] [--sample N] --out tok.json`
- `forge tokenize write <dataset> <output> --strategy openvla-bins [--tokenizer tok.json] [--keep-actions]`
  — emit a LeRobot v3 dataset with `action_tokens`.

All use the existing `_resolve_dataset_path` helper (`hf://` / local / registry-id),
`rich.progress` (`cli.py:1902-1907`), and `--export`→`report.to_json` conventions.

---

## Files

**New (`forge/tokenize/`):** `__init__.py`, `base.py`, `registry.py`, `config.py`,
`models.py`, `analyzer.py`, `writer.py`, `README.md`, and
`strategies/{__init__.py, bins.py, mulaw.py}` (committed) +
`strategies/{fast.py, kmeans.py}` (optional/stretch).

**Modified:**
- `forge/formats/lerobot_v3/writer.py` — opt-in `tokenized_action_feature` column hook (additive).
- `forge/cli.py` — `tokenize` sub-app + commands.
- `forge/__init__.py` — export `tokenize` symbols (match how `quality`/`segment` are surfaced).
- `pyproject.toml` — optional extras `tokenize-fast` (`scipy`,`tokenizers`) and
  `tokenize-learned` (`scikit-learn`); add to `all`. **No new base deps** (analytical = numpy).
- `README.md` / `ROADMAP.md` — document the featureset.

**Tests:** `tests/test_tokenize.py` — encode/decode roundtrip & determinism; vocab
bounds; reconstruction error ↓ as bins ↑; mu-law finer near zero; `save`/`load` param
fidelity; registry registration; `TokenizerComparator` on a synthetic dataset; and the
writer integration producing a readable LeRobot v3 dataset with an `action_tokens`
column. Reuse the synthetic-Episode/mock-reader pattern from `tests/test_filter.py:18-71`
and `conftest.py` fixtures.

---

## Verification

1. `make test` (or `pytest tests/test_tokenize.py -v`) — all new tests green; existing
   writer tests (`test_lerobot_v3_writer.py`) still pass (column hook is opt-in/default-off).
2. `ruff check forge/tokenize forge/cli.py` and `mypy forge/tokenize` clean.
3. End-to-end on a demo dataset:
   - `forge tokenize list`
   - `forge tokenize compare pusht --sample 20 --export /tmp/tok.json` → sane recon errors,
     `openvla-bins` ≲ `uniform-bins` MAE; vocab-util reported.
   - `forge tokenize write pusht /tmp/pusht_tok --strategy openvla-bins --sample 5` then
     `forge inspect /tmp/pusht_tok` shows the dataset reads back and the `action_tokens`
     feature is present in `meta/info.json`.
   - Roundtrip in Python: `load("/tmp/pusht_tok/meta/action_tokenizer.json").decode(tokens)`
     reconstructs actions within the bin tolerance.

---

## Scope notes / assumptions

- **v1 committed deliverable = analytical strategies + compare + per-step writer + inference
  decode.** FAST and k-means are designed, registered, and dependency-gated, implemented as
  a stretch once the core is green (you left the strategy-tier question open; this keeps base
  install numpy-only and proves the extension pattern without blocking on heavy deps).
- Chunk-granularity tokenizers (FAST) work through the library/compare API but **not** the
  dataset writer in v1 — flagged in the writer with a clear error and noted as follow-up.
- Tokenized-dataset writing is **sequential** in v1; parallel-worker support is a follow-up.
