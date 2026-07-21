# Registry Candidates — backlog

Datasets researched but **not yet added** to `datasets.json`. These are the
"Tier 3 / niche-but-notable" set plus anything deferred. Promote one by filling
in the template in [CONTRIBUTING.md](CONTRIBUTING.md), running
`forge registry validate --probe`, and moving it out of this file.

Last researched: 2026-07-21. Metadata below is approximate and should be
re-verified at promotion time (licenses and repo ids drift).

## Tier 3 — niche but notable

### DexMimicGen
- **What:** Auto-generated **bimanual dexterous-hand** demos — ~21k trajectories
  from ~60 human source demos across 9 tasks (sim + real). Fills the
  dexterous-hand gap the registry currently has zero of.
- **Paper:** https://arxiv.org/abs/2410.24185 · project: https://dexmimicgen.github.io/
- **License:** check repo (NVIDIA-affiliated; likely NVIDIA Source Code License for code)
- **Format:** hdf5 (robomimic-style) · **Embodiment:** bimanual dexterous humanoid
- **Source:** see project page / GitHub
- **Tags:** manipulation, bi_manual, simulation, contact_rich

### EgoDex (Apple)
- **What:** Very large **egocentric human** manipulation dataset with paired hand
  poses; human-video collection paradigm for pretraining.
- **Paper:** search "EgoDex Apple egocentric manipulation 2025"
- **License:** verify (Apple research license — likely non-commercial/gated → tag
  `registration_required` and/or `non_commercial`)
- **Format:** other (video + hand pose) · **Embodiment:** human hands
- **Tags:** manipulation, real_world (human)

### FMB — Functional Manipulation Benchmark
- **What:** Real-world **contact-rich assembly** on a Franka; peg-in-hole style
  functional manipulation with a standard benchmark protocol.
- **Paper:** https://functional-manipulation-benchmark.github.io/
- **License:** verify · **Format:** rlds/other · **Embodiment:** franka
- **Tags:** manipulation, real_world, contact_rich

### BEHAVIOR-1K / OmniGibson
- **What:** Huge sim task diversity incl. **mobile manipulation** and household
  activities; OmniGibson is the simulator, BEHAVIOR-1K the task/dataset suite.
- **Paper:** https://behavior.stanford.edu/
- **License:** MIT (verify) · **Format:** other · **Embodiment:** mobile manipulators, humanoid
- **Tags:** manipulation, mobile_manipulation, simulation, multi_task, large_scale

## Deferred / worth a look

- **Mobile ALOHA (full dataset)** — bimanual **mobile** manipulation; registry
  already has `aloha_mobile_cabinet`, but the full release is broader.
  Project: https://mobile-aloha.github.io/
- **RoboCasa365** — 2026 successor to RoboCasa: 365 tasks, 2,500 scenes,
  ~600h human + ~1,600h synthetic. Promote once distribution stabilizes.
- **ARIO** — unified cross-embodiment standard/dataset; overlaps OXE in spirit.
- **Galaxea Open-World / other 2025-2026 humanoid releases** — re-check scale &
  licensing before adding.

## Notes on categories still thin after this batch

- **Dexterous multi-fingered hands:** only ManiSkill (partial) + DexMimicGen
  (candidate). Worth a dedicated dexterous dataset.
- **Human/egocentric:** UMI added; EgoDex candidate. Room for more (EgoMimic, etc.).
- **Mobile manipulation:** BEHAVIOR-1K + Mobile ALOHA are the main gaps to close.
