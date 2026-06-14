# video_dedup_samples

A small synthetic **LeRobot v3** dataset (10 episodes, 2 cameras: `top` + `wrist`,
30 frames each) for exercising `forge dedup` across interesting conditions.

Regenerate with:

```bash
python scripts/gen_video_dedup_samples.py
forge dedup sample_data/video_dedup_samples
```

Scenes are low-frequency (sinusoidal gratings + a moving Gaussian blob + a
per-identity colour tint) so they survive H.264 — random noise would be blurred
away by the codec and make every episode look alike.

## Episodes & intended behavior

Four scene identities (A, B, C, D) drive the content; `top` and `wrist` use
different identities within an episode.

| Episode | Content | At default `--threshold 0.10` |
|---|---|---|
| `episode_000000` | scene A | **kept** — representative of the A cluster |
| `episode_000001` | exact copy of A | dropped — dist **0.000** |
| `episode_000002` | A + mild pixel noise (re-encode) | dropped — dist **~0.07** |
| `episode_000003` | A + brightness **+50** | dropped — dist **~0.06** (pHash is brightness-robust) |
| `episode_000004` | scene B | **kept** — unique |
| `episode_000005` | B time-shifted **6 frames** | **kept** — dist **~0.20** to B |
| `episode_000006` | scene C | **kept** — unique |
| `episode_000007` | `top`=C (same as 000006), `wrist`=unique | **kept** — dist **~0.53** |
| `episode_000008` | scene D | **kept** — representative |
| `episode_000009` | exact copy of D | dropped — dist **0.000** |

Result: **6 unique kept, 4 near-duplicates dropped, 2 clusters.**

### What each case demonstrates

- **Exact copies** (1, 9) — caught at distance 0, even through independent H.264 re-encodes.
- **Re-encode noise** (2) and **brightness shift** (3) — caught; pHash keys on
  low-frequency structure, so mild noise and global brightness barely move it.
- **Time-shifted take** (5) — *not* caught. Keyframes are compared by position, so
  a temporal shift raises the distance above threshold. This is the known
  alignment trade-off of position-aligned hashing (a shift-tolerant matcher would
  be the fix if this matters for your data).
- **Partial multi-camera match** (7) — *not* caught. `top` matches 000006 exactly
  (dist 0.000) but `wrist` differs (dist ~0.53); the engine takes the **max across
  shared cameras**, so an episode is a duplicate only if *every* shared camera matches.

Tune `--threshold` to see the boundaries move: at `0.05` the noise/brightness
near-dupes (2, 3) survive; at `0.25` the time-shifted take (5) gets absorbed.

## Note: a forge round-trip quirk this dataset works around

forge's LeRobot v3 **writer** flushes a new chunk per episode, so each episode's
video is its own `chunk-XXX/file-000.mp4` (one episode per file). But it stamps
`meta/info.json` with the *multi-episode* template `...file-{file_index}.mp4`,
which makes the **reader** seek by global frame index and return black frames for
every episode after the first. The generator rewrites `video_path` in `info.json`
to the single-episode template so the reader uses the local `frame_index`. See
`_fix_video_path_template()` in `scripts/gen_video_dedup_samples.py`. This is a
writer/reader inconsistency worth fixing in forge proper, independent of dedup.
