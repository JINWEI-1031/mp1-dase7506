# DASE7506 MP1 — submission

**Frozen full-test BPB: `1.4574991`** (supplied baseline: 2.1012597).

| | measured | limit |
|---|---:|---:|
| CPU scoring time (test split, FP32, 4 threads), **worst of 5 alternated reps** | **4.29×** baseline (range 3.95× – 4.29×) | ≤ 5× |
| Peak evaluation RAM | 2,205.0 MiB | ≤ 4 GiB |
| Inference assets, uncompressed | 24.938 MiB | ≤ 64 MiB |

The BPB above is the `bpb` field of the complete-test JSON produced by
`code/evaluate.py --split test --device cpu --precision fp32` on the submitted
checkpoint (exact value `1.4574990646`). The gate rows are the shipped configuration,
measured on the test split with the method frozen; the same file measured on the
validation split gives **4.61×**. §4 explains why the ratio is reported as a range
gated on its worst reading, and why those two splits do not agree.

---

## What is here

| path | contents |
|---|---|
| `REPORT_en_v7.md` / `REPORT_en_v7.pdf` | report (method, comparisons, ablation, critical analysis) |
| `GUIDE.md` | the assignment guide, as supplied |
| `code/` | release code, plus the modules this submission adds (see below) |
| `bundle/checkpoint.pt` | **the submitted checkpoint** (26,149,082 B) — distributed as this tag's release asset; put the downloaded file here (or point the flag at wherever it was saved) before running §2. One file, plus the implementation module it names |
| `.gitattributes` | sets `* -text` for every file, so nothing undergoes line-ending conversion on clone; the data, tokenizer and evaluator therefore arrive byte-identical and reproduce the hashes below whatever the cloning machine's `core.autocrlf` is |
| `.gitignore` | keeps `bundle/`, `__pycache__/` and `*.pyc` out of the repository; the checkpoint is distributed as a release asset instead |

The search, packing, measurement and verification scripts behind the report are **not**
published here: they share a directory with machine-access credentials. They are
available on request, and every number they produced is quoted in `REPORT_en_v7.md` together
with the command that produced it — including the packing step of §3, whose commands are
inlined there rather than shipped as a script.

### Which parts of `code/` are mine

`code/PACKAGE_MANIFEST.json` lists 17 release files. Their status in this submission:

| status | count | files |
|---|---:|---|
| byte-identical to the manifest | 14 | `common.py`, `evaluate.py`, `model.py`, `train.py`, `requirements.txt`, `configs/baseline.json`, `tests/test_contract.py`, `data/*` (5 files), `.gitignore`, `RUN_LOG_TEMPLATE.csv` |
| modified, as the release invites | 1 | `student.py` |
| completed as the release invites | 1 | `code/README.md` — §4 shipped with three blank measurement rows (`CPU time ≤5× baseline:` / `Peak RAM ≤4 GiB:` / `Inference assets ≤64 MiB uncompressed:`); they are now filled with the measured values |
| differs from the manifest, cause not established | 1 | the guide (`132409af…` in the manifest vs `fa62a767…`, 4,180 B, in every copy on disk) |

The 14 byte-identical files include everything `GUIDE.md` requires be kept unchanged:
data, tokenizer and evaluator.

Added by this submission: `ensemble.py` (probability averaging over members),
`agent7_fastens.py` (**the module the submitted checkpoint names** — see below),
`pack_ensemble.py` (packs members into one self-describing checkpoint), `train_v2.py`
(the training entry point the members actually used — see §3), `configs/member_m35_*.json`
/ `member_m37_*.json` / `member_m47_*.json` (the three member definitions, copied
verbatim from their run directories), and `gen_quality.py` / `sample.py` (reporting aids,
not used at evaluation time).

**What the evaluator actually loads.** The submitted checkpoint records
`implementation = "agent7_fastens"`, so `common.make_model` imports that module and the
load path is:

    agent7_fastens.py  ->  student.py  ->  model.py  (+ data/tokenizer)

So both files I added or modified are on that path, and this is stated plainly rather
than minimised:

* `student.py` was extended as `code/README.md` §2/§3 invites — "edit `student.py` and
  supporting files" — by adding a library of 26 optional architecture switches
  (`student.FLAG_SPEC`) and a `VariantEnsemble` wrapper. **Every switch defaults to off**,
  so the supplied factory's default behaviour is unchanged; the members' configs turn on
  exactly two of them (`pos_mode`, `qk_norm`), recorded in their config files.
* `agent7_fastens.py` is a drop-in `build_model` that wraps `student.VariantEnsemble` and
  **numerically must not change the score**. It exists for one reason: the CPU-time gate.
  Measured back to back on the same box, the identical checkpoint scores `1.4404595`
  either way, but the worst-of-5 time ratio is **5.3277× without it (FAILS)** versus
  **4.6129× with it (OK)**. It is therefore a correctness requirement for gate R1, not an
  optional optimisation. §4 gives the numbers, including the falsified alternatives.

`tests/` holds the supplied `test_contract.py` and nothing else, so
`python -m unittest discover -s tests` reports the 5 contract tests quoted in the report.
Tests written during this work (for `ensemble.py` and for the search-time architecture
switches) are not published: they exercise code paths the submitted predictor never
loads, and a test the evaluator cannot use is only a way to fail for the wrong reason.

Both manifest deviations are prose files, and neither touches the predictor. `code/README.md`
differs because §4's three resource rows were blank in the release and are now filled in
(the release asks for exactly this); it names none of the files added here. The guide's
deviation pre-dates this work — `132409af…` in the manifest against `fa62a767…` (4,180 B)
in every copy on disk, one upload timestamp, identical bytes in both locations — and its
cause is **not established**, so it is recorded here rather than explained away. Note the
path follows the release layout too: `code/README.md` links to `../guide/GUIDE.md`, which
resolves in the release tree but not here, where the same bytes sit at `../GUIDE.md`.

The three submitted members are three independently trained GPTs, all of shape
`w192 d4 h4` and differing only in seed (35, 47, 37), each trained on 157,286,400
targets. The packed file holds their **EMA** weights. Each member's slice inside
`bundle/checkpoint.pt` was compared against that run's three checkpoint files and is
bit-identical to `checkpoint.pt` and `checkpoint_ema.pt` and **different from**
`checkpoint_raw.pt` — the raw file is the negative control that shows the comparison
distinguishes rather than always agreeing. Two independent criteria agree (name-
normalised weights, and a name-blind multiset of the tensor bytes). Independently, the
members' validation BPB recomputed offline from their per-token NLL dumps reproduces the
runs' recorded `validation_ema` (1.5398848 / 1.5398679 / 1.5390429) to seven decimals,
whereas the raw-weight readings differ by 0.039. Their predicted
next-token distributions are averaged at inference (probability averaging, `K = 3`). The
packed checkpoint records each member's config and is rebuilt by the evaluator without
retraining.

Everything the evaluator needs is inside `bundle/checkpoint.pt` plus the supplied
`code/model.py` and this repository's `code/agent7_fastens.py`, `code/student.py` and
`code/ensemble.py`.

---

## 1. Install

Python 3.12. From this repository's root:

```bash
cd code
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v      # expect: 5 tests, OK
```

No API key, no pretrained weights and no network access are needed; data and
tokenizer are included.

## 2. Score the submitted checkpoint (no retraining)

```bash
cd code
python evaluate.py --checkpoint ../bundle/checkpoint.pt \
    --device cpu --precision fp32 --threads 4 --split test --output peer-test.json
```

Expected: `"bpb": 1.4574991` (exact `1.4574990646`), `"targets": 428405`,
`"utf8_bytes": 1292013`. This is the exact command a peer would run (`code/README.md` §5).

The same command with `--split validation` was re-run from a copy of this repository alone
(no training outputs, no search tooling) on a **different machine and a different PyTorch**
(2.12.1 vs the 2.7.1 used for the reported run) and returned
`bpb = 1.4404595460468363` against the reported `1.4404595450138231` — a relative
difference of 7×10⁻¹⁰, i.e. fp32 reassociation noise. The JSON also echoed
`checkpoint_sha256`, `implementation_sha256`, `evaluator_sha256` and `tokenizer_sha256`
matching the values recorded here, so the shipped bytes are the bytes that produced the
reported score.

Note that `peak_allocated_gb` / `peak_reserved_gb` in that JSON are CUDA counters and read
`0.0` on CPU; the RAM figure in the table above comes from an external RSS measurement
(§4), not from this file.

## 3. Reproduce the training (optional — §2 needs no retraining)

The members were **not** trained with the supplied `train.py`: that script hardcodes
`lr=1e-3` and a cosine schedule with 100 warmup steps (`train.py:39`, `train.py:52`),
whereas every member records `lr=0.005` and the `d2z` schedule. They were trained with
`train_v2.py`, included here. The flags below are the ones each member's own
`run_spec.json` records, re-issued from `code/`:

```bash
# all three members: d4/w192/h4, differing only in --seed (35, 47, 37)
# and in --config (the config files carry the two architecture switches)
for S in 35 47 37; do
python train_v2.py --implementation student --device cuda --precision fp32 \
    --config configs/member_m${S}_d4w192h4.json --seed $S \
    --steps 19200 --batch-size 32 --threads 1 --ema-decay 0.999 \
    --lr 0.005 --schedule d2z --weight-decay 0.1 --decay-scope all \
    --warmup-steps 100 --min-lr-frac 0.1 --d2z-decay-frac 0.2 \
    --dropout 0.0 --beta2 0.999 --parent-checkpoints none \
    --run-dir runs/member_m${S}
done
```

`steps × batch_size × 256 = 19200 × 32 × 256 = 157,286,400` targets per member, which is
the `train_tokens` each run recorded. The three config files are copied verbatim from the
members' own run directories. Pack them into the single submitted file — `--out` is a run
**directory**, and `--member` is repeated once per member:

```bash
python pack_ensemble.py \
    --member runs/member_m35/checkpoint.pt \
    --member runs/member_m47/checkpoint.pt \
    --member runs/member_m37/checkpoint.pt \
    --out ../bundle --overwrite
```

Do not pass `pack_ensemble.py --search-cost`: the submitted file was packed with it
unset, and setting it would change the file's bytes. (The `--search-cost` recorded in
each member's own `run_spec.json` is a different flag on `train_v2.py`, and is recorded
in metrics only — `train_v2.py:471-475`.)

`pack_ensemble.py` does **not** write the dispatch field or the level. Two more writes
are required, or the pack scores the same but fails the CPU gate (§4). Run from `code/`:

```bash
python - <<'PY'
import hashlib, pathlib, torch
p = pathlib.Path('../bundle/checkpoint.pt')
pl = torch.load(p, map_location='cpu', weights_only=False)
pl['implementation'] = 'agent7_fastens'
pl['config']['l4_level'] = 4
pl['implementation_sha256'] = hashlib.sha256(
    pathlib.Path('agent7_fastens.py').read_bytes()).hexdigest()
torch.save(pl, p)
PY
```

Reloading with `weights_only=True` then reports `implementation = 'agent7_fastens'`,
`resolve_level(config) = 4`, and weights bit-identical to the pre-write pack; the
`implementation_sha256` of the shipped file is
`de64fa1e33cb8504840a3d8f6a50a8922378e932eb92da2ce892896496691de4`, and the pack's
`checkpoint_sha256` is
`94b5af1ea5282bf86ec6543cacb3bd90138895b7516f626f8cebfac31af03259`.

## 4. Gate measurement, and why the CPU row is a range

R1 is a **ratio**, and its denominator is measured live by the grader
(`code/README.md:59`: the reference machine yields 5.92 s, and this is "not a fixed time
allowance"). Across the gate rehearsals below, the same baseline command varied between
5.22 s and 7.06 s on the training box, so the ratio's variance lives mostly in the
denominator and no single reading is decisive. The measurement is therefore **alternated**
(baseline, ensemble, baseline, ensemble, …) and judged on the **MAX** ratio, since the
submission is scored once.

**The three-arm comparison (validation split).** Three arms, same checkpoint, back to
back on the same otherwise-idle box, 5 alternating pairs each (`_rehearsal_fastens.py`):

| arm | dispatch | worst-of-5 ratio | verdict |
|---|---|---:|---|
| control | `student`, level 0 | 5.3277× | **FAILS** |
| level 1 | `agent7_fastens`, `l4_level=1` | 4.8666× | OK |
| level 4 | `agent7_fastens`, `l4_level=4` | **4.6129×** | **OK (shipped)** |

All three arms score `bpb 1.4404595` in every repeat, so the fastens levels are
score-invariant and affect only speed. (BPB is printed to 7 decimals while two
independently packed copies of the same weights diverge only in the 10th, so the stronger
identity check is the member-weight fingerprint — REPORT_en_v7.md §7.2.) That is what makes `agent7_fastens` safe to ship
despite being on the load path: it changes *how* the same arithmetic is executed. But it
is also **load-bearing for the gate** — the control arm, which is the same file with the
stock `student` dispatch, fails outright at 5.3277×.

**The gate row (test split).** With the method frozen, the shipped configuration was
re-measured on the split the grader scores, under the harness's `--frozen` flag:

| | five per-repeat readings | min / median / **MAX** |
|---|---|---|
| baseline | 7.056 / 6.536 / 6.942 / 6.891 / 6.669 s | — |
| ensemble | 27.923 / 27.519 / 27.497 / 27.235 / 28.616 s | — |
| **ratio** | 3.9572 / 4.2101 / 3.9611 / 3.9523 / 4.2910 | 3.9523 / 3.9611 / **4.2910** |

All five returned `bpb 1.4574991` — the frozen score — so these repeats measured time,
not the score. Peak RAM 2,205.0 MiB; assets 24.938 MiB.

Note that the two splits **disagree**: 4.61× on validation against 4.29× on test, for the
same file. That is not a split effect. Normalised by target count both terms are flat
across the two sessions — ensemble 65.42 vs 64.79 µs/target, baseline 15.67 vs 15.92 —
and the test split is simply 13.8% larger (428,405 targets against 376,599), so its
absolute seconds are proportionally bigger. The gap between the two MAX readings lives in
the *pairing*: the gate is a MAX over five alternated pairs, and validation's worst pair
came from an unusually fast baseline (5.4147 s) while test's came from an unusually slow
ensemble (28.616 s). In the validation session the baseline's per-target cost swung 21%
across five reps, against 4.7% for the ensemble — which is exactly why the gate is quoted
as the MAX of alternated pairs rather than as one number.

Two things this does **not** establish, stated because they are easy to overclaim:

* Absolute ratios are only comparable **within** one session. The container reads a
  host-global `/proc/loadavg` (~20 with zero local runnable processes), so other tenants
  share the CPU. The control arm's failing 5.3277× is a same-session comparison against
  the shipped arm's 4.6129×, and that is the claim being made — not that 5.3277 would
  reproduce on the grader's machine.
* The margin is not robust to an arbitrarily fast denominator. The fastest baseline
  reading in any gate rehearsal this session was 5.2232 s; at that denominator the
  test-split numerator of 28.616 s would read 5.48× and fail. It passes as long as the
  grader's baseline is at least `28.616 / 5.0 = 5.72 s`; the reference figure is 5.92 s.
  That is a margin statement, not a guarantee.

An earlier three-arm run reported `5.0186×` for the control arm. That number is **not**
usable and is not used above: a queued training wave had started and was contending for
the CPUs, so the "trainers have exited" check fired while the box was immediately
re-contended. The run was repeated on an idle box and the table above is the repeat.

Full search and training costs are disclosed in [Report](REPORT_en_v7.md) §10.2.

---

## Reused work

`GUIDE.md` §3 asks that reused work be acknowledged. Two kinds are reused here, and
neither is presented as new.

**Supplied code.** The baseline GPT, training loop, evaluator, tokenizer and data are
provided by the assignment and are byte-identical to `PACKAGE_MANIFEST.json`; the
per-file status is tabulated under "Which parts of `code/` are mine" above. Nothing
supplied was modified — the two changed files are additions the release explicitly
invites.

**Published methods.** The method builds on techniques introduced in the literature.
They are marked inline in the report and listed in its §14 References:

| used for | source |
|---|---|
| rotary position embedding (RoPE), in all three shipped members | Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864 |
| query–key normalisation (QK-norm) | Dehghani et al., *Scaling Vision Transformers to 22 Billion Parameters*, ICML 2023, arXiv:2302.05442 |
| averaging weights of seeds trained separately ("model soup") — measured, then closed | Wortsman et al., *Model soups*, ICML 2022, arXiv:2203.05482 |
| permutation-symmetry alignment (weight matching / activation matching) | Ainsworth et al., *Git Re-Basin: Merging Models modulo Permutation Symmetries*, ICLR 2023, arXiv:2209.04836 |
| the Muon optimiser — measured, then killed | Jordan et al., 2024, kellerjordan.github.io/posts/muon; Liu et al., *Muon is Scalable for LLM Training*, arXiv:2502.16982 |
| averaging weights along one run's trajectory (EMA) in the shipped recipe | Polyak & Juditsky, *Acceleration of Stochastic Approximation by Averaging*, SIAM J. Control Optim. 30(4): 838-855, 1992 |
| the loss-barrier measure used in the weight-merging section | Garipov et al., *Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs*, NeurIPS 2018, arXiv:1802.10026 |

No pretrained weights, no external training text and no evaluation-time network access
are used. All three members are trained from random initialisation on the supplied data.

---

## AI assistance

**Substantive AI assistance was used in this work, and is disclosed here as
required.** Specifically, an AI assistant was used to draft and iterate the search,
enumeration, packaging and measurement tooling in `_plans/`; to help design and run the
training-recipe and architecture searches; to write `agent7_fastens.py`; and to draft
this README and the report.

**This includes code the evaluator loads.** The submitted checkpoint's `implementation`
field names `agent7_fastens`, a module written with AI assistance, so that module is on
the scoring path — as is `student.py`, which it imports. The AI did not write the supplied
`model.py`, `evaluate.py`, `common.py` or the data and tokenizer, which are
byte-identical to the manifest, and did not train the members from any pretrained
weights: all three start from random initialisation.

The assistant's `agent7_fastens.py` was verified not to change the score before it was
shipped: all three arms of the §4 table produce the same `bpb 1.4404595` on the same
checkpoint, and the module provides only numerically-equivalent re-associations
(`log_softmax(..., out=)`, `matmul(..., out=)`, persistent buffers). It changes *how* the
same arithmetic is executed, not what is computed.

All experimental results reported, including every number in `REPORT_en_v7.md`, were
produced by executing code on the hardware described, and each decisive claim was
checked against raw program output. Where a result could not be established, it is
reported as unmeasured rather than estimated.

**Costs**, per `GUIDE.md` §3 and `code/README.md` §3: 400.2 GPU-hours of training
across 1008 logged runs in total; the three submitted members account for 2.02 GPU-hours
of that (`process_seconds` 2,426.5 / 2,426.6 / 2,427.6 s, read from their own
`metrics.json`). Checkpoint ancestry: all three members trained from random
initialisation (`parent_checkpoints = 'none'`).

## Data attribution

WikiText-2 (Merity et al., arXiv:1609.07843), text by Wikipedia contributors,
distributed under CC BY-SA 3.0 and the GNU Free Documentation License. The supplied
splits, tokenizer and their hashes are unchanged (`code/data/manifest.json`).
