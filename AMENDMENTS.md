# Amendments after PLAN.md Amendment 1 (sha256 1bed50d8...a27)

## Amendment 2 (2026-10-24, before any foundation-model arm was scored)
- Implementation bug: `fm/embed_state.py` picked the checkpoint by sorting file names, which chose
  `se600m_epoch4.ckpt` (an intermediate checkpoint; "epoch16" sorts before "epoch4"). The plan specifies the released
  SE-600M model. The epoch-4 embeddings were never scored; they are kept in `unused_R3_epoch4/` for the record.
- Fix: call `state emb transform --model-folder weights/SE-600M` without `--checkpoint`, exactly as the model card shows,
  and record from the log which checkpoint the tool loads. R3 is recomputed.
- Correction (2026-10-26): "exactly as the model card shows" is loose; the card names no checkpoint, and with
  `--model-folder` the CLI takes `glob("*.ckpt")[-1]`, i.e. directory order (here `[epoch4, epoch16]`), so the selection
  was arbitrary and verified after the fact, not by design. Independent check on 2026-10-26: the checkpoint used,
  `se600m_epoch16.ckpt`, is identical to the repo's released `model.safetensors` in all 238 tensors (epoch 4 matches it in
  1), so the results use the released model. The epoch-4 set stays in `unused_R3_epoch4/` as the record.

## Amendment 3 (2026-10-25, before any foundation-model arm was scored)
Environment and device fixes for R1 and R2 after both failed at import on 2026-10-24. Nothing about what is computed
changes (same cells, genes, normalisation, checkpoints, embedding layer and mode, pooling). Logs: `logs/fix_1025_*`,
`logs/R{1,2}_*_1024_failed.log`, smoke tests `logs/smoke_{scgpt,gf}.py`.
- R1 scGPT: pyarrow 25.0.1 -> 14.0.2 in `.venv_scgpt` (datasets 2.14.4 needs `pa.PyExtensionType`; scgpt 0.2.4 pins
  datasets<3.0 and only imports it). `fm/embed_scgpt.py`: macOS has no `os.sched_getaffinity`, which scGPT uses only to
  set DataLoader `num_workers`; it is shimmed so data loading runs in the main process (`num_workers=0`).
- R2 Geneformer: the setup download of `ctheodoris/Geneformer` (revision 1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5) was
  partial: `geneformer/` had the four gc104M `.pkl` dictionaries but no Python files, so an empty package was installed.
  The missing `.py` files (and the 30M dictionaries) were fetched at the same revision; existing files were not touched.
  Installed with the repo's `requirements.txt`, which pins transformers==4.46 (setup.py does not; transformers 5.20.2
  was installed and lacks internals Geneformer imports). `fm/common.py`: the cuda->cpu shim now also covers torch tensor
  factories called with `device="cuda"` (Geneformer `get_embs`).
- Scheduling, deviating from PLAN.md "Compute and time" ("run after the Boltz-2 benchmark finishes"): R1 and R2 run while
  Boltz-2 is still running on the same Mac, as R3 already did. This affects run time only, not outputs.
- Recorded, not changed: scGPT `embed_data` truncates cells with more than 1,199 expressed genes to a random 1,199-gene
  subset (`max_length=1200`, collator `sampling=True`) and the script sets no seed, so R1 is not bit-reproducible.

## Amendment 4 (2026-10-25, before any foundation-model arm was scored and before the second R1 run exists)
**R1 sampling noise is a confound, measured with a second seed.** scGPT's input is random in two places: the 51-bin
expression binning breaks ties at bin edges with `np.random.rand` (`scgpt.preprocess._digitize`, every cell), and cells
with more than 1,199 expressed genes get a random 1,199-gene subset (`torch.randperm`, collator `sampling=True`;
`max_length=1200` = CLS + 1,199 genes). 96.9% to 100% of cells per line exceed 1,199 expressed panel genes. Any gap
between R1 and another arm therefore contains sampling variance that would look like a model property.
- `emb/emb_R1.npz` stays the primary R1 (unseeded run of 2026-10-25). The same 14,000 cells are re-embedded with
  `np.random.seed(2); torch.manual_seed(2)` to `emb/emb_R1_seed2.npz` (`fm/embed_scgpt.py 2`, `run_r1_seed2.sh`). A
  smoke test showed that these two seeds make the run bit-identical, so they cover all of scGPT's randomness.
- Disagreement, fixed now (`fm/r1_seed_disagreement.py`, embedding level only, no responses or folds): per cell, cosine
  of the same cell across runs (median, 5th percentile, min, per line) against the median cosine between different cells
  of the same line; per line, cosine of the mean vectors raw and after Amendment 1.3 centring, and the relative change
  of the centred vector; and the change in each line's centred cosine to each source (the predictor's input) against
  that line's spread across sources, with whether the source rank order is preserved.
- These numbers are reported **in the results next to every R1 comparison**, as the noise floor for R1-versus-arm
  differences, not under limitations. Scoring both R1 runs is not added here; that would need a further amendment.

**Input gene budgets differ by arm and must be stated wherever arms are compared.** All arms see the 6,360-gene panel
(scGPT vocabulary covers 6,357, Geneformer V2 dictionary 6,359, State protein embeddings 6,360). Median expressed panel
genes per cell: K562 2,586, HCT116 3,350, HEK293T 4,010, H1_2025 5,505, H1_GSE295214 2,812, Jurkat 2,770, HepG2 3,013.
- scGPT: 1,200-token context = CLS + 1,199 genes, a **random** subset of expressed genes (median 1,199 seen).
- Geneformer V2-104M: 4,096-token context = CLS + EOS + 4,094 genes, the top genes by median-normalised rank,
  deterministic (median 3,201 seen). Truncation hits H1_2025 100%, HEK293T 45.2%, HCT116 17.6%, HepG2 11.2%,
  Jurkat 6.5%, H1_GSE295214 1.7%, K562 0.9% of cells.
- State SE-600M: 2,048-token sentence = CLS + 2,047 genes, top-expressed first (ties broken by `torch.randperm`);
  cells with fewer than 2,047 expressed genes (K562 21.4%, Jurkat 12.4%, HepG2 8.8%, others 0% to 4%) are padded with
  unexpressed genes **drawn at random with replacement**. No seed is set, so R3 is also one random draw (see
  AMENDMENTS_PENDING.md). A further dataset token is appended; output dim 2,058.
- The fraction of a cell's expressed genes that each model sees therefore differs by arm and by line (for example scGPT
  sees about 22% of an H1_2025 cell's expressed genes and 46% of a K562 cell's). Part of any arm difference is a
  difference in input, not only in model.

## Amendment 5 (2026-10-25, written before any score was computed): how the scorer implements the plan
Everything below is implementation detail where PLAN.md and Amendment 1 are silent; no design element changes.
`fm/score.py` implements it; results go to `results/score_fm.{json,txt}` and `RESULTS_FM.md`.
- **Truth:** `ext/fold_{jurkat,hepg2}/truth.npz`, found on the mounted share at `/Volumes/VCC26/scratch/vcc/ext/` (the
  same share `extract_sources.py` read as `~/mnt/VCC26`). The scorer first reproduces the gate's ensemble rho
  (0.125 Jurkat, 0.122 HepG2, as `head_to_head.py`) and aborts if that fails. Metric functions are copied verbatim from
  `head_to_head.py` (itself verbatim from the gate).
- **Knockdowns scored:** each fold's covered set (`covered.npy`: Jurkat 1,811, HepG2 1,059), as in the head-to-head.
  Every covered knockdown has at least one of the five sources.
- **Source responses:** `data/src_*.npz`, checked on 2026-10-25 against the original statistics files (identical target
  lists, cell counts and NaN pattern; recomputed rows agree to 1e-7).
- **R0 profile:** the same 2,000 control cells per line as the FM arms (`h5ad/all_lines.h5ad`, 6,360-gene panel), CPM
  computed per cell on the panel, averaged per line, then log1p; the 2,000 genes with the highest variance across the
  five source profiles; no PCA (Amendment 1.2). `data/controls_cpm.npz` (CPM on each line's full, differently sized gene
  axis, written before Amendment 1) is not used, because Amendment 1.1 requires every representation on the panel.
- **Similarity:** centred cosine (Amendment 1.3); the centre is the mean of the five source vectors throughout,
  including inside leave-one-source-out. Held-out vectors never enter the centre.
- **Prediction:** for each knockdown and gene, softmax(cos / T) weights over the sources with a finite value
  (target present with >= 20 cells, gene measured), renormalised over those sources (Amendment 1.4). U uses equal weights.
- **T selection (per arm, per fold):** for each source s, predict s from the other four on the fold's covered knockdowns
  that s has and at least one other source has, on the fold's primary gene set, scored with the gate metric against s's
  own responses (target gene masked); average rho over the five sources with equal weight; ties go to the smaller T.
  Jurkat and HepG2 responses and embeddings are never used. The secondary gene set uses the T chosen on the primary set.
- **Gene sets:** primary = fold genes measured in all five sources (Jurkat 6,700, HepG2 6,801). Secondary = the 451-gene
  head-to-head set, rebuilt as fold genes on LRJ's gene axis with finite VAE predictions; used only if it reproduces 451
  genes and the ensemble's head-to-head rho (0.119, 0.149).
- **Bootstrap:** one generator (seed 20261014) per fold and gene set, the same 2,000 knockdown resamples for every
  comparison (paired); percentile CIs. Deranged control: prediction rows permuted with the gate derangement (seed
  20261008) over the fold's full knockdown list, then restricted to covered knockdowns, as in the head-to-head.
- **Decisions:** an arm is interpreted on a fold only if it beats its own deranged control (95% CI excluding zero). H1 for
  an FM arm: "helps" if valid and FM minus R0 has its CI above zero on both folds; above zero on one fold only =
  "inconclusive"; otherwise "does not help". It is evaluated with both the 95% and the Bonferroni 98.3% CI; if the two
  disagree, both verdicts are stated. H2 the same against U. H3: the FM arm with the highest mean primary rho, against
  the ensemble on the secondary set (paired CI); LRJ is quoted from `results/h2h/head_to_head.json` without a paired CI,
  since its predictions were not stored.
- **Noise floor (reported, never used for H1 to H3):** the second-seed runs (`emb_R1_seed2`, and `emb_R3_seed2` per
  Amendment 6) are scored by the same procedure, with their own T, and each primary-minus-seed-2 difference is reported
  with its paired CI.

## Amendment 6 (2026-10-25, user decision, before any score was computed): R3 second seed (adopts P1)
- State SE-600M inference is random (Amendment 4), so R3 gets the same check as R1: the same 14,000 cells re-embedded with
  `random`, numpy and torch seeded to 2 (`fm/state_seeded.py` wraps the unchanged `state emb transform` call;
  `fm/embed_state.py 2`) to `emb/emb_R3_seed2.npz`. `emb/emb_R3.npz` stays the primary R3. The disagreement is reported
  with the same quantities as `fm/r1_seed_disagreement.py`.

## 2026-10-26: per-cell information content measured; Tier 2 not run, for compute, premise intact
- Purpose. Tier 2 (PLAN.md) is ST trained on SE-600M embeddings (ST-SE, the R3 representation) against ST
  trained on 2,000 HVG expression (ST-HVG), and PLAN.md's "Known limitations" gives its rationale: a null
  H1 at Tier 1 may reflect too few sources, which Tier 2 addresses by letting ST use the embeddings per
  cell. Whether that premise holds is testable from the saved per-cell embeddings and the two seed
  replicates with no GPU, so it was tested (fm/percell_info.py, results/percell_info.json).
- Measured, reproducible within-line share of variance (within-line minus same-cell seed noise, over
  total), with per-cell raw expression on the same 6,360 genes as reference:
  R0 0.810, R1 0.352, R2 0.396, R3 0.619. Seed-noise share: R1 0.170, R3 0.027, R2 zero.
  Signal-to-noise: R1 2.1 pooled (1.6 to 3.0 per line, 0.4 on H1_2025), R3 22.6 (17.6 to 25.4 every line).
  Participation ratio of the within-line subspace: R0 402, R3 274, R1 12, R2 7.6.
- A decision rule was fixed before these numbers: a GPU is justified only for a representation whose
  reproducible within-line share is comparable to R0's AND which showed Tier 1 signal. That rule drops
  Tier 2. **The rule is recorded and then set aside as ill-posed, for two reasons found in the data.**
  1. Criterion (a) is not determined by what is on disk. It subtracted embedding-seed noise from the FM
     arms but nothing from R0, whose within-line variance is about half count-sampling noise (binomial
     thinning, 0.42 of total, a labelled sensitivity). Net of it R0 is about 0.39, comparable to R1's
     0.352, which reverses the verdict for R1. The FM arms' own count noise is unmeasured, since it would
     need thinning plus re-embedding. The two defensible comparisons bracket opposite answers for R1.
  2. Criterion (b) is Tier 1's line-mean result, which is the test PLAN.md identifies as too weak and that
     Tier 2 was pre-registered to supersede. Declining Tier 2 because Tier 1 was null is circular against
     Tier 2's stated purpose.
- What the measurement does establish. R3, the embedding Tier 2 is built on, carries abundant reproducible
  per-cell structure (share 0.619, signal-to-noise 22.6, 274 effective dimensions). **Tier 2's premise is
  therefore supported, not refuted.** Separately, for R1 and R2 averaging to a line mean discards comparatively little, which follows from
  their reproducible within-line shares of 0.352 and 0.396 against 0.619 for R3 and 0.810 for R0, and not
  from the participation ratios. Independently of that, the per-cell variance R1 and R2 do retain occupies
  roughly 8 to 12 effective directions against 274 for R3 and 402 for R0, so what a per-cell model could
  draw on in those two is both comparatively small and low-dimensional. The two statistics answer different
  questions: the share says how much per-cell structure survives averaging, the participation ratio says
  how many directions it occupies. That distinction is between representations, not a verdict on Tier 2.
- Decision. Tier 2 is not run, because it needs a GPU and this work stays on this Mac: 10 to 30 A100-hours
  is of order weeks to months of CPU here, and the machine is committed to the Boltz-2 queue. This is a
  compute constraint and is recorded as one. It is NOT a finding that per-cell training on these embeddings
  would not help; the measurement points the other way for R3. The source-count limitation stands
  unresolved, and ST-SE against ST-HVG remains the pre-registered experiment that would resolve it,
  now with its premise measured.
- No GPU was rented. No Tier 1 arm, score, input or analysis changed.
