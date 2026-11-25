# Foundation-model context weighting on the competition folds (Tier 1)

Pre-registered in `PLAN.md` (sha256 1bed50d8...a27, including Amendment 1) before any foundation-model output existed;
implementation details fixed in `AMENDMENTS.md` (Amendments 2 to 6) before any score was computed. This section down to
"Results" was written before the scorer was run.

**Question.** Does a foundation-model (FM) representation of a held-out line's control cells predict its CRISPRi knockdown
responses better than the same pipeline using the raw control profile (R0), and better than plain pooling (U)?

**Predictor (identical for R0 to R3).** Prediction for held-out line h = softmax(cosine / T)-weighted average of the five
source lines' per-target responses (log2 fold change vs that line's controls). Each line's vector is centred by the mean
of the five source vectors before the cosine. T is chosen from {0.01, 0.03, 0.1, 0.3, 1} by inner leave-one-source-out
(each source predicted from the other four), never using Jurkat or HepG2. A knockdown's weights are renormalised over the
sources that have it with at least 20 cells.

- **R0:** raw control profile, log1p CPM of the same 2,000 control cells on the 6,360-gene panel, 2,000 genes with the
  highest variance across the five source profiles (no PCA: five profiles allow at most four components, Amendment 1.2).
- **R1:** scGPT whole-human, mean cell embedding (dim 512). **R2:** Geneformer V2-104M, CLS, last layer (dim 768).
  **R3:** State SE-600M, epoch-16 checkpoint (dim 2,058). Each is the mean over the same 2,000 control cells per line.
- **U:** per-target mean over the five sources (equal weights).

**Metric.** The competition gate's own, code copied verbatim: pooled Spearman rho of gene-centred knockdown-specific
effects, target gene masked; paired bootstrap over knockdowns (2,000 resamples, seed 20261014). The scorer reproduces the
gate's ensemble values before scoring anything. Validity control: every arm is re-scored with its knockdown rows deranged
(gate derangement, seed 20261008); an arm that does not beat its own deranged control (95% CI excluding zero) on a fold is
not interpreted on that fold.

**Decision rules (fixed in the plan).** H1: an FM arm helps if it beats R0 on the primary gene set with the CI excluding
zero on both folds; one fold only is "inconclusive". H2: the same against U. Three FM arms against R0 form a family of
three, so H1 is shown with 95% and Bonferroni 98.3% CIs. H3 (best FM arm against the JaviCell ensemble and LRJ on the
451-gene set) is reference only.

**Data.** Held out: Jurkat (1,811 covered knockdowns) and HepG2 (1,059), Nadig et al. 2025. Sources: K562, HCT116,
HEK293T, H1 2025, H1 GSE295214. CD4 T cells are excluded from every arm (no single cells). Primary gene set: fold genes
measured in all five sources (Jurkat 6,700, HepG2 6,801).

## Stated before the results: what this design can and cannot detect

**The weighting has effectively three sources.** Amendment 1 (2026-10-24, before any embedding existed) recorded the
fold knockdowns each source covers with at least 20 cells: K562 1,959, HEK293T 1,799, HCT116 1,786, H1 2025 90,
H1 GSE295214 44. On the scored knockdowns, 1,559 of 1,811 Jurkat and 903 of 1,059 HepG2 knockdowns are covered by
exactly three sources, and 133 and 82 by only one, where every arm necessarily makes the same prediction. A representation
can therefore only move predictions by re-weighting essentially three lines. The plan named this limitation in advance
("five source lines leave little room for weighting to matter, so a null H1 may reflect too few sources rather than
uninformative embeddings") and assigned Tier 2 to address it. A null H1 here is read in that light; it is not a post-hoc
explanation.

**The arms see different inputs, and the difference varies by line** (Amendment 4). All arms use the same cells and the
same 6,360-gene panel, but:

| Arm | Context | Genes per cell the model sees | Selection |
|---|---|---|---|
| R1 scGPT | 1,200 tokens = CLS + 1,199 genes | median 1,199 | random subset of expressed genes, unseeded |
| R2 Geneformer V2-104M | 4,096 tokens = CLS + EOS + 4,094 genes | median 3,201 | top genes by median-normalised rank, deterministic |
| R3 State SE-600M | 2,048 tokens = CLS + 2,047 genes | 2,047 | top-expressed genes; random unexpressed filler when a cell has fewer, unseeded |
| R0 raw profile | n/a | 2,000 fixed genes (line mean) | highest variance across the five sources |

Median expressed panel genes per cell range from 2,586 (K562) to 5,505 (H1 2025), so scGPT sees about 46% of a K562
cell's expressed genes but 22% of an H1 2025 cell's, and Geneformer truncates every H1 2025 cell but under 1% of K562
cells. Any difference between arms is partly a difference in what each model is shown, and that difference is not the
same for every line.

**Two arms are random draws.** R1 and R3 depend on unseeded sampling inside the models (Amendments 4 and 6); R2 is
deterministic. Second-seed runs measure that noise; it is reported with the results as the noise floor for any
comparison involving R1 or R3.

## Results

Scorer: `fm/score.py`; full output `results/score_fm.txt` and `results/score_fm.json`. The metric code reproduces the
gate's ensemble values exactly (0.12468 Jurkat, 0.12154 HepG2) and the head-to-head's ensemble values on the 451-gene
set (0.1186, 0.1493). Primary gene set throughout unless stated.

| rho, primary genes | T (Jurkat) | Jurkat (1,811 knockdowns, 6,700 genes) | T (HepG2) | HepG2 (1,059 knockdowns, 6,801 genes) |
|---|---|---|---|---|
| R0 raw profile | 0.3 | 0.1339 | 0.3 | 0.1395 |
| U uniform | n/a | 0.1333 | n/a | 0.1585 |
| R1 scGPT | 1 | **0.1365** | 1 | **0.1609** |
| R2 Geneformer | 1 | 0.1356 | 1 | 0.1583 |
| R3 State | 0.3 | 0.1330 | 0.3 | 0.1459 |

T was chosen by inner leave-one-source-out on the sources only. For R1 and R2 it fell at the upper edge of the
pre-registered grid on both folds: in the inner loop, sharper weighting by FM similarity was worse at every step
(for example R1 on Jurkat: 0.068, 0.071, 0.085, 0.106, 0.125 for T = 0.01 to 1).

**Validity (deranged knockdown rows).** Every arm passes on both folds by a wide margin, so every arm is interpreted.

| Arm minus its own deranged control, 95% CI | Jurkat | HepG2 |
|---|---|---|
| R0 | +0.132 [+0.126, +0.139] | +0.136 [+0.127, +0.144] |
| U | +0.132 [+0.125, +0.138] | +0.155 [+0.146, +0.164] |
| R1 | +0.135 [+0.129, +0.141] | +0.157 [+0.148, +0.167] |
| R2 | +0.134 [+0.128, +0.141] | +0.155 [+0.146, +0.164] |
| R3 | +0.131 [+0.125, +0.138] | +0.142 [+0.134, +0.151] |

Deranged rho ranges from 0.0015 to 0.0018 on Jurkat and from 0.0035 to 0.0037 on HepG2.

**H1: FM arm against R0** (95% CI; Bonferroni 98.3% CI in brackets after it).

| Paired difference | Jurkat | HepG2 | Verdict (95% and 98.3% agree) |
|---|---|---|---|
| R1 minus R0 | +0.0026 [+0.0011, +0.0040] [+0.0007, +0.0043] | +0.0213 [+0.0180, +0.0247] [+0.0170, +0.0254] | **helps** |
| R2 minus R0 | +0.0017 [+0.0004, +0.0030] [+0.0001, +0.0034] | +0.0188 [+0.0166, +0.0208] [+0.0161, +0.0213] | **helps** |
| R3 minus R0 | -0.0009 [-0.0010, -0.0007] [-0.0010, -0.0007] | +0.0064 [+0.0058, +0.0070] [+0.0057, +0.0071] | **inconclusive** (better on HepG2 only; worse on Jurkat) |

**H2: FM arm against U (plain pooling).**

| Paired difference, 95% CI | Jurkat | HepG2 | Verdict |
|---|---|---|---|
| R1 minus U | +0.0032 [+0.0027, +0.0036] | +0.0024 [+0.0014, +0.0033] | **helps** (also at 98.3%) |
| R2 minus U | +0.0023 [+0.0017, +0.0030] | -0.0002 [-0.0011, +0.0007] | **inconclusive** (Jurkat only) |
| R3 minus U | -0.0003 [-0.0022, +0.0017] | -0.0125 [-0.0148, -0.0103] | **does not help** |
| R0 minus U (for reference) | +0.0006 [-0.0012, +0.0025] | -0.0190 [-0.0217, -0.0162] | |

**Noise floor (second seed, not used for any verdict).** R1 seed 2 chose the same T on both folds. R1 minus R1 seed 2:
-0.00001 [-0.000015, -0.000005] on Jurkat and +0.00002 [+0.00001, +0.00003] on HepG2, at least 100 times smaller than
any R1 difference above. The embedding-level disagreement (Amendment 4) was per-cell cosine 0.976 to 0.988 but
line-level source rankings fully preserved; at T = 1 the weights barely move. R3's second seed: see the section below.

**H3 (reference only, no claim): 451-gene set.** The best FM arm by mean primary rho is R1.

| rho, 451 genes | Jurkat | HepG2 |
|---|---|---|
| JaviCell ensemble | **0.1186** | 0.1493 |
| R1 scGPT | 0.1154 | **0.2035** |
| R2 Geneformer | 0.1149 | 0.1895 |
| U uniform | 0.1138 | 0.1905 |
| R0 raw profile | 0.1110 | 0.1509 |
| R3 State | 0.1102 | 0.1627 |
| LRJ (quoted from the head-to-head, no paired CI) | 0.062 | 0.123 |

R1 minus ensemble: -0.0032 [-0.0052, -0.0011] on Jurkat, +0.0542 [+0.0466, +0.0623] on HepG2. U here is not the
head-to-head's "target mean" (0.100, 0.097 on the same genes): that one averaged LRJ's own response tables over eight
training contexts including the three CD4 summary-statistic lines, while U averages the five single-cell sources'
statistics with at least 20 cells per target. The two are not interchangeable.

## Reading

1. **Pre-registered verdicts.** H1: R1 and R2 help (both folds, 95% and Bonferroni 98.3%); R3 is inconclusive.
   H2: R1 helps; R2 is inconclusive; R3 does not help. All five arms pass the validity control.
2. **What the H1 wins consist of** (observation from the fitted weights, not a pre-registered analysis). R1 and R2 chose
   T = 1, where their weights are much flatter (R1 0.14 to 0.27 per source, R2 0.06 to 0.32). R0 and R3 chose T = 0.3 and put 47% to 64%
   of the weight on K562. On HepG2 that concentration costs R0 0.019 against plain pooling. Most of the FM-over-R0
   margin, especially on HepG2, therefore measures R0 weighting worse than uniform rather than FMs identifying the right
   sources. H2 is the sharper test of informative weighting.
3. **Only R1 beats plain pooling on both folds, and by little:** +0.003 and +0.002 rho, about 2% of U's rho. The
   interval excludes zero on both folds and the seed noise is two orders of magnitude smaller, so it is not sampling
   noise, but it is a small tilt of near-uniform weights. Seed robustness (`fm/r1_seed2_verdicts.py`, same bootstrap
   resamples, `results/r1_seed2_verdicts.json`): R1 minus R1 seed 2 is -0.000010 [-0.000015, -0.000005] on Jurkat and
   +0.000022 [+0.000011, +0.000032] on HepG2, 0.3% and 0.9% of R1's margin over U (+0.0032 [+0.0027, +0.0036],
   +0.0024 [+0.0014, +0.0033]). R1 seed 2 clears H1 and H2 on both folds at 95% and 98.3%.
   **Read with `results/r1_loo/README.md` (exploratory, 2026-10-26):** under a post-hoc rule fixed before the numbers
   the margin is "concentrated": dropping the top 5% of contributing knockdowns leaves 8% of it on HepG2 (60% on
   Jurkat); a symmetric-noise reference keeps 13%, and 56% to 68% of knockdowns contribute positively, so it is a small
   margin spread thinly rather than a few large gains, but it is not robust to that trim on HepG2.
4. **The inner loop preferred the flattest weighting the grid allowed** for R1 and R2: among the sources, FM similarity
   did not identify which lines respond alike. Taken with point 2, the evidence that these embeddings carry useful
   context information beyond pooling is weak in this design.
5. **This is what the stated limitation predicts.** With effectively three sources (K562, HCT116, HEK293T), the room for
   any weighting to beat pooling is small; Tier 2 was planned for this and has not been started.
6. **Arm differences are not purely model differences.** The arms see different numbers and selections of genes per
   cell, and the gap differs by line (scGPT sees 22% to 46% of expressed genes, Geneformer truncates H1 2025 completely
   and K562 hardly at all). R1, which sees the fewest genes and a random subset, scored highest; nothing here separates
   the model from its input view.

Caveats: two folds is the same evidence standard as the head-to-head; HepG2's knockdowns target essential genes with
stereotyped responses (deranged rho stays near 0.004). An unplanned observation, not tested: U beats the ensemble on
HepG2 on both gene sets but not on Jurkat; the ensemble includes CD4 and calibrations that U does not.

## R3 second seed (Amendment 6)

The same 14,000 cells re-embedded with State's RNGs seeded to 2 (`emb/emb_R3_seed2.npz`, 279 min); a 32-cell test first
confirmed that this seeding makes State bit-identical across runs. Same procedure and quantities as R1
(`fm/r1_seed_disagreement.py R3`; `emb/R3_seed_disagreement.json`). The re-score with R3 seed 2 added reproduced every
earlier rho exactly.

| Line | Same cell across seeds, cosine median / 5th pct | Different cells, same line | Centred line vector change | Max similarity change / spread across sources | Nearest source and rank order preserved |
|---|---|---|---|---|---|
| K562 | 0.987 / 0.978 | 0.722 | 0.8% | 0.0005 / 0.45 | yes / yes |
| HCT116 | 0.991 / 0.980 | 0.759 | 0.9% | 0.0003 / 0.63 | yes / yes |
| HEK293T | 0.991 / 0.979 | 0.764 | 1.0% | 0.0007 / 0.59 | yes / yes |
| H1 2025 | 0.996 / 0.993 | 0.927 | 0.5% | 0.0005 / 0.36 | yes / yes |
| H1 GSE295214 | 0.987 / 0.979 | 0.777 | 1.0% | 0.0007 / 0.32 | yes / yes |
| Jurkat | 0.987 / 0.977 | 0.723 | 1.0% | 0.0009 / 0.77 | yes / yes |
| HepG2 | 0.990 / 0.979 | 0.743 | 0.8% | 0.0002 / 0.53 | yes / yes |

Unlike R1, where per-cell seed noise was comparable to the spread between cells of a line (0.976 to 0.988 vs 0.919 to
0.967), R3's per-cell seed noise is small next to its between-cell spread (0.99 vs 0.72 to 0.93). At the line level the source similarities move by at
most 0.0009, so even at T = 0.3 the weights change in the fourth decimal (Jurkat K562 0.5819 vs 0.5811). R3 seed 2
chose the same T on both folds. In metric units, R3 minus R3 seed 2: -0.00003 [-0.000034, -0.000027] on Jurkat and
+0.000001 [+0.000001, +0.000002] on HepG2, about 30 times smaller than R3's smallest difference from R0 (-0.0009 on
Jurkat). Neither seed check changes any verdict.

## Per-cell information content (2026-10-26, added after Tier 1; not part of any Tier 1 verdict)

Variance of the saved per-cell vectors over the seven lines (2,000 cells each), decomposed into between-line and
within-line parts; seed noise is the same-cell variance between the two seed replicates (half the squared distance
between them, paired by position as in the seed-disagreement script). Per-cell raw expression (R0) is log1p of the
cell's CPM on the 6,360-gene panel, the per-cell form of how R0's line means were built. Shares are of total variance;
dimensions differ by arm, so only shares and ratios are compared. Script `fm/percell_info.py`, output
`results/percell_info.json`, log `logs/percell_info_1026.log`.

| | Dim | Within-line share | Seed-noise share | Reproducible within-line share | Signal-to-noise | Participation ratio |
|---|---|---|---|---|---|---|
| R0, 6,360 genes | 6,360 | 0.810 | 0 | 0.810 | n/a | 402 |
| R0, its 2,000 Tier 1 genes | 2,000 | 0.731 | 0 | 0.731 | n/a | 376 |
| R1 scGPT | 512 | 0.522 | 0.170 | 0.352 | 2.1 | 12 |
| R2 Geneformer | 768 | 0.396 | 0 (code inspection) | 0.396 | n/a | 7.6 |
| R3 State | 2,058 | 0.646 | 0.027 | 0.619 | 22.6 | 274 |

- **What each model keeps per cell.** The participation ratio of the within-line covariance (its effective number of
  directions) is 402 for raw expression and 274 for State, but 12 for scGPT and 7.6 for Geneformer: those two compress
  cell-to-cell variation into roughly 8 to 12 directions. This is a property of the representations, independent of Tier 2.
- **Seed noise.** R3's per-cell structure is highly reproducible (signal-to-noise 17.6 to 25.4 on every line). R1's is
  2.1 pooled and 1.6 to 3.0 on most lines, but **0.4 on H1_2025: on that source, seed noise exceeds reproducible
  signal** (only 26% of its within-line variance reproduces, against 61% to 75% on the other lines).
- **R2's zero seed noise rests on code inspection** (deterministic gene ranking, no sampling), not on a second run.
  The risk runs one way: if R2 had seed noise, its reproducible share would be lower than 0.396, so the assumption
  cannot have flattered R2.
- **R0's reference contains count-sampling noise.** R0 has no seed, but binomial thinning (p = 0.5, a labelled
  sensitivity) estimates count-sampling noise at 0.42 of total variance, about half of R0's within-line variance; net
  of it, R0 is about 0.39 (0.36 on its 2,000 genes). The FM arms' own count noise was not measured (it would need
  thinning plus re-embedding), so the shares above are not a like-for-like comparison of biological signal.
- The cross-replicate covariance estimate of the reproducible share, which needs no equal-noise assumption, agrees
  (R1 0.3520 vs 0.3523, R3 0.6186 vs 0.6187). Consequences for Tier 2 are recorded in AMENDMENTS.md (2026-10-26).
