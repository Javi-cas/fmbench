# Is R1's margin over pooling broad or concentrated? (exploratory, post hoc)

**Status: exploratory.** Prompted by a post-hoc question about the benchmark's only positive Tier 1 finding (R1 beats
plain pooling U on both folds by about 2% of rho; `RESULTS_FM.md`). Not pre-registered; Tier 1's record
(`RESULTS_FM.md`, `results/score_fm.*`, `AMENDMENTS.md`) is untouched. This section was written before the numbers.

**Rule (fixed by the user before the numbers).**
- **Broad:** share of knockdowns contributing positively > 0.55 AND removing the top 5% of contributors leaves the
  margin positive at more than half its original value, on both folds.
- **Concentrated:** removing the top 5% takes the margin below half, or reverses its sign, on either fold.
- Anything else: **equivocal**.

**User's prediction:** broad but weakly so, positive share about 0.55 to 0.60, margin surviving the top-5% trim at
about 60% to 70%. **Second question, direction written down before computing:** R1 leans towards K562 relative to
pooling on Jurkat (K562 share 0.44 vs 0.38) and away from it on HepG2 (0.34 vs 0.39). If line-appropriate weighting is
the mechanism, R1's per-knockdown gain should be larger where K562's response is *close* to the truth on Jurkat
(negative correlation with K562-to-truth distance) and where it is *far* on HepG2 (positive correlation).

**Method.** Primary gene set, Tier 1 knockdowns, T values and predictions (each arm is rebuilt and must reproduce
`results/score_fm.json` exactly). contribution(k) = margin with all knockdowns minus margin with k removed. Removing k
drops k's row of the gate's per-knockdown rank statistics, with ranks fixed once on the full set, which is how the gate's
own bootstrap treats resamples; a full recompute (re-centring and re-ranking without k) is checked on 10 knockdowns.
Trimmed margins remove the top 1%, 5% and 10% by contribution jointly; their 95% CIs use the Tier 1 bootstrap resamples
(seed 20261014) with the removed knockdowns dropped from each resample. Concentration: Gini over |contribution|, and the
number of knockdowns whose removal halves the margin. Comparators: R2 and R3 against U.

**Caveat on the trim criterion, stated before the numbers.** Removing the top 5% of contributors selects on noise as
well as signal: when the mean effect per knockdown is small relative to its spread, trimming the top tail can cut a
uniformly spread margin by more than half. A random 5% removal is reported beside it as a reference (it should leave the
margin about unchanged), and R2 and R3 show what the statistics look like for arms that do not beat pooling.

## Results

Scripts `fm/r1_loo.py` (leave-one-out, as specified), `fm/r1_additive.py` (exact additive decomposition); outputs
`r1_loo.{json,txt}`, `r1_additive.{json,txt}`, `r1_k562_distance.json`; logs `logs/r1_loo_1026.log`,
`logs/r1_additive_1026.log`, `logs/r1_additive_corr_1026.log`. Every arm reproduces Tier 1's rho exactly.

### A problem with the specified contribution, found in the output

Leave-one-out contributions to a pooled correlation sum to about zero by construction (the jackknife property: the mean
of the leave-one-out estimates is about the full estimate). Every run shows "sum of contributions 0.00000". So
contribution(k) measures each knockdown's influence relative to the average, not its share of the margin: its positive
share sits near 0.5 whether the margin is broad or not, and "knockdowns needed for half the margin" by cumulative sum
degenerates to 1. Trims (remove knockdowns, recompute) remain meaningful. The pooled Spearman does have an exact additive
decomposition: its numerator is a sum over knockdowns of rank cross-products about the global mean ranks, so
d_k = a_k(arm) / D(arm) - a_k(U) / D(U) sums exactly to the margin (checked to 1e-12). Both are reported; the rule is
applied to both. The full-recompute check (re-centring and re-ranking without k) agrees with the fixed-rank
leave-one-out in sign and order; on HepG2 the fixed-rank values run about 20% smaller for the largest contributors.

### Distribution and trims, R1 minus U (primary genes)

| | Jurkat (1,811 kd) | HepG2 (1,059 kd) |
|---|---|---|
| Margin | +0.00317 | +0.00239 |
| Leave-one-out: median contribution, IQR | -4.2e-7 [-2.5e-6, +2.1e-6] | -1.2e-6 [-7.7e-6, +7.0e-6] |
| Leave-one-out: positive share | 0.441 | 0.466 |
| **Additive d_k: median, IQR** | +1.1e-6 [-5.1e-7, +3.2e-6] | +1.7e-6 [-5.3e-6, +7.9e-6] |
| **Additive d_k: positive share** | **0.676** | **0.562** |
| Without top 1% | 0.89 of original, +0.0028 [+0.0024, +0.0032] | 0.74, +0.0018 [+0.0009, +0.0027] |
| **Without top 5%** | **0.60**, +0.0019 [+0.0016, +0.0022] | **0.08**, +0.0002 [-0.0006, +0.0010] |
| Without top 10% | 0.37, +0.0012 [+0.0009, +0.0015] | -0.48, -0.0012 [-0.0019, -0.0004] |
| Without a random 5% | 1.01 | 0.94 |
| Symmetric-noise reference, top 1 / 5 / 10% | 0.88 / 0.61 / 0.40 | 0.75 / 0.13 / -0.35 |
| Knockdowns whose removal halves the margin | 127 (7.0%) | 24 (2.3%) |
| Gini of absolute d_k | 0.546 | 0.519 |

Trims ranked by additive d_k give the same fractions within 0.02 (Jurkat 0.89 / 0.60 / 0.37, HepG2 0.74 / 0.08 / -0.48).
Largest contributors are ribosome-biogenesis and translation knockdowns on both folds (Jurkat POLRMT, LAS1L, RPS24, RRS1,
EIF4E; HepG2 RBM28, ISG20L2, PDCD11, PES1, RPL7L1).

**Comparators.** R2 minus U on Jurkat (+0.00232, an arm that beats pooling on that fold only): additive positive share
0.600, top-5% trim leaves 0.31 against a symmetric reference of 0.27, Gini 0.530. Where R2 and R3 do not beat pooling
(margins -0.0002 to -0.0125) the trimmed fractions are ratios of near-zero or negative numbers and carry no information;
their Gini values (0.53 to 0.55) match R1's (0.52 to 0.55). R3 on HepG2 has additive positive share 0.324, a broad deficit.

### Verdict against the rule

| Criterion | Leave-one-out (as specified) | Additive d_k |
|---|---|---|
| Positive share > 0.55 on both folds | No (0.441, 0.466; near 0.5 by construction) | Yes (0.676, 0.562) |
| Top-5% trim leaves > half, positive, on both folds | No: Jurkat 0.59, **HepG2 0.07** | No: Jurkat 0.60, **HepG2 0.08** |
| **Rule verdict** | **Concentrated** (HepG2 trim below half) | **Concentrated** (HepG2 trim below half) |

**By the rule fixed in advance, R1's margin is concentrated.** On HepG2, removing the top 5% of contributors (53 of
1,059 knockdowns) leaves 7% to 8% of the margin, with a CI including zero; 24 knockdowns halve it.

**What the evidence says about mechanism, beside the verdict.** The HepG2 result is what trimming produces for a broad
margin at this noise level: a perfectly symmetric distribution with the same mean and spread (mirrored d_k) keeps 13%
after the same trim, against 8% observed; Jurkat matches its reference too (0.60 vs 0.61). The right tail is no heavier
than the left, more than half the knockdowns contribute positively on both folds, and R1's concentration (Gini) equals
the comparators'. So R1's margin is not carried by a few knockdowns with large idiosyncratic gains; it is a small shift
spread over most knockdowns, too small relative to per-knockdown variation to survive removing the top 5% on HepG2. The
rule's trim criterion cannot tell those two apart at this effect size, which is the caveat stated above before the
numbers. Practically: on HepG2 the margin is not robust to dropping its best 5% of knockdowns, and that belongs next to
the 2% figure whatever its cause.

**The user's prediction**: positive share 0.55 to 0.60 (additive: 0.68 Jurkat, 0.56 HepG2) and a top-5% trim keeping
60% to 70% (Jurkat 0.60, yes; HepG2 0.08, no).

### Second question: does R1 gain where line-appropriate weighting should matter? (descriptive)

Spearman correlation between d_k and K562-to-truth distance (1 - Spearman of K562's response with the held-out truth,
target masked), and mean d_k by distance tertile (near, mid, far):

| Arm (K562 share minus U's) | Jurkat | HepG2 |
|---|---|---|
| R1 (+0.06 Jurkat, -0.05 HepG2) | **-0.310** (+3.8e-6, +1.1e-6, +0.4e-6) | **+0.198** (+0.1e-6, +3.7e-6, +3.0e-6) |
| R2 (+0.10, +0.09) | -0.251 | -0.093 |
| R3 (+0.37, +0.26) | -0.098 | -0.018 |

The direction written down in advance holds for R1 on both folds: on Jurkat, where R1 leans towards K562, its gain is
largest where K562 resembles the truth; on HepG2, where it leans away, its gain is largest where K562 differs. In all six
arm-fold cells the sign follows the arm's K562 lean relative to pooling, and R1 is the only arm whose sign flips between
folds. A response-strength confound (strong knockdowns have both larger |d_k| and closer K562 agreement) would push every
arm the same way, so it cannot produce R1's flip, though it may inflate the Jurkat magnitudes. Leave-one-out
contributions give the same signs (R1 -0.171, +0.153). Two folds; descriptive only.
