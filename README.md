# fmbench: do single-cell foundation models know which cell lines respond alike?

A pre-registered test of whether a foundation-model representation of a cell line's **unperturbed control
cells** helps predict that line's **CRISPRi knockdown responses**, for a line held out entirely.

Three models are compared against two baselines on the same pipeline, the same cells and the same gene panel.
Everything ran on a laptop CPU.

## The question

To predict knockdown responses in an unseen line, you can borrow each knockdown's measured response from
other lines and average it. The only question a representation can answer here is **how to weight those
source lines**. So:

    prediction(held-out line) = softmax(cosine similarity / T)-weighted average of five source lines' responses

The arms differ only in what the cosine is computed on:

| Arm | Representation of a line's control cells |
|---|---|
| R0 | Raw control profile, log1p CPM, 2,000 highest-variance genes |
| U | None: uniform weights (plain pooling) |
| R1 | scGPT whole-human, mean cell embedding (512 dims) |
| R2 | Geneformer V2-104M, CLS token, last layer (768 dims) |
| R3 | Arc State SE-600M, epoch-16 checkpoint (2,058 dims) |

T is chosen by inner leave-one-source-out among the sources only. Held out: **Jurkat** (1,811 knockdowns) and
**HepG2** (1,059), from Nadig et al. 2025. Sources: K562, HCT116, HEK293T and two H1 screens, 2,000 control
cells each. The metric is the competition gate's own pooled Spearman correlation of knockdown-specific
effects, with a paired bootstrap over knockdowns.

## Result

Against the raw control profile (R0), scGPT and Geneformer both help on both folds. Against **plain
pooling**, which is the sharper test, only scGPT helps, and by about 2% of the baseline's correlation.

| Paired difference in rho, 95% CI | Jurkat | HepG2 | Verdict |
|---|---|---|---|
| R1 scGPT minus U | +0.0032 [+0.0027, +0.0036] | +0.0024 [+0.0014, +0.0033] | helps |
| R2 Geneformer minus U | +0.0023 [+0.0017, +0.0030] | -0.0002 [-0.0011, +0.0007] | inconclusive |
| R3 State minus U | -0.0003 [-0.0022, +0.0017] | -0.0125 [-0.0148, -0.0103] | does not help |

Three things make that small margin hard to read as evidence that the embeddings carry context information:

1. **The inner loop preferred the flattest weighting the grid allowed** for scGPT and Geneformer. Among the
   sources, foundation-model similarity did not identify which lines respond alike.
2. **Most of the margin over R0 is R0 weighting worse than uniform**, not the models picking the right
   sources. R0 put 47% to 64% of its weight on K562, which costs it 0.019 rho against plain pooling on HepG2.
3. **The design has effectively three sources.** Most scored knockdowns are covered by exactly three of the
   five source lines, so there is little room for any weighting to beat pooling. This limitation was written
   into the plan before any embedding existed.

Controls: every arm beats its own deranged-label control by a wide margin on both folds, so every arm is
interpreted. Second-seed runs put the sampling noise inside scGPT and State two orders of magnitude below any
reported difference. A leave-one-out trim shows the scGPT margin is not robust on HepG2.

One honest confound: the three models see different numbers and selections of genes per cell (scGPT about
46% of a K562 cell's expressed genes but 22% of an H1 cell's; Geneformer truncates nearly every H1 cell and
almost no K562 cell). Nothing here separates a model from its input view.

## What is here

| File | Contents |
|---|---|
| `PLAN.md`, `PLAN*.sha256`, `AMENDMENTS.md` | The plan, hashed before any foundation-model output existed, and the implementation decisions fixed before scoring |
| `RESULTS_FM.md` | The full result, including the validity controls, seed checks and per-cell information analysis |
| `fm/embed_*.py` | The three embedding runners, redirected to CPU |
| `fm/score.py` | The scorer, using the competition gate's metric code |
| `fm/r1_*.py`, `fm/percell_info.py` | Seed, leave-one-out and information-content checks |
| `results/`, `figures/` | Scores and figures |
| `RUNNING.md`, `setup.sh`, `run*.sh` | How the embeddings were produced, including the model licence snags |

Model weights (about 29 GB), the embeddings and the cell matrices are not in the repository. `setup.sh`
downloads the weights; `extract_controls.py` and `fm/make_h5ad.py` rebuild the inputs from public sources.
`fm/ensemble_probe_*.py` additionally needs the competition pipeline's own package and will not run alone.

## Licence

Code MIT (see `LICENSE`). scGPT, Geneformer and Arc State are their authors' work under their own licences;
the knockdown datasets remain under their original terms and are not redistributed.

## A note on absolute paths

Several scripts here carry absolute paths from the environment they ran in
(`/home/claude/...`, `/mnt/user-data/...`). These are left exactly as they
were. Most of the analysis scripts are hashed by a plan manifest written
before the results existed, so rewriting their paths to look tidier would
break the record of what was actually run, which is the point of hashing them
in the first place. Where an input is not redistributed here, the script's
header and the matching RESULTS file say what it was and where it came from.

The test suite (`tests/`, run in CI) verifies every manifest in the tree on
each push, so any later change to a hashed file shows up immediately.
