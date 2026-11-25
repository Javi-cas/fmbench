# Do single-cell foundation models help predict knockdown responses in an unseen cell line?

Pre-registered plan, written 2026-10-24 before any foundation-model output exists. Hash this file before running anything; amendments go at the end, dated, with the reason.

## Question

On the two honest held-out folds already used for JaviCell, LRJ and the VAE (Jurkat and HepG2, Nadig et al. 2025, GSE264667), does a foundation-model (FM) representation of the held-out line's **control cells** predict its CRISPRi knockdown responses better than:
1. the same pipeline using the raw control expression profile, and
2. the per-target mean over the same source lines?

This is the question FMs are supposed to answer in this setting: the knockdown targets were all seen in some training line, so the only thing to transfer is the cell context.

## Source and held-out data

- **Held out (test only):** Jurkat (1,811 covered knockdowns), HepG2 (1,059). Truth = the fold files `ext/fold_{jurkat,hepg2}/truth.npz`; only control cells of the held-out line are visible to any arm.
- **Sources with single-cell controls:** K562 (Replogle GWPS raw), HCT116 and HEK293T (X-Atlas/Orion), H1 2025 (Arc VCC 2025 training set), H1 GSE295214.
- **CD4 T cells are excluded from every arm**, because only summary statistics (no single cells) are available for them. The per-target mean baseline is recomputed on the same five sources, so all arms see the same source responses.
- Source responses: the same per-target statistics the ensemble uses (`atlas_data/*_statistics.npz`), log2 fold change vs that line's controls.

## Representations (Tier 1, runs on the Mac)

For each line, embed a fixed random sample of 2,000 control cells (seed 20261024) and average to one vector per line.

| Arm | Representation | Weights / license |
|---|---|---|
| R0 | Raw control profile: log1p CPM on the 2,000 most variable genes, PCA to 50 dimensions fitted on the five sources only | none |
| R1 | scGPT, whole-human checkpoint, mean cell embedding | MIT code |
| R2 | Geneformer V2 (104M), mean cell embedding | Apache-2.0 |
| R3 | State SE-600M cell embedding | Arc non-commercial licence (fine for academic use; stated in the README) |
| U | Uniform weights (per-target mean over the five sources) | none |

**Predictor (identical for R0 to R3):** prediction for held-out line h = similarity-weighted average of the source lines' per-target responses, weights = softmax(cosine similarity / T). T is chosen from {0.01, 0.03, 0.1, 0.3, 1} by inner leave-one-source-out (each source predicted from the other four, gate metric averaged), **never using Jurkat or HepG2**.

## Tier 2 (cloud GPU, only if approved after Tier 1)

Train State's transition model (ST) on the five sources with Jurkat and HepG2 held out (`[zeroshot]` config), cells capped at 50 per knockdown per source (seed 20261024). Two variants, identical except the input: **ST-SE** (SE-600M embeddings) and **ST-HVG** (2,000 HVG expression). The pair separates the architecture's contribution from the embedding's. Released ST checkpoints are not used, because their training data may include Nadig et al. Jurkat and HepG2.

## Metric and gene sets

- **Metric:** the gate's own (functions copied verbatim from `head_to_head.py`): pooled Spearman of gene-centred knockdown-specific effects, target gene masked; paired bootstrap over knockdowns, 2,000 resamples, seed 20261014.
- **Primary gene set:** fold genes measured in all five sources.
- **Secondary:** the 451-gene common set from the head-to-head, so the arms can be placed next to the ensemble, LRJ, the target mean and the VAE.
- **Validity control:** every arm is re-scored with knockdown rows deranged (gate derangement, seed 20261008). An arm must beat its own shuffled control (CI excluding zero) to be interpreted.

## Decision rules (fixed now)

- **H1 (primary):** an FM arm "helps" if it beats R0 on the primary gene set with the 95% CI excluding zero **on both folds**. One fold only = "inconclusive".
- **H2:** an FM arm vs U (does any context weighting beat plain pooling?), same rule.
- **H3 (reference only, no claim):** best FM arm vs the JaviCell ensemble and LRJ on the 451-gene set.
- **Tier 2 rule:** ST-SE vs ST-HVG, and ST-SE vs R0 and U, same both-folds rule.
- All arms are reported whatever the outcome. Three FM arms against R0 are a family of three: CIs for H1 are Bonferroni-adjusted to 98.3% and both are reported.

## Contamination record

- SE-600M was pretrained on observational data that includes GEO-mined collections (scBaseCount); Jurkat and HepG2 control cells may be in it. Control cells are inputs at test time anyway, so this is disclosed, not disqualifying. Knockdown responses of the held-out lines are not in any FM's pretraining as far as documented.
- scGPT whole-human and Geneformer were pretrained on CELLxGENE observational data, with no CRISPR responses.

## Known limitations, stated in advance

Five source lines leave little room for weighting to matter, so a null H1 may reflect too few sources rather than uninformative embeddings. Tier 2 addresses that by letting ST use the embeddings per cell. Two folds is the same evidence standard as the head-to-head.

## Compute and time

- Tier 1: about 1 week on the Mac (CPU or Apple GPU). About 8 GB of model weights; embedding 14,000 cells per model.
- Tier 2: estimated 10 to 30 GPU-hours on one rented A100/H100-class GPU (about US$20 to $80). A one-hour pilot measures the real speed before committing.
- Run after the Boltz-2 benchmark finishes, so the two don't compete for memory.

## Outputs

`fmbench/` folder in the lrj-perturb repo: scripts with this plan's hash in the header, per-arm predictions, results JSON, a figure, and RESULTS_FM.md written in the same style as the other benchmarks.

## Amendment 1 (2026-10-24, before any arm was scored or any embedding computed)

Reasons are mechanical: the data and the plan text could not both be followed as written.

1. **Common gene panel.** The three Replogle/Nadig lines measure only 8,246 to 9,623 filtered genes, while X-Atlas and H1 measure 18,080 to 38,584. Every representation (R0 to R3) is computed on the **6,360 genes measured in all seven lines**, so no arm can tell lines apart by which genes were recorded.
2. **R0 without PCA.** PCA to 50 dimensions fitted on five source profiles is impossible (at most 4 components). R0 = log1p CPM on the 2,000 genes (from the common panel) with the highest variance across the five source control profiles, no PCA.
3. **Centring.** For every arm, each line's vector is centred by the mean of the five source vectors before cosine similarity (applied identically to R0 to R3).
4. **Source responses.** Only targets with at least 20 cells in a source are used from that source; a knockdown's weights are renormalised over the sources that have it. Inner leave-one-source-out uses the same knockdowns and the primary gene set.
5. **Control cells.** X-Atlas controls = `gene_target == "Non-Targeting"` and `pass_guide_filter`. H1 GSE295214 controls follow the repository's GEM rules (UMI >= 5,000, mito <= 0.20, top guide >= 5 UMI and >= 3x second); GEMs are visited in seeded random order until at least 8,000 controls are collected, then 2,000 are sampled (a seeded cluster sample, because reading all 85 GEMs is impractical).
6. **Observed source coverage (data fact, no results):** fold knockdowns with >= 20 cells: K562 1,959, HEK293T 1,799, HCT116 1,786, H1 2025 90, H1 GSE295214 44. The two H1 sources therefore contribute to few knockdowns; this was not known when the plan was written.
