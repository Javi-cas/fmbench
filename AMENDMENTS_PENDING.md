# Proposed amendments

No items are awaiting a decision. P1 below was adopted as Amendment 6 on 2026-10-25 and is kept for the record.

## P1 (2026-10-25): ADOPTED as Amendment 6 (user decision, 2026-10-25). Measure R3 sampling noise with a second seed, as Amendment 4 does for R1
- Finding: State SE-600M inference is also random and unseeded. `state.emb.data.loader.sample_cell_sentences` breaks
  expression ties with `torch.randperm` before ranking, and fills the 2,048-token sentence of any cell with fewer than
  2,047 expressed genes with unexpressed genes drawn by `torch.randint` (with replacement). This affects K562 21.4%,
  Jurkat 12.4%, HepG2 8.8%, H1_GSE295214 4.0%, HCT116 3.5%, HEK293T 0.6%, H1_2025 0% of cells. Neither
  `state emb transform` nor `fm/embed_state.py` sets a seed. Amendment 4's premise that only R1 is noisy is incomplete.
- Proposal: keep `emb/emb_R3.npz` as the primary R3 and re-embed the same cells with a fixed torch seed to
  `emb/emb_R3_seed2.npz`; report the same disagreement statistics as R1. Geneformer (R2) is deterministic and needs none.
- Cost: about 340 min of CPU (the R3 run time), run after R1 seed 2 to avoid competing for memory.
- Open detail: `state emb transform` exposes no seed flag, so this needs a small wrapper that seeds torch before calling
  the same transform entry point.
