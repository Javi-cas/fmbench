# Context for Claude Code in this folder

Task: produce three foundation-model embeddings for a pre-registered benchmark (PLAN.md), on this Mac's CPU.
Outputs: `emb/emb_R1.npz` (scGPT whole-human), `emb/emb_R2.npz` (Geneformer V2-104M, CLS, emb_layer -1),
`emb/emb_R3.npz` (State SE-600M); each holds one mean vector per cell line (keys: K562, HCT116, HEK293T, H1_2025,
H1_GSE295214, Jurkat, HepG2), written by `fm/common.py:save`. Entry points: `setup.sh`, then `run.sh`.

## Rules (the benchmark is pre-registered)
- Never edit `PLAN.md`, `PLAN*.sha256`, or anything in `cells/` or `data/`. Never read fold truth or score anything.
- Fixing installation, download, version or device problems is fine (packages, paths, cuda->cpu, batch size, threads).
- Anything that changes **what is computed** (cells, genes, normalisation, model checkpoint or size, embedding layer or
  mode, pooling) is NOT a fix: stop, write the proposed change and its reason in `AMENDMENTS_PENDING.md`, and tell the user.
- Same model, same inputs for all seven lines. Do not drop a line or a model to make something run; report it instead.
- Keep every log in `logs/`. A separate Boltz-2 job (`~/scratch/boltzbench`) is running on this Mac: do not stop it.
