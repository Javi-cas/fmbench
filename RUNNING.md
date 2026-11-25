# Foundation-model embeddings for the Tier 1 benchmark

This folder computes the three foundation-model representations (R1 scGPT, R2 Geneformer, R3 State SE-600M) of 2,000 control
cells per cell line, on your Mac's CPU. Scoring happens elsewhere; nothing here looks at knockdown results.

## Steps
1. VS Code: File > Open Folder > `~/scratch/fmbench` (this folder).
2. Terminal > New Terminal, then:
   ```bash
   bash setup.sh                 # downloads about 4 GB of weights and builds three environments (20 to 40 min)
   caffeinate -i bash run.sh     # the three embeddings, one after another (several hours on CPU)
   ```
3. When `run.sh` prints the `emb/` listing with `emb_R1.npz`, `emb_R2.npz` and `emb_R3.npz`, tell Claude in the chat.

If anything fails, the error is in `logs/`. You can ask Claude Code in this folder to fix it; it reads `CLAUDE.md` first.

## Known possible snags
- **SE-600M may require accepting its licence on Hugging Face**: log in at huggingface.co, open `arcinstitute/SE-600M`, accept,
  then run `uvx --from huggingface_hub hf auth login` with a read token and re-run `bash setup.sh`.
- **scGPT weights come from Google Drive**: if `gdown` is blocked, open the scGPT GitHub README, "whole-human" link, and put
  `best_model.pt`, `args.json` and `vocab.json` into `weights/scGPT_human/`.
- Geneformer and State are written for NVIDIA GPUs; `fm/common.py` redirects their GPU calls to the CPU.
