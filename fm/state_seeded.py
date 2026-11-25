"""Run the `state` CLI with the Python, numpy and torch RNGs seeded (Amendment 6). Usage: state_seeded.py <seed> emb transform ...
Runs in State's own interpreter; everything after the seed is passed to `state` unchanged. The __main__ guard matters:
State's DataLoader spawns workers that re-import this file, and they must not re-run the CLI."""
import sys, random, numpy as np, torch
if __name__ == "__main__":
    seed = int(sys.argv[1]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sys.argv = ["state"] + sys.argv[2:]
    from state.__main__ import main
    sys.exit(main())
