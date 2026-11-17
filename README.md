# Prioritized Generative Replay

[Renhao Wang](https://renwang435.github.io/), [Kevin Frans](https://kvfrans.com/), [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/), [Alexei A. Efros](http://people.eecs.berkeley.edu/~efros/)

[[`arXiv`](https://arxiv.org/abs/2410.18082)] [[`BibTeX`](#Citing)]

## Installation

This repository is based heavily off the release code of [SynthER](https://github.com/conglu1997/SynthER?tab=readme-ov-file#setup).
It has been ported to a modern stack (Python 3.10+, `gymnasium`, the official `mujoco` and `dm_control` packages), so
`mujoco-py`, `gym`, `d4rl` and `dmcgym` are no longer needed, and it installs natively on Apple Silicon Macs, Linux and Windows.

```bash
git clone --recurse-submodules https://github.com/renwang435/pgr.git
cd pgr
# if you already cloned without --recurse-submodules:  git submodule update --init --recursive

python3.11 -m venv .venv && source .venv/bin/activate    # or: conda create -n pgr python=3.11 && conda activate pgr
pip install -e .          # installs this repo, the REDQ submodule (as `redq`) and all dependencies
bash scripts/smoke_test.sh   # ~1-2 min on CPU: tiny end-to-end run incl. diffusion training + sampling
```

Notes:
- On Apple Silicon use a native **arm64** Python (`python -c "import platform; print(platform.machine())"` should print `arm64`);
  the `mujoco` package refuses to run under Rosetta/x86_64 Python.
- Device: CUDA is used if available, otherwise CPU. Choose explicitly with `--device cpu|cuda|mps` or `SYNTHER_DEVICE`.
  `mps` (Apple GPU) is opt-in and experimental; it is not selected automatically.
- Simulation now uses MuJoCo 3.x instead of the MuJoCo 2.2 used for the paper, so numbers may differ slightly from the original runs.
- Training logs are written to `synther/REDQ/data/` (unchanged from the original code).

## Running Instructions

```bash
python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1'
```

```bash
python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25'
```

```bash
python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25'
```

## <a name="Citing"></a>Citing PGR

```BibTeX
@inproceedings{wang2025prioritized,
  title={Prioritized Generative Replay},
  author={Renhao Wang and Kevin Frans and Pieter Abbeel and Sergey Levine and Alexei A Efros},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=5IkDAfabuo}
}
```

## License and Acknowledgements

This codebase inherits all licenses from the public release of [SynthER](https://github.com/conglu1997/SynthER).
