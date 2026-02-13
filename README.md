# DrQ-v2 Reproduction & N-step Ablation Study

**Author:** Bangcheng Wang | **Course:** CSC415

This repository is a reproduction of the DrQ-v2 algorithm based on the [official codebase](https://github.com/facebookresearch/drqv2). The main contribution is an **n-step return ablation study** on `cartpole_swingup` and `walker_walk` tasks from the DeepMind Control Suite.

## What's Changed (vs. Official Repo)

The original code targets Python 3.8 + mujoco_py (deprecated). We updated it to run on modern environments:

- **Python 3.12 + mujoco 3.x + dm_control 1.x** (no license required)
- **hydra-core 1.3.2** (1.1.0 incompatible with Python 3.12; added `version_base='1.1'`)
- **Windows support** (skip `MUJOCO_GL=egl` on Windows)
- **dtype fix** for newer dm_control (cast action/reward/discount to float32)
- **replay_buffer fix** for numpy/Python 3.12 compatibility
- **Ablation script** (`scripts/run_ablation.sh`) for n-step experiments

## Ablation: N-step Returns

| Config | Values |
|---|---|
| Tasks | `cartpole_swingup`, `walker_walk` |
| N-step | 1, 3, 5, 10 |
| Seeds | 1, 2, 3 |
| Frames | cartpole: 500K, walker: 1M |

```sh
# Dry run (print commands only)
bash scripts/run_ablation.sh --dry-run

# Run all experiments
bash scripts/run_ablation.sh
```

---

*Below is the original README from the DrQ-v2 authors.*

---

# DrQ-v2: Improved Data-Augmented RL Agent

This is an original PyTorch implementation of DrQ-v2 from

[[Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning]](https://arxiv.org/abs/2107.09645) by

[Denis Yarats](https://cs.nyu.edu/~dy1042/), [Rob Fergus](https://cs.nyu.edu/~fergus/pmwiki/pmwiki.php), [Alessandro Lazaric](http://chercheurs.lille.inria.fr/~lazaric/Webpage/Home/Home.html), and [Lerrel Pinto](https://www.lerrelpinto.com).

<p align="center">
  <img width="19.5%" src="https://i.imgur.com/NzY7Pyv.gif">
  <img width="19.5%" src="https://imgur.com/O5Va3NY.gif">
  <img width="19.5%" src="https://imgur.com/PCOR9Mm.gif">
  <img width="19.5%" src="https://imgur.com/H0ab6tz.gif">
  <img width="19.5%" src="https://imgur.com/sDGgRos.gif">
  <img width="19.5%" src="https://imgur.com/gj3qo1X.gif">
  <img width="19.5%" src="https://imgur.com/FFzRwFt.gif">
  <img width="19.5%" src="https://imgur.com/W5BKyRL.gif">
  <img width="19.5%" src="https://imgur.com/qwOGfRQ.gif">
  <img width="19.5%" src="https://imgur.com/Uubf00R.gif">
 </p>

## Method
DrQ-v2 is a model-free off-policy algorithm for image-based continuous control. DrQ-v2 builds on [DrQ](https://github.com/denisyarats/drq), an actor-critic approach that uses data augmentation to learn directly from pixels. We introduce several improvements including:
- Switch the base RL learner from SAC to DDPG.
- Incorporate n-step returns to estimate TD error.
- Introduce a decaying schedule for exploration noise.
- Make implementation 3.5 times faster.
- Find better hyper-parameters.

<p align="center">
  <img src="https://i.imgur.com/SemY10G.png" width="100%"/>
</p>

These changes allow us to significantly improve sample efficiency and wall-clock training time on a set of challenging tasks from the [DeepMind Control Suite](https://github.com/deepmind/dm_control) compared to prior methods. Furthermore, DrQ-v2 is able to solve complex humanoid locomotion tasks directly from pixel observations, previously unattained by model-free RL.

<p align="center">
  <img width="100%" src="https://imgur.com/mrS4fFA.png">
  <img width="100%" src="https://imgur.com/pPd1ks6.png">
 </p>

## Citation

If you use this repo in your research, please consider citing the paper as follows:
```
@article{yarats2021drqv2,
  title={Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning},
  author={Denis Yarats and Rob Fergus and Alessandro Lazaric and Lerrel Pinto},
  journal={arXiv preprint arXiv:2107.09645},
  year={2021}
}
```
Please also cite our original paper:
```
@inproceedings{yarats2021image,
  title={Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels},
  author={Denis Yarats and Ilya Kostrikov and Rob Fergus},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=GY6-6sTvGaf}
}
```

## Instructions (Updated)

MuJoCo 3.x is free and open-source. No license file is needed.

### Setup
```sh
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install mujoco dm_control hydra-core==1.3.2 omegaconf==2.3.0 hydra-submitit-launcher \
    numpy termcolor imageio imageio-ffmpeg tb-nightly pandas matplotlib opencv-python-headless
```

On Linux, also install system dependencies:
```sh
sudo apt update && sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

### Train
```sh
# Note: use "task@_global_=" syntax with hydra 1.3
python train.py "task@_global_=quadruped_walk"

# On Windows, add replay_buffer_num_workers=1
python train.py "task@_global_=cartpole_swingup" replay_buffer_num_workers=1
```

### Monitor
```sh
tensorboard --logdir exp_local
```

### Original Instructions (deprecated)

<details>
<summary>Click to expand original setup (Python 3.8 + mujoco_py)</summary>

Install [MuJoCo](http://www.mujoco.org/):

* Obtain a license on the [MuJoCo website](https://www.roboti.us/license.html).
* Download MuJoCo binaries [here](https://www.roboti.us/index.html).
* Unzip the downloaded archive into `~/.mujoco/mujoco200` and place your license key file `mjkey.txt` at `~/.mujoco`.

Install dependencies:
```sh
conda env create -f conda_env.yml
conda activate drqv2
```

Train the agent:
```sh
python train.py task=quadruped_walk
```
</details>

## License
The majority of DrQ-v2 is licensed under the MIT license, however portions of the project are available under separate license terms: DeepMind is licensed under the Apache 2.0 license.
