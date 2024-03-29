# Offline Reinforcement Learning with Implicit Q-Learning

This repository contains the official implementation of [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169) by [Ilya Kostrikov](https://kostrikov.xyz), [Ashvin Nair](https://ashvin.me/), and [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/).

If you use this code for your research, please consider citing the paper:
```
@article{kostrikov2021iql,
    title={Offline Reinforcement Learning with Implicit Q-Learning},
    author={Ilya Kostrikov and Ashvin Nair and Sergey Levine},
    year={2021},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## How to run the code

### Install dependencies

```bash
conda env create -f conda_env.yaml
conda activate iql
pip install flax optax
pip install --upgrade 'jax[cuda12_pip]' #-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # for cuda 12
pip install --upgrade 'jax[cuda11_pip]' #-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # for cuda 11
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
conda deactivate && conda activate iql
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

### Run training

Locomotion
```bash
python train_offline.py --env_name=halfcheetah-medium-expert-v2 --config=configs/mujoco_config.py
```

AntMaze
```bash
python train_offline.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000
```

Kitchen and Adroit
```bash
python train_offline.py --env_name=pen-human-v0 --config=configs/kitchen_config.py
```

Finetuning on AntMaze tasks
```bash
python train_finetune.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_finetune_config.py --eval_episodes=100 --eval_interval=100000 --replay_buffer_size 2000000
```

## Misc
The implementation is based on [JAXRL](https://github.com/ikostrikov/jaxrl).
