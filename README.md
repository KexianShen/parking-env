# parking-env


[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/KexianShen/parking-env)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[<img src="https://img.shields.io/badge/%F0%9F%A4%97%20models-Huggingface-F8D521">](https://huggingface.co/shenkexian/parking-env-model)

Parking-env is a gymnasium-based environment for reinforcement learning, written in a single Python file and accelerated by Numba. The environment is designed to simulate the task of parking a vehicle in a parking lot, where the agent controls the steering angle and the speed to park the vehicle successfully.

<p align="center">
    <img src="https://raw.githubusercontent.com/KexianShen/parking-env/media/ppo-discrete-0.gif?raw=true"><br/>
    <em>PPO agent with discrete actions</em>
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/KexianShen/parking-env/media/ppo-multidiscrete-0.gif?raw=true"><br/>
    <em>PPO agent with multidiscrete actions</em>
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/KexianShen/parking-env/media/ppo-continuous-0.gif?raw=true"><br/>
    <em>PPO agent with continuous actions</em>
</p>

## Installation
To install the stable version,

```bash
pip install parking-env
```

To install the current version with additional scripts in editable mode,

```bash
git clone https://github.com/KexianShen/parking-env.git
cd parking-env
pip install -e .
```

## Usage
Pre-trained models are uploaded to [Hugging Face Hub](https://huggingface.co/shenkexian/parking-env-model) with detailed notes.

To use parking-env, you can code as follows:

```python
import gymnasium as gym

env = gym.make("Parking-v0", render_mode="human")

env.reset()
terminated = False
truncated = False

while not terminated and not truncated:
    action = 2
    obs, reward, terminated, truncated, info = env.step(action)

```

## Credits
Parking-env is heavily inspired by the [HighwayEnv](https://github.com/eleurent/highway-env) environment, and some of its code was adapted for use in parking-env.

Additionally, parking-env uses the algorithms provided in [CleanRL](https://github.com/vwxyzjn/cleanrl), a collection of clean implementations of popular RL algorithms.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
