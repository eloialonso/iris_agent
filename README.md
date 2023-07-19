# IRIS Agent

Minimal repo to use the policy of any [IRIS](https://github.com/eloialonso/iris) agent trained on the 26 Atari 100k games.  

## Useful Links

- [Paper](https://openreview.net/forum?id=vhFu1Acb0xb)
- [Codebase](https://github.com/eloialonso/iris)
- [Pretrained models](https://github.com/eloialonso/iris_pretrained_models)


## Setup

Install with pip:

```bash
pip install git+https://github.com/eloialonso/iris_agent.git
```

## Usage

Create agent:

```python
from iris_agent import Agent

agent = Agent('Breakout') # specify game name, or  
agent = Agent()           # choose from list of games

```

Use the policy:

```python
import torch

n = 1
agent.reset(n)
obs = torch.randn(n, 3, 64, 64) # obs is a (n, 3, 64, 64) tensor in [0.,1.], and you should use the standard atari wrappers (see IRIS codebase)
act = agent.act(obs) # act is a (n,) long tensor in {0,...,num_actions-1}
```
