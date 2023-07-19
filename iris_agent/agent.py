import base64
from collections import OrderedDict
import json
from pathlib import Path
import requests
from typing import Optional
from urllib.request import urlretrieve

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

from models.actor_critic import ActorCritic
from models.tokenizer import Tokenizer


class Agent(nn.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        name = name if name is not None else _prompt()
        sd = _load_ckpt(name)
        sda = _extract_state_dict(sd, 'actor_critic')
        sdt = _extract_state_dict(sd, 'tokenizer')
        cfg = OmegaConf.load('default.yaml')
        cfg.actor_critic.act_vocab_size = sda['actor_linear.weight'].size(0)
        self.tokenizer = Tokenizer(**instantiate(cfg.tokenizer))
        self.actor_critic = ActorCritic(**cfg.actor_critic)
        self.tokenizer.load_state_dict(sdt, strict=False)
        self.actor_critic.load_state_dict(sda)

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        input_ac = obs if self.actor_critic.use_original_obs else torch.clamp(self.tokenizer.encode_decode(obs, should_preprocess=True, should_postprocess=True), 0, 1)
        logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        return act_token
    

######################
# Utils
######################


GAMES = ['Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone', 'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner', 'Seaquest', 'UpNDown']


def _extract_state_dict(state_dict, module_name):
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def _get_url_lfs(name):
    first_url = f'https://api.github.com/repos/eloialonso/iris_pretrained_models/contents/pretrained_models/{name}.pt'
    r = requests.get(first_url)
    assert r.status_code == 200
    x = base64.b64decode(json.loads(r.content)['content']).decode()
    sha = x.split('sha256:')[1].split('\n')[0]
    size = int(x.split('size ')[1].split('\n')[0])
    post_url = 'https://github.com/eloialonso/iris_pretrained_models.git/info/lfs/objects/batch'
    post_json = json.dumps({"operation": "download", "transfers": ["basic"], "objects": [{"oid": sha, "size": size}]})
    post_headers = {"Accept": "application/vnd.git-lfs+json", "Content-type": "application/json"} 
    r = requests.post(post_url, post_json, headers=post_headers)
    assert r.status_code == 200
    download_url = dict(json.loads(r.content.decode()))['objects'][0]['actions']['download']['href']
    return download_url


def _load_ckpt(name):
    """Get checkpoint from cache or from git lfs server."""
    assert name in GAMES
    url = _get_url_lfs(name)
    cache_dir = Path('checkpoints')
    cache_dir.mkdir(exist_ok=True, parents=False)
    ckpt_path = cache_dir / f'{name}.pt'
    if ckpt_path.is_file():
        print(f'{name} checkpoint already downloaded at {ckpt_path}')
    else:
        print(f'Downloading {name} checkpoint from https://github.com/eloialonso/iris_pretrained_models')
        urlretrieve(url, str(ckpt_path.absolute()))
        print(f'Downloaded {name} checkpoint at {ckpt_path}')
    return torch.load(ckpt_path, map_location='cpu')


def _prompt():
    games = '\n'.join([f'{i:2d}: {g}' for i, g in enumerate(GAMES)])
    print(games)
    while True:
        id = input('\nEnter game id: ')
        if id.isdigit() and 0 <= int(id) < len(GAMES): break
        print(f'\n/!\ Invalid game id ({id}), please enter an integer between 0 and {len(GAMES) - 1}')
    name = GAMES[int(id)]
    print(f'You chose {name}')
    return name
