import base64
from collections import OrderedDict
import json
from pathlib import Path
import requests
from typing import Optional

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

from .models.actor_critic import ActorCritic
from .models.tokenizer import Encoder, Decoder, EncoderDecoderConfig, Tokenizer


HF_REPO = "https://huggingface.co/eloialonso/iris"
ROOTDIR = Path(__file__).parent


class Agent(nn.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        name = name if name is not None else _prompt()
        sd = _load_ckpt(name)
        sda = _extract_state_dict(sd, "actor_critic")
        sdt = _extract_state_dict(sd, "tokenizer")
        enc_dec_cfg = EncoderDecoderConfig(resolution=64, in_channels=3, z_channels=512, ch=64, ch_mult=(1, 1, 1, 1, 1),
                                           num_res_blocks=2, attn_resolutions=(8, 16), out_ch=3, dropout=0.0)
        self.tokenizer = Tokenizer(vocab_size=512, embed_dim=512, encoder=Encoder(enc_dec_cfg), decoder=Decoder(enc_dec_cfg))
        self.actor_critic = ActorCritic(use_original_obs=False, act_vocab_size=sda["actor_linear.weight"].size(0))
        self.tokenizer.load_state_dict(sdt, strict=False)
        self.actor_critic.load_state_dict(sda)

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device
    
    def reset(self, n: int) -> None:
        self.actor_critic.reset(n) # Reset lstm hidden state

    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        input_ac = obs if self.actor_critic.use_original_obs else torch.clamp(self.tokenizer.encode_decode(obs, should_preprocess=True, should_postprocess=True), 0, 1)
        logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        return act_token
    

######################
# Utils
######################


GAMES = ["Alien", "Amidar", "Assault", "Asterix", "BankHeist", "BattleZone", "Boxing", "Breakout", "ChopperCommand", "CrazyClimber", "DemonAttack", "Freeway", "Frostbite", "Gopher", "Hero", "Jamesbond", "Kangaroo", "Krull", "KungFuMaster", "MsPacman", "Pong", "PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown"]


def _extract_state_dict(state_dict, module_name):
    return OrderedDict({k.split(".", 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def _get_url_hf(name):
    url = f"{HF_REPO}/resolve/main/pretrained_models/{name}.pt"
    return url


def _load_ckpt(name):
    """Get checkpoint from cache or from hugging face server."""
    assert name in GAMES
    url = _get_url_hf(name)
    cache_dir = Path(f"{ROOTDIR}/checkpoints").absolute()
    cache_dir.mkdir(exist_ok=True, parents=False)
    ckpt_path = cache_dir / f"{name}.pt"
    if ckpt_path.is_file():
        print(f"{name} checkpoint already downloaded at {ckpt_path}")
    else:
        print(f"Downloading {name} checkpoint from {HF_REPO}")
        r = requests.get(url, allow_redirects=True)
        if r.status_code != 200: 
            raise ConnectionError("could not download {}\nerror code: {}".format(url, r.status_code))
        ckpt_path.write_bytes(r.content)
        print(f"Downloaded {name} checkpoint at {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


def _prompt():
    games = "\n".join([f"{i:2d}: {g}" for i, g in enumerate(GAMES)])
    print(games)
    while True:
        id = input("\nEnter game id: ")
        if id.isdigit() and 0 <= int(id) < len(GAMES): break
        print(f"\n/!\ Invalid game id ({id}), please enter an integer between 0 and {len(GAMES) - 1}")
    name = GAMES[int(id)]
    print(f"You chose {name}")
    return name
