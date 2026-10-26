"""Explicit scripted baseline and learned policy adapter."""
from dataclasses import dataclass
import numpy as np
import json
from pathlib import Path
from .config import CONFIG

@dataclass
class Personality:
    desire: float=.65
    curiosity: float=.55
    confidence: float=.55
    playfulness: float=.45
    reciprocity: float=.65

    def vector(self):
        return np.clip([self.desire,self.curiosity,self.confidence,self.playfulness,self.reciprocity],0,1).astype(np.float32)


def scripted_controls(t,side,personality,tip,other_tip,*,sim=None):
    """Four targets: base rotation, base hinge, middle hinge, tip hinge."""
    if sim is not None:
        from .interaction import interaction_controls
        return interaction_controls(sim,side)
    p=personality
    sign=1 if side=='a' else -1
    phase=0 if side=='a' else 1.1
    goal_y=p.reciprocity*other_tip[1]+(1-p.reciprocity)*.025*np.sin(t*.45+phase)
    rotate=sign*.3*(goal_y-tip[1])+.012*p.curiosity*np.sin(t*.5+phase)
    rotate+=(1-p.desire)*sign*.12+.008*p.playfulness*np.sin(t*.8+phase)
    lift=-.04-.04*p.confidence-.10*(1.23-tip[2])
    middle=.06+.035*p.curiosity*np.sin(t*.4+phase)
    curl=-.06+.06*p.playfulness*np.sin(t*.65+phase)
    curl+=.08*p.reciprocity*(other_tip[2]-tip[2])
    return np.clip(np.array([rotate,lift,middle,curl])*CONFIG['scripted_amplitude_gain'],-CONFIG['target_limit_rad'],CONFIG['target_limit_rad'])


def policy_action_scale(model_path):
    """Replay saved networks using the action scale they were trained with."""
    path=Path(model_path)
    side=next((s for s in ('a','b') if path.stem.startswith(s+'_') or ('_'+s+'_') in path.stem),None)
    metadata=path.parent/f'{side}_training.json'
    if side and metadata.exists():
        return float(json.loads(metadata.read_text())['config']['target_limit_rad'])
    return CONFIG['target_limit_rad']


class PolicyAgent:
    def __init__(self,model_path=None,seed=0):
        self.model=None
        self.action_scale=CONFIG['target_limit_rad']
        if model_path:
            import torch
            torch.set_num_threads(2)
            from stable_baselines3 import PPO
            self.model=PPO.load(model_path,device='cpu')
            self.action_scale=policy_action_scale(model_path)
            from .model import JOINTS_PER_TONGUE
            if self.model.action_space.shape != (JOINTS_PER_TONGUE,):
                raise ValueError('Checkpoint uses the previous mechanism. Train a new four-DOF policy in models/four-dof-v1.')

    def act(self,observation,fallback):
        if self.model is None:
            return fallback
        action,_=self.model.predict(observation,deterministic=True)
        return np.asarray(action,dtype=float).clip(-1,1)*self.action_scale
