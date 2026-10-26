"""Personality-conditioned PPO with frozen partner pools and contact rewards."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .simulation import FrenchKissSim
from .config import CONFIG
from .model import JOINTS_PER_TONGUE
from .agents import policy_action_scale


class TongueEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, side='a', episode_steps=400, partner_paths=()):
        if side not in ('a','b'):
            raise ValueError('side must be a or b')
        self.sim=FrenchKissSim()
        self.side=side
        self.other='b' if side=='a' else 'a'
        self.episode_steps=episode_steps
        self.partner_paths=list(partner_paths)
        self._partners={}
        self.partner=None
        self.partner_scale=CONFIG['target_limit_rad']
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=self.sim.observe(side).shape,dtype=np.float32)
        self.action_space=spaces.Box(-1,1,shape=(JOINTS_PER_TONGUE,),dtype=np.float32)
        self.ticks=0
        self.previous_action=np.zeros(JOINTS_PER_TONGUE)
        self.visits={}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.ticks=0
        self.visits={}
        self.previous_action=np.zeros(JOINTS_PER_TONGUE)
        for p in self.sim.personalities.values():
            for key in p.__dataclass_fields__:
                setattr(p,key,float(self.np_random.uniform(0,1)))
        self.sim.data.qpos[:]+=self.np_random.uniform(-.025,.025,self.sim.model.nq)
        import mujoco
        mujoco.mj_forward(self.sim.model,self.sim.data)
        self.partner=None
        if self.partner_paths and self.np_random.random() < .8:
            path=str(self.np_random.choice(self.partner_paths))
            if path not in self._partners:
                from stable_baselines3 import PPO
                self._partners[path]=PPO.load(path,device='cpu')
            self.partner=self._partners[path]
            self.partner_scale=policy_action_scale(path)
        return self.sim.observe(self.side),{}

    def step(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape != (JOINTS_PER_TONGUE,) or not np.isfinite(action).all():
            raise ValueError('Expected four finite actions')
        action=action.clip(-1,1)
        controls={self.side:action*CONFIG['target_limit_rad']}
        if self.partner is not None:
            other,_=self.partner.predict(self.sim.observe(self.other),deterministic=True)
            controls[self.other]=other*self.partner_scale
        before=self.sim.proximity()
        tips_before=self.sim.tips()
        forces=[]
        for _ in range(CONFIG['control_period_steps']):
            self.sim.step(controls)
            forces.append(self.sim.contact_metrics()['contact_force'])
        metrics=self.sim.contact_metrics()
        force=float(np.mean(forces))
        p=self.sim.personalities[self.side]
        distance=metrics['tip_distance']
        cell=tuple(np.round(self.sim.tips()[self.side]/.08).astype(int))
        visits=self.visits.get(cell,0)+1
        self.visits[cell]=visits
        own=self.sim.data.qvel[:JOINTS_PER_TONGUE] if self.side=='a' else self.sim.data.qvel[JOINTS_PER_TONGUE:]
        other=self.sim.data.qvel[JOINTS_PER_TONGUE:] if self.side=='a' else self.sim.data.qvel[:JOINTS_PER_TONGUE]
        tip_velocity={s:(self.sim.tips()[s]-tips_before[s])/(CONFIG['control_period_steps']*self.sim.model.opt.timestep) for s in ('a','b')}
        relative_speed=np.linalg.norm(tip_velocity[self.side]-tip_velocity[self.other])
        motion=float(np.linalg.norm(tip_velocity[self.side]))
        reciprocal=np.exp(-relative_speed*3)
        gentle=float(0.1 < force < CONFIG['gentle_force_threshold_N'])
        reward=(p.desire*np.exp(-4*distance)+4*p.confidence*np.clip(before-distance,-.1,.1)
                +gentle*(.3+p.reciprocity*reciprocal)
                +.12*p.curiosity/np.sqrt(visits)
                +.04*p.playfulness*np.tanh(np.mean(abs(own)))
                -.015*np.mean(action**2)-.03*np.mean((action-self.previous_action)**2)
                -.03*max(0,force-CONFIG['gentle_force_threshold_N'])-(1-p.desire)*gentle*.5
                -.15*gentle*float(motion<.015)
                +2*(1-p.desire)*np.clip(distance-before,-.1,.1))
        self.previous_action=action.copy()
        self.ticks+=1
        finite=np.isfinite(self.sim.data.qpos).all() and np.isfinite(self.sim.data.qvel).all()
        metrics.update(mean_contact_force=force,gentle_contact=gentle,tip_speed=motion,relative_tip_speed=float(relative_speed),novelty=1/np.sqrt(visits))
        return self.sim.observe(self.side),float(reward),not finite,self.ticks>=self.episode_steps,metrics
