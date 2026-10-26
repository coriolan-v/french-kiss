"""Physics, bounded tendon commands and inter-tongue contact telemetry."""
import numpy as np
import mujoco
from .model import make_model, SEGMENTS, JOINTS_PER_TONGUE, tip_pos
from .agents import Personality, scripted_controls
from .config import CONFIG


class FrenchKissSim:
    def __init__(self):
        self.interaction_style=CONFIG.get('interaction_style','explore')
        self.model = make_model()
        self.data = mujoco.MjData(self.model)
        self.personalities = {'a': Personality(), 'b': Personality(desire=.8, curiosity=.7, confidence=.8)}
        self._geom_sides = {self.model.geom(f'{s}_segment{i}').id:s for s in ('a','b') for i in range(SEGMENTS)}
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:]=[.04,-.06,.02,-.02,-.04,.06,-.02,.02]
        self.last_controls = {s:self.data.qpos[n*JOINTS_PER_TONGUE:(n+1)*JOINTS_PER_TONGUE].copy() for n,s in enumerate(('a','b'))}
        mujoco.mj_forward(self.model, self.data)

    def tips(self):
        return {s:tip_pos(self.model,self.data,s) for s in ('a','b')}

    def observe(self, side):
        if side not in ('a','b'):
            raise ValueError('side must be a or b')
        tips = self.tips()
        return np.concatenate([self.data.qpos, self.data.qvel, tips[side], tips['b' if side=='a' else 'a'], self.personalities[side].vector()]).astype(np.float32)

    def step(self, controls=None, *, joint_test=False):
        tips = self.tips()
        # Validate all inputs before changing state.
        targets = {}
        for side in ('a','b'):
            value = None if controls is None else controls.get(side)
            if value is None:
                value = scripted_controls(self.data.time,side,self.personalities[side],tips[side],tips['b' if side=='a' else 'a'],sim=self)
            value = np.asarray(value,dtype=float)
            if value.shape != (JOINTS_PER_TONGUE,) or not np.isfinite(value).all():
                raise ValueError(f'{side}: expected {JOINTS_PER_TONGUE} finite controls')
            if joint_test:
                offset=0 if side=='a' else JOINTS_PER_TONGUE
                limits=self.model.jnt_range[offset:offset+JOINTS_PER_TONGUE]
                targets[side]=np.clip(value,limits[:,0],limits[:,1])
            else:
                targets[side] = np.clip(value,-CONFIG['target_limit_rad'],CONFIG['target_limit_rad'])
        for n,side in enumerate(('a','b')):
            # Rate-limited joint targets become antagonistic tensile cable forces.
            max_change=CONFIG['target_speed_rad_s']*self.model.opt.timestep
            target = self.last_controls[side] + np.clip(targets[side]-self.last_controls[side],-max_change,max_change)
            self.last_controls[side] = target
            q = self.data.qpos[n*JOINTS_PER_TONGUE:(n+1)*JOINTS_PER_TONGUE]
            v = self.data.qvel[n*JOINTS_PER_TONGUE:(n+1)*JOINTS_PER_TONGUE]
            torque = np.clip(180*(target-q)-10*v,-18,18)
            tension = torque/(2*CONFIG['cable_moment_arm_m'])
            preload=CONFIG['max_tension_N']/2
            pair = np.stack([-preload-tension, -preload+tension],axis=1)
            self.data.ctrl[n*JOINTS_PER_TONGUE*2:(n+1)*JOINTS_PER_TONGUE*2] = pair.ravel().clip(-CONFIG['max_tension_N'],0)
        mujoco.mj_step(self.model,self.data)
        return self.tips()

    def contact_metrics(self):
        force = 0.
        count = 0
        wrench = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            a,b = self._geom_sides.get(contact.geom1),self._geom_sides.get(contact.geom2)
            if a is not None and b is not None and a != b:
                mujoco.mj_contactForce(self.model,self.data,index,wrench)
                force += max(0.,float(wrench[0]))
                count += 1
        return {'contacts':count,'contact_force':force,'tip_distance':self.proximity()}

    def proximity(self):
        tips=self.tips()
        return float(np.linalg.norm(tips['a']-tips['b']))

