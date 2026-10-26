"""Isolated full-range joint inspection and optional tendon-physics exercise."""
import mujoco
import numpy as np
from .model import JOINT_NAMES


class JointTest:
    def __init__(self,sim):
        self.sim=sim
        self.enabled=False
        self.physical=False
        self.selected=0
        self.targets=np.zeros(sim.model.nq)
        self.limits=sim.model.jnt_range.copy()
        self.sweep=False
        self.sweep_all=False
        self.stage=0
        self.completed=[]

    def enter(self):
        self.enabled=True
        self.sim.reset()
        self.targets=self.sim.data.qpos.copy()
        self.sweep=False
        self.sweep_all=False
        self.completed=[]

    def leave(self):
        self.enabled=False
        self.sweep=False
        self.sim.reset()

    def select(self,index):
        if not 0<=index<8: raise ValueError('Joint index must be 0 to 7')
        self.selected=index
        self.sweep=False
        self.sweep_all=False

    def set_target(self,value):
        self.sweep=False
        self.targets[self.selected]=np.clip(value,*self.limits[self.selected])

    def center(self):
        self.sweep=False
        self.targets[:]=0

    def start_sweep(self,all_joints=False):
        self.targets[:]=0
        self.sweep=True
        self.sweep_all=all_joints
        self.stage=0
        self.completed=[]
        if all_joints: self.selected=0

    def step(self,dt):
        if self.sweep:
            index=self.selected
            destination=[self.limits[index,0],self.limits[index,1],0.][self.stage]
            delta=float(np.clip(destination-self.targets[index],-.25*dt,.25*dt))
            self.targets[index]+=delta
            if abs(self.targets[index]-destination)<1e-8:
                self.stage+=1
                if self.stage==3:
                    self.completed.append(index)
                    self.stage=0
                    if self.sweep_all:
                        if index==7: self.sweep=False
                        else: self.selected+=1
        if self.physical:
            self.sim.step({'a':self.targets[:4],'b':self.targets[4:]},joint_test=True)
        else:
            # Exact range inspection: no dynamics, contact response or agent control.
            before=self.sim.data.qpos.copy()
            self.sim.data.qpos[:]=self.targets
            self.sim.data.qvel[:]=(self.targets-before)/dt
            self.sim.data.ctrl[:]=0
            self.sim.data.time+=dt
            mujoco.mj_forward(self.sim.model,self.sim.data)
            self.sim.last_controls={s:self.targets[n*4:(n+1)*4].copy() for n,s in enumerate(('a','b'))}

    @property
    def name(self):
        return ('A' if self.selected<4 else 'B')+' / '+JOINT_NAMES[self.selected%4].replace('_',' ').title()
