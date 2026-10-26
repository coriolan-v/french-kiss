"""Explicitly scripted, changing exploration and alternating retreat/chase."""
import mujoco
import numpy as np
from .config import CONFIG
from .model import JOINTS_PER_TONGUE
from .reference_motion import reference_goal,reference_phase

PHASES=(('Meet',2.),('Explore together',2.8),('A pulls away / B follows',3.6),
        ('Explore apart',3.),('B pulls away / A follows',3.6),('Reunite',3.))


def interaction_phase(t,style='explore'):
    if style=='reference': return reference_phase(t)
    duration=sum(seconds for _,seconds in PHASES)
    cycle=int(t//duration)
    local=t%duration
    for index,(name,seconds) in enumerate(PHASES):
        if local<seconds:
            # Swap who leads each complete cycle.
            if cycle%2 and index in (2,4): name=PHASES[6-index][0]
            return name,index,local/seconds,cycle
        local-=seconds
    return 'Meet',0,0.,cycle


def interaction_goal(sim,side):
    if sim.interaction_style=='reference': return reference_goal(sim,side)
    t=sim.data.time
    p=sim.personalities[side]
    n=0 if side=='a' else 1
    sign=1 if side=='a' else -1
    name,stage,u,cycle=interaction_phase(t)
    envelope=np.sin(np.pi*u)**2
    direction=1 if cycle%2==0 else -1
    # Substantial shared motion, plus a different path every cycle.
    common_y=.13*np.sin(.9*t)+.045*np.sin(.43*t+cycle*1.7)
    common_z=1.25+.10*np.sin(.8*t)
    nominal=np.array([sign*(.13-.06*(1-p.desire)),common_y+sign*.025*(1-p.desire),common_z])
    goal=nominal.copy()
    if stage in (1,3):
        spread=.18 if stage==1 else .32
        goal[0]-=sign*envelope*.13*(.4+p.curiosity)
        goal[1]+=sign*direction*envelope*spread*(.45+.55*p.curiosity)
        goal[2]+=envelope*.14*np.sin(t*.9+n*2.1)*( .5+.5*p.playfulness)
    elif stage in (2,4):
        leader='a' if ((stage==2) != bool(cycle%2)) else 'b'
        if side==leader:
            retreat=envelope*(.7+.3*(1-p.desire))
            goal[0]-=sign*.34*retreat
            goal[1]+=direction*.34*retreat*(.6+.4*p.curiosity)
            goal[2]+=.18*retreat
        else:
            # Follow the partner's actual moving tip, rather than a preset pose.
            partner=sim.tips()[leader]
            following=partner+np.array([-sign*.13,0,.025*direction])
            interest=.5+.3*p.desire+.2*p.reciprocity
            goal=(1-envelope*interest)*nominal+envelope*interest*following
    # Independent exploration remains visible at every slider setting.
    goal[1]+=.025*p.curiosity*(1-p.reciprocity)*np.sin(.7*t+n)
    goal[2]+=.025*p.playfulness*np.sin(1.1*t+n)
    return goal,name


def interaction_controls(sim,side):
    p=sim.personalities[side]
    t=sim.data.time
    n=0 if side=='a' else 1
    goal,_=interaction_goal(sim,side)
    tips=sim.tips()
    jac=np.zeros((3,sim.model.nv))
    mujoco.mj_jacSite(sim.model,sim.data,jac,None,sim.model.site(f'{side}_tip').id)
    columns=slice(n*JOINTS_PER_TONGUE,(n+1)*JOINTS_PER_TONGUE)
    j=jac[:,columns]
    q=sim.data.qpos[columns]
    inverse=j.T@np.linalg.solve(j@j.T+.003*np.eye(3),np.eye(3))
    delta=inverse@((goal-tips[side])*(.65+.35*p.confidence))
    posture=np.array([0.,-.15,.30+.24*(.4+.6*p.playfulness)*np.sin(1.1*t+n*.9),
                      -.18+.30*(.4+.6*p.playfulness)*np.sin(.95*t+n*.9)])
    if sim.interaction_style=='reference':
        posture=np.array([0.,-.07,.14+.12*(.4+.6*p.playfulness)*np.sin(1.35*t+n*.6),
                          -.06+.20*(.4+.6*p.playfulness)*np.sin(1.2*t+n*.6)])
    delta+=(np.eye(4)-inverse@j)@(.65*(posture-q))
    compensation=(sim.data.qfrc_bias[columns]+CONFIG['joint_stiffness_Nm_rad']*q)/180.
    force=sim.contact_metrics()['contact_force']
    # Give way during loaded contact instead of fighting the partner's path.
    pressure=max(0.,force-10.)
    delta/=1+(pressure/12.)**2
    yielding=np.clip(sim.data.qfrc_constraint[columns]/180.,-.12,.12)*min(1.,pressure/30.)
    target=q+np.clip(delta,-.25,.25)+compensation+yielding
    return np.clip(target,-CONFIG['target_limit_rad'],CONFIG['target_limit_rad'])
