"""Hand-tuned choreography informed by inspected video sequences, not learned motion."""
import json
from pathlib import Path
import numpy as np

PROFILE=json.loads((Path(__file__).resolve().parents[1]/'reference-motion-profile.json').read_text())
NAMES=('Approach','Close exploration','A initiates / B responds','Brief separation','B initiates / A responds','Reconnect')
if PROFILE['schema']!=1 or len(PROFILE['phase_durations_s'])!=len(NAMES) or min(PROFILE['phase_durations_s'])<=0:
    raise ValueError('Invalid reference motion profile')


def reference_phase(t):
    durations=PROFILE['phase_durations_s']
    cycle=int(t//sum(durations)); local=t%sum(durations)
    for stage,(name,duration) in enumerate(zip(NAMES,durations)):
        if local<duration:
            if cycle%2 and stage in (2,4): name=NAMES[6-stage]
            return name,stage,local/duration,cycle
        local-=duration
    return NAMES[0],0,0.,cycle


def reference_goal(sim,side):
    t=sim.data.time; p=sim.personalities[side]
    sign=1 if side=='a' else -1
    n=0 if side=='a' else 1
    name,stage,u,cycle=reference_phase(t)
    envelope=np.sin(np.pi*u)**2
    y=PROFILE['shared_lateral_amplitude_m']*np.sin(1.0*t+cycle*.5)
    z=1.25+PROFILE['shared_vertical_amplitude_m']*np.sin(.85*t)
    # Small alternating height offsets encourage over/under contact, not rigid symmetry.
    goal=np.array([sign*(.18-.025*(1-p.desire)),
        y+sign*.014*(1-p.desire)+.015*p.curiosity*(1-p.reciprocity)*np.sin(1.25*t+n),
        z+sign*.025*np.sin(.9*t+cycle*.5)+.015*p.playfulness*np.sin(1.4*t+n)])
    if stage==1:
        goal[1]+=sign*.03*p.curiosity*envelope*np.sin(1.3*t)
    elif stage in (2,4):
        leader='a' if ((stage==2)!=bool(cycle%2)) else 'b'
        if side==leader:
            goal[0]-=sign*.07*envelope*(.6+.4*p.confidence)
            goal[1]+=.07*envelope*np.sin(t+cycle)
            goal[2]+=.06*envelope
        else:
            actual=sim.tips()[leader]+np.array([-sign*.12,0,.02*sign])
            response=envelope*(.45+.25*p.desire+.3*p.reciprocity)
            goal=(1-response)*goal+response*actual
    elif stage==3:
        retreat=envelope*(.6+.4*(1-p.desire))
        goal[0]-=sign*PROFILE['pause_withdrawal_m']*retreat
        goal[1]+=sign*.10*retreat
        goal[2]+=.09*retreat
    return goal,name
