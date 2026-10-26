"""Replay both policies or report reproducible digital interaction metrics."""
import argparse
import json
import numpy as np
from stable_baselines3 import PPO
from frenchkiss.simulation import FrenchKissSim
from frenchkiss.config import CONFIG
from frenchkiss.agents import policy_action_scale, scripted_controls


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--style',choices=['reference','explore'],default=CONFIG.get('interaction_style','reference'))
    parser.add_argument('--side',choices=['a','b','both'],default='both')
    parser.add_argument('--model')
    parser.add_argument('--model-dir',default='models/four-dof-v1')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--baseline',action='store_true')
    args=parser.parse_args()
    sides=('a','b') if args.side=='both' else (args.side,)
    paths={s:args.model or f'{args.model_dir}/tongue_{s}_ppo.zip' for s in sides}
    if not args.headless:
        import sys
        from scripts.demo import main as demo
        sys.argv=[sys.argv[0],'--style',args.style]+([] if args.baseline else [v for s,path in paths.items() for v in (f'--policy-{s}',path)])
        demo(); return
    policies={} if args.baseline else {s:PPO.load(path,device='cpu') for s,path in paths.items()}
    scales={s:policy_action_scale(paths[s]) for s in policies}
    sim=FrenchKissSim()
    sim.interaction_style=args.style
    forces=[]; gaps=[]; velocities=[]; positions=[]
    controls={}
    for tick in range(args.steps):
        if tick%6==0:
            controls={s:np.asarray(policy.predict(sim.observe(s),deterministic=True)[0])*scales[s] for s,policy in policies.items()}
            if args.baseline:
                tips=sim.tips()
                controls={s:scripted_controls(sim.data.time,s,sim.personalities[s],tips[s],tips['b' if s=='a' else 'a'],sim=sim) for s in ('a','b')}
        sim.step(controls)
        m=sim.contact_metrics()
        forces.append(m['contact_force']); gaps.append(m['tip_distance'])
        positions.append(np.stack(list(sim.tips().values())))
        velocities.append(float(np.mean(abs(sim.data.qvel))))
    force=np.array(forces)
    positions=np.array(positions)
    window=int(round(3/sim.model.opt.timestep))
    excursions=[np.linalg.norm(np.ptp(positions[start:start+window],axis=0),axis=1) for start in range(0,len(positions)-window+1,window)]
    minimum_excursion=np.min(excursions,axis=0).tolist() if excursions else None
    print(json.dumps({'steps':args.steps,'simulated_seconds':float(sim.data.time),'style':sim.interaction_style,'controller':'scripted' if args.baseline else 'PPO','contact_fraction':float(np.mean(force>.1)),'gentle_contact_fraction':float(np.mean((force>.1)&(force<35))),'peak_contact_force_N':float(force.max()),'mean_tip_gap_m':float(np.mean(gaps)),'mean_joint_speed_rad_s':float(np.mean(velocities)),'minimum_tip_excursion_in_3s_windows_m':minimum_excursion,'finite':bool(np.isfinite(sim.data.qpos).all())},indent=2))

if __name__=='__main__': main()
