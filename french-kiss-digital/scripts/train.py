"""Alternating independent PPO agents against a pool of frozen partners."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from frenchkiss.config import CONFIG
from frenchkiss.rl_env import TongueEnv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='models/four-dof-v1')
    p.add_argument('--side',choices=['a','b','both'],default='both')
    p.add_argument('--steps',type=int,default=100000,help='Steps per agent per round')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    if min(args.steps,args.rounds,args.threads)<=0:
        p.error('steps, rounds and threads must be positive')
    torch.set_num_threads(args.threads)
    folder=Path(args.output); folder.mkdir(parents=True,exist_ok=True)
    for round_id in range(args.rounds):
        for side in (('a','b') if args.side=='both' else (args.side,)):
            other='b' if side=='a' else 'a'
            pool=sorted(folder.glob(f'{other}_round_*.zip'))[-4:]
            env=TongueEnv(side=side,partner_paths=[str(x) for x in pool])
            check_env(env)
            env=Monitor(env,filename=str(folder/f'{side}_round_{round_id:03d}_episodes'))
            latest=folder/f'tongue_{side}_ppo.zip'
            if round_id and latest.exists():
                model=PPO.load(latest,env=env,device='cpu')
            else:
                model=PPO('MlpPolicy',env,verbose=1,device='cpu',seed=args.seed+(side=='b'),n_steps=512,batch_size=128,policy_kwargs={'net_arch':[128,128]})
            model.learn(total_timesteps=args.steps,reset_num_timesteps=False)
            model.save(latest)
            model.save(folder/f'{side}_round_{round_id:03d}')
            (folder/f'{side}_training.json').write_text(json.dumps({'config':CONFIG,'schema':2,'side':side,'seed':args.seed+(side=='b'),'timesteps':model.num_timesteps,'round':round_id,'partners':[str(x) for x in pool],'status':'experimental; assess rendered behaviour'},indent=2))
            env.close()

if __name__=='__main__':
    main()
