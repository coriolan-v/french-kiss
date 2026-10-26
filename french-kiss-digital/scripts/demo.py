"""Single-window digital exhibition. No hardware imports or connections."""
import argparse
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import pygame
from frenchkiss.simulation import FrenchKissSim
from frenchkiss.agents import PolicyAgent,scripted_controls
from frenchkiss.tracking import StereoTracker
from frenchkiss.config import CONFIG
from frenchkiss.joint_test import JointTest
from frenchkiss.interaction import interaction_phase
from scripts.joint_panel import JointPanel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--style',choices=['reference','explore'],default=CONFIG.get('interaction_style','reference'))
    parser.add_argument('--joint-test',action='store_true',help='Open full-range joint inspection')
    parser.add_argument('--sweep-all',action='store_true',help='Sweep all eight joints in test mode')
    parser.add_argument('--trained',action='store_true',help='Load the four-dof-v1 trained pair')
    parser.add_argument('--record',help='Record simulation states to JSONL')
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--policy-a')
    parser.add_argument('--policy-b')
    parser.add_argument('--seconds',type=float,default=0)
    parser.add_argument('--screenshot')
    parser.add_argument('--replay',help='JSONL recording to replay')
    args=parser.parse_args()
    if args.sweep_all and not args.joint_test:
        parser.error('--sweep-all requires --joint-test')
    if args.trained:
        args.policy_a=args.policy_a or 'models/four-dof-v1/tongue_a_ppo.zip'
        args.policy_b=args.policy_b or 'models/four-dof-v1/tongue_b_ppo.zip'
    sim=FrenchKissSim()
    sim.interaction_style=args.style
    if args.replay and args.joint_test:
        parser.error('Replay and joint test cannot be combined')
    joint_test=JointTest(sim)
    joint_panel=JointPanel(joint_test)
    if args.joint_test:
        joint_test.enter()
        if args.sweep_all: joint_test.start_sweep(True)
    agents={s:PolicyAgent(getattr(args,f'policy_{s}')) for s in ('a','b')}
    pygame.init()
    screen=pygame.display.set_mode((1280,800))
    pygame.display.set_caption('FRENCH KISS | Digital exhibition')
    font=pygame.font.SysFont('Segoe UI',18)
    title=pygame.font.SysFont('Segoe UI',28,bold=True)
    small=pygame.font.SysFont('Segoe UI',15)
    renderer=mujoco.Renderer(sim.model,height=680,width=960)
    camera=mujoco.MjvCamera()
    camera.lookat[:]=(0,0,1.05)
    camera.distance=3.6
    camera.azimuth=90
    camera.elevation=-30
    option=mujoco.MjvOption()
    option.geomgroup[4]=0
    option.geomgroup[2]=0
    option.sitegroup[3]=0
    clock=pygame.time.Clock()
    tracker=StereoTracker()
    use_learned=any(agent.model is not None for agent in agents.values())
    paused=False; debug=args.debug; recording=None; running=True
    option.geomgroup[2]=int(debug); option.sitegroup[3]=int(debug)
    if args.record:
        Path(args.record).parent.mkdir(parents=True,exist_ok=True)
        recording=open(args.record,'w')
    replay=[] if not args.replay else [json.loads(line) for line in Path(args.replay).read_text().splitlines()]
    if args.replay and not replay:
        parser.error('Replay file contains no frames')
    if replay and any(len(f.get('qpos',[]))!=sim.model.nq or len(f.get('qvel',[]))!=sim.model.nv for f in replay):
        parser.error('Recording belongs to a different mechanism. Record a new four-DOF demo.')
    replay_index=0
    frame=replay[0] if replay else {}
    sliders=[]
    colors={'a':(236,108,143),'b':(170,139,224)}
    for n,s in enumerate(('a','b')):
        for i,field in enumerate(sim.personalities[s].__dataclass_fields__):
            sliders.append((s,field,pygame.Rect(995,180+n*270+i*39,245,9)))
    start=time.perf_counter()
    accumulator=0.
    next_control_time=0.
    controls={}
    def text(value,x,y,color=(213,219,232),face=None):
        screen.blit((face or font).render(str(value),True,color),(x,y))
    try:
        while running:
            dt=min(clock.tick(30)/1000,.1)
            for event in pygame.event.get():
                if event.type==pygame.QUIT: running=False
                if joint_test.enabled: joint_panel.event(event)
                if event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_j and not replay:
                        if joint_test.enabled: joint_test.leave()
                        else: joint_test.enter()
                        next_control_time=0.
                        accumulator=0.
                    if event.key==pygame.K_ESCAPE: running=False
                    if event.key==pygame.K_SPACE: paused=not paused
                    if event.key==pygame.K_r:
                        if joint_test.enabled: joint_test.enter()
                        else: sim.reset()
                        next_control_time=0.
                    if event.key==pygame.K_t and not joint_test.enabled:
                        if not any(agent.model is not None for agent in agents.values()):
                            paths={s:Path(f'models/four-dof-v1/tongue_{s}_ppo.zip') for s in ('a','b')}
                            if all(path.exists() for path in paths.values()):
                                agents={s:PolicyAgent(str(path)) for s,path in paths.items()}
                        use_learned=not use_learned if any(agent.model is not None for agent in agents.values()) else False
                    if event.key==pygame.K_v and not replay and not joint_test.enabled:
                        sim.interaction_style='explore' if sim.interaction_style=='reference' else 'reference'
                        next_control_time=0.
                    if event.key==pygame.K_d:
                        debug=not debug
                        option.geomgroup[2]=int(debug); option.sitegroup[3]=int(debug)
                    if event.key==pygame.K_1: camera.azimuth=90; camera.elevation=-30
                    if event.key==pygame.K_2: camera.azimuth=45; camera.elevation=-35
                    if event.key==pygame.K_3: camera.elevation=-89
                    if event.key==pygame.K_s:
                        if recording:
                            recording.close(); recording=None
                        else:
                            Path('recordings').mkdir(exist_ok=True)
                            recording=open(Path('recordings')/f'demo-{time.time_ns()}.jsonl','w')
                if event.type==pygame.MOUSEWHEEL:
                    camera.distance=float(np.clip(camera.distance-event.y*.15,1.5,7))
                if event.type==pygame.MOUSEMOTION and event.buttons[2] and event.pos[0]<960:
                    camera.azimuth-=event.rel[0]*.3
                    camera.elevation=float(np.clip(camera.elevation-event.rel[1]*.3,-89,0))
            if pygame.mouse.get_pressed()[0] and not replay and not joint_test.enabled:
                mx,my=pygame.mouse.get_pos()
                for side,field,rect in sliders:
                    if rect.inflate(15,25).collidepoint(mx,my):
                        setattr(sim.personalities[side],field,float(np.clip((mx-rect.x)/rect.w,0,1)))
            if not paused:
                accumulator+=dt*(1. if joint_test.enabled else CONFIG['presentation_speed'])
                while accumulator>=sim.model.opt.timestep:
                    if joint_test.enabled:
                        joint_test.step(sim.model.opt.timestep)
                    elif replay:
                        frame=replay[replay_index]
                        sim.data.qpos[:]=frame['qpos']; sim.data.qvel[:]=frame['qvel']; sim.data.time=frame['time']
                        for side in ('a','b'):
                            for field,value in frame['personalities'][side].items(): setattr(sim.personalities[side],field,value)
                        mujoco.mj_forward(sim.model,sim.data)
                        replay_index=(replay_index+1)%len(replay)
                    else:
                        tips=sim.tips()
                        if sim.data.time>=next_control_time or sim.data.time==0:
                            next_control_time=sim.data.time+.03
                            controls={}
                            for s in ('a','b'):
                                fallback=scripted_controls(sim.data.time,s,sim.personalities[s],tips[s],tips['b' if s=='a' else 'a'],sim=sim)
                                controls[s]=agents[s].act(sim.observe(s),fallback) if use_learned else fallback
                        sim.step(controls)
                    if recording:
                        recording.write(json.dumps({'schema':1,'mechanism':CONFIG['mechanism'],'style':sim.interaction_style,'mode':'joint_pose' if joint_test.enabled and not joint_test.physical else 'physics','time':sim.data.time,'qpos':sim.data.qpos.tolist(),'qvel':sim.data.qvel.tolist(),'metrics':sim.contact_metrics(),'personalities':{s:vars(p) for s,p in sim.personalities.items()}})+'\n')
                    accumulator-=sim.model.opt.timestep
            screen.fill((15,19,29))
            renderer.update_scene(sim.data,camera=camera,scene_option=option)
            rgb=renderer.render()
            screen.blit(pygame.surfarray.make_surface(rgb.swapaxes(0,1)),(0,80))
            text('FRENCH KISS',26,16,face=title)
            text('Full-range joint inspection' if joint_test.enabled else f"4 joints per tongue / slow motion {CONFIG['presentation_speed']:.1f}x",280,26,face=small)
            if joint_test.enabled:
                joint_panel.draw(screen,text,small,title)
            else:
                text('AUDIENCE',990,26,face=title)
                for n,s in enumerate(('a','b')):
                    y=104+n*270
                    text('TONGUE '+s.upper(),992,y,colors[s],title)
                    text('Replay' if replay else (f'Experimental PPO / {agents[s].model.num_timesteps:,}' if agents[s].model and use_learned else ('Reference / scripted' if sim.interaction_style=='reference' else 'Explore / scripted')),993,y+36,face=small)
                for side,field,rect in sliders:
                    value=getattr(sim.personalities[side],field)
                    text(field.title(),rect.x,rect.y-24,face=small)
                    text(f'{value:.2f}',rect.right-35,rect.y-24,face=small)
                    pygame.draw.rect(screen,(45,51,67),rect,border_radius=4)
                    pygame.draw.rect(screen,colors[side],(rect.x,rect.y,int(rect.w*value),rect.h),border_radius=4)
                    pygame.draw.circle(screen,(240,235,240),(int(rect.x+rect.w*value),rect.centery),6)
            metrics=frame.get('metrics',sim.contact_metrics()) if replay else sim.contact_metrics()
            if not joint_test.enabled and not replay and not use_learned and not debug:
                text(interaction_phase(sim.data.time,sim.interaction_style)[0]+' / '+('contact' if metrics['contacts'] else 'moving'),24,92,face=small)
            if (joint_test.enabled and not joint_test.physical) or (replay and frame.get('mode')=='joint_pose'):
                text('POSE PREVIEW / agents and contact dynamics paused / '+joint_test.name,24,766,face=small)
            else:
                text(f"{sim.data.time:6.1f}s   |   Tip separation {metrics['tip_distance']:.2f} m   |   Contact {metrics['contact_force']:.1f} N",24,766,face=small)
            text('PAUSED' if paused else ('RECORDING' if recording else f'{clock.get_fps():.0f} FPS'),800,766,colors['a'],small)
            if not joint_test.enabled:
                for i,line in enumerate(['Space  pause     R  reset','1 / 2 / 3  camera views','Right drag  orbit / wheel zoom','D  tracking + ribs','S  start / stop recording','T  PPO / V  reference-explore','J  joint test / Esc  close']): text(line,990,645+i*20,face=small)
            if debug:
                points=np.array([sim.data.site(f'{s}_marker{i}').xpos for s in ('a','b') for i in range(8)])
                tracked=tracker.observe(points)
                valid=tracked['visible']
                error=np.mean(np.linalg.norm(tracked['positions'][valid]-points[valid],axis=1)) if valid.any() else float('nan')
                text(f'Synthetic stereo: {valid.sum()}/16 markers | error {error*1000:.1f} mm',24,92,face=small)
            pygame.display.flip()
            if args.seconds and time.perf_counter()-start>=args.seconds:
                if args.screenshot: pygame.image.save(screen,args.screenshot)
                running=False
    finally:
        if recording: recording.close()
        renderer.close()
        pygame.quit()

if __name__=='__main__': main()
