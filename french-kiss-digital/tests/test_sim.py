import numpy as np
import pytest
import mujoco
from frenchkiss.simulation import FrenchKissSim
from frenchkiss.config import CONFIG
from frenchkiss.rl_env import TongueEnv
from frenchkiss.tracking import StereoTracker
from frenchkiss.agents import Personality,scripted_controls,PolicyAgent


def test_scene_and_tendon_stability():
    sim=FrenchKissSim()
    assert sim.model.nq==8 and sim.model.nu==16 and sim.model.ntendon==16
    assert sim.observe('a').shape==(27,)
    assert sim.contact_metrics()['contacts']==0
    for _ in range(3000): sim.step()
    assert np.isfinite(sim.data.qpos).all()
    assert max(abs(sim.data.qpos))<CONFIG['joint_limit_rad']+.05
    assert np.all(sim.data.ctrl<=0) and np.all(sim.data.ctrl>=-600)
    assert sim.model.site('a_tip').pos[0]==pytest.approx(.15)
    assert [sim.model.joint(i).name for i in range(4)] == ['a_base_rotate','a_base_hinge','a_middle_hinge','a_tip_hinge']


def test_tendon_direction_and_validation():
    sim=FrenchKissSim()
    zero=np.zeros(4)
    action=zero.copy(); action[0]=.2
    for _ in range(100): sim.step({'a':action,'b':zero})
    assert sim.data.qpos[0]>.05
    state=sim.data.qpos.copy()
    for invalid in (np.full(4,np.nan),np.zeros(16)):
        with pytest.raises(ValueError): sim.step({'b':invalid})
        np.testing.assert_array_equal(state,sim.data.qpos)


def test_env_reproducible_truncation():
    env=TongueEnv(episode_steps=2)
    a,_=env.reset(seed=123)
    b,_=env.reset(seed=123)
    np.testing.assert_array_equal(a,b)
    result=env.step(np.zeros(4))
    assert np.isfinite(result[1]) and 'mean_contact_force' in result[4]
    assert not result[3]
    assert env.step(np.zeros(4))[3]


def test_intertongue_force_excludes_floor():
    sim=FrenchKissSim()
    sim.data.qpos[:]=0
    mujoco.mj_forward(sim.model,sim.data)
    assert sim.contact_metrics()['contact_force']>0
    sim.data.qpos[0]=.4
    sim.data.qpos[4]=.4
    mujoco.mj_forward(sim.model,sim.data)
    assert sim.contact_metrics()['contacts']==0


def test_stereo_calibration_and_dropout():
    points=np.array([[0,0,1.2],[.1,.1,1.1],[-.4,0,1.]])
    tracker=StereoTracker()
    tracked=tracker.observe(points,noise_px=0,dropout=0)
    assert tracked['visible'].all()
    np.testing.assert_allclose(tracked['positions'],points,atol=1e-8)
    assert not tracker.observe(points,dropout=1)['visible'].any()


@pytest.mark.parametrize('field',list(Personality.__dataclass_fields__))
def test_each_audience_control_affects_baseline(field):
    p=Personality()
    tip=np.array([0,.08,1.]); other=np.array([0,-.1,1.2])
    setattr(p,field,0)
    low=scripted_controls(1,'a',p,tip,other)
    setattr(p,field,1)
    high=scripted_controls(1,'a',p,tip,other)
    assert not np.allclose(low,high)


def test_policy_scale():
    class Fake:
        def predict(self,*args,**kwargs): return np.ones(4),None
    agent=PolicyAgent(); agent.model=Fake()
    np.testing.assert_allclose(agent.act(np.zeros(27),None),CONFIG['target_limit_rad'])


@pytest.mark.parametrize('name,link,axis', [
    ('base_rotate',0,[0,0,1]), ('base_hinge',0,[0,1,0]),
    ('middle_hinge',4,[0,1,0]), ('tip_hinge',6,[0,1,0]),
])
def test_four_joint_axes_and_pivots(name,link,axis):
    sim=FrenchKissSim()
    for side in ('a','b'):
        joint=sim.model.joint(f'{side}_{name}')
        assert joint.bodyid[0]==sim.model.body(f'{side}_link{link}').id
        np.testing.assert_array_equal(joint.axis,axis)


def test_joint_preview_sweeps_all_eight_full_ranges_in_isolation():
    from frenchkiss.joint_test import JointTest
    sim=FrenchKissSim()
    test=JointTest(sim)
    test.enter()
    test.start_sweep(all_joints=True)
    minimum=np.zeros(8); maximum=np.zeros(8)
    for _ in range(4000):
        index=test.selected
        test.step(.05)
        minimum=np.minimum(minimum,sim.data.qpos)
        maximum=np.maximum(maximum,sim.data.qpos)
        other=np.delete(sim.data.qpos,index)
        np.testing.assert_allclose(other,0,atol=1e-8)
        if not test.sweep: break
    assert test.completed==list(range(8))
    np.testing.assert_allclose(minimum,sim.model.jnt_range[:,0])
    np.testing.assert_allclose(maximum,sim.model.jnt_range[:,1])
    np.testing.assert_allclose(sim.data.qpos,0,atol=1e-8)
    test.leave()
    assert not test.enabled
    np.testing.assert_array_equal(sim.data.qpos,[.04,-.06,.02,-.02,-.04,.06,-.02,.02])


def test_joint_physics_can_command_full_limit_beyond_agent_limit():
    from frenchkiss.joint_test import JointTest
    sim=FrenchKissSim()
    test=JointTest(sim)
    test.enter(); test.physical=True
    test.set_target(10)
    assert test.targets[0]==pytest.approx(CONFIG['joint_limit_rad'])
    for _ in range(1000): test.step(sim.model.opt.timestep)
    assert sim.last_controls['a'][0]==pytest.approx(CONFIG['joint_limit_rad'])
    assert np.isfinite(sim.data.qpos).all()
    sim.step({'a':np.ones(4),'b':np.zeros(4)})
    assert sim.last_controls['a'][0]<CONFIG['joint_limit_rad']


@pytest.mark.parametrize('index',range(8))
def test_wide_joint_travel_is_stable_in_isolation_on_both_tongues(index):
    from frenchkiss.joint_test import JointTest
    sim=FrenchKissSim()
    test=JointTest(sim); test.enter(); test.physical=True; test.select(index)
    # Separate the partner to measure actuator travel without contact obstruction.
    sim.model.body('a_base').pos[1]=-1
    sim.model.body('b_base').pos[1]=1
    mujoco.mj_forward(sim.model,sim.data)
    for direction in (-1,1):
        test.set_target(direction*CONFIG['joint_limit_rad'])
        for _ in range(1800): test.step(sim.model.opt.timestep)
        assert np.isfinite(sim.data.qpos).all()
        assert max(abs(sim.data.qpos))<CONFIG['joint_limit_rad']+.08
        assert np.all(sim.data.ctrl>=-CONFIG['max_tension_N'])
        assert np.all(sim.data.ctrl<=0)
        actual=sim.data.qpos[index]
        assert direction*actual>.55


@pytest.mark.parametrize('low_desire',[False,True])
def test_interaction_recovers_contact_and_keeps_moving(low_desire):
    sim=FrenchKissSim(); sim.interaction_style='explore'
    if low_desire: sim.personalities['b'].desire=.09
    sim.data.qpos[:]=[0,.65,.25,0,0,-.5,-.1,0]
    mujoco.mj_forward(sim.model,sim.data)
    sim.last_controls={side:sim.data.qpos[n*4:(n+1)*4].copy() for n,side in enumerate(('a','b'))}
    first_contact=None
    contacts=[]; speed=[]
    controls={}
    for tick in range(6000):
        if tick%6==0:
            tips=sim.tips()
            controls={s:scripted_controls(sim.data.time,s,sim.personalities[s],tips[s],tips['b' if s=='a' else 'a'],sim=sim) for s in ('a','b')}
        sim.step(controls)
        contact=sim.contact_metrics()['contacts']>0
        if contact and first_contact is None: first_contact=sim.data.time
        contacts.append(contact)
        speed.append(abs(sim.data.qvel.copy()))
    assert first_contact is not None and first_contact<6
    # Retreat phases deliberately break contact; later approaches must restore it.
    assert .1<np.mean(contacts[-2000:])<.98
    speed=np.array(speed[-2000:])
    assert speed[:,:4].mean()>.01 and speed[:,4:].mean()>.01
    assert np.isfinite(sim.data.qpos).all()


def test_saved_policy_uses_its_training_action_scale(tmp_path):
    from frenchkiss.agents import policy_action_scale
    (tmp_path/'a_training.json').write_text('{"config":{"target_limit_rad":0.45}}')
    assert policy_action_scale(tmp_path/'tongue_a_ppo.zip')==.45
    assert policy_action_scale(tmp_path/'a_round_000.zip')==.45


@pytest.mark.parametrize('field',list(Personality.__dataclass_fields__))
def test_personality_changes_actual_feedback_controller(field):
    sim=FrenchKissSim(); sim.data.time=2
    p=sim.personalities['a']; tips=sim.tips()
    setattr(p,field,0)
    low=scripted_controls(2,'a',p,tips['a'],tips['b'],sim=sim)
    setattr(p,field,1)
    high=scripted_controls(2,'a',p,tips['a'],tips['b'],sim=sim)
    assert not np.allclose(low,high)


def test_choreography_keeps_both_tongues_moving_over_ninety_seconds():
    from frenchkiss.interaction import interaction_phase
    sim=FrenchKissSim(); sim.interaction_style='explore'
    controls={}; positions=[]; contacts=[]; phases=set()
    for tick in range(18000):
        if tick%6==0:
            tips=sim.tips()
            controls={s:scripted_controls(sim.data.time,s,sim.personalities[s],tips[s],tips['b' if s=='a' else 'a'],sim=sim) for s in ('a','b')}
        sim.step(controls)
        if tick%6==0:
            positions.append(np.stack(list(sim.tips().values())))
            contacts.append(sim.contact_metrics()['contacts']>0)
            phases.add(interaction_phase(sim.data.time)[0])
    positions=np.array(positions)
    # Every three simulated seconds, both tips must visibly explore new space.
    for start in range(0,len(positions)-99,100):
        excursion=np.linalg.norm(np.ptp(positions[start:start+100],axis=0),axis=1)
        assert (excursion>.10).all()
    assert .1<np.mean(contacts)<.95
    assert 'A pulls away / B follows' in phases and 'B pulls away / A follows' in phases
    assert np.isfinite(sim.data.qpos).all()


def test_chaser_target_tracks_partner_actual_position_and_roles_swap():
    from frenchkiss.interaction import interaction_goal,interaction_phase
    sim=FrenchKissSim(); sim.interaction_style='explore'; sim.data.time=6.6
    before,_=interaction_goal(sim,'b')
    sim.data.qpos[0]+=.2
    mujoco.mj_forward(sim.model,sim.data)
    after,name=interaction_goal(sim,'b')
    assert name=='A pulls away / B follows'
    assert np.linalg.norm(after-before)>.05
    assert interaction_phase(24.6)[0]=='B pulls away / A follows'


def test_reference_profile_maintains_close_but_continuously_changing_motion():
    from frenchkiss.reference_motion import reference_phase
    sim=FrenchKissSim(); sim.interaction_style='reference'
    controls={}; positions=[]; forces=[]; phases=set()
    for tick in range(18000):
        if tick%6==0:
            tips=sim.tips()
            controls={s:scripted_controls(sim.data.time,s,sim.personalities[s],tips[s],tips['b' if s=='a' else 'a'],sim=sim) for s in ('a','b')}
        sim.step(controls)
        if tick%6==0:
            positions.append(np.stack(list(sim.tips().values())))
            forces.append(sim.contact_metrics()['contact_force'])
            phases.add(reference_phase(sim.data.time)[0])
    positions=np.array(positions); forces=np.array(forces)
    for start in range(0,len(positions)-99,100):
        assert (np.linalg.norm(np.ptp(positions[start:start+100],axis=0),axis=1)>.06).all()
    assert .7<np.mean(forces>.1)<.99
    assert np.mean((forces>.1)&(forces<35))>.7
    assert forces.max()<50
    assert 'A initiates / B responds' in phases and 'B initiates / A responds' in phases
    assert 'Brief separation' in phases and 'Reconnect' in phases
    assert np.isfinite(sim.data.qpos).all()


def test_reference_initiative_tracks_partner_and_style_switch_is_finite():
    from frenchkiss.reference_motion import reference_goal,reference_phase
    sim=FrenchKissSim(); sim.interaction_style='reference'; sim.data.time=6.9
    before,_=reference_goal(sim,'b')
    sim.data.qpos[0]+=.2; mujoco.mj_forward(sim.model,sim.data)
    after,name=reference_goal(sim,'b')
    assert name=='A initiates / B responds'
    assert np.linalg.norm(after-before)>.05
    assert reference_phase(21.9)[0]=='B initiates / A responds'
    for style in ('reference','explore','reference'):
        sim.interaction_style=style
        for _ in range(200): sim.step()
        assert np.isfinite(sim.data.qpos).all()
