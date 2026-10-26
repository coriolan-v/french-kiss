"""Articulated silicone approximation with antagonistic tendon transmissions.

Fixed tendons model ideal constant-radius cable moment arms, not routed cable
friction or continuum tissue. Dimensions and forces are simulation parameters.
"""
import mujoco
from .surface import tongue_skin, tongue_profile
from .config import CONFIG

SEGMENTS = CONFIG['segments']
LENGTH = CONFIG['length_m']
JOINT_NAMES = ('base_rotate', 'base_hinge', 'middle_hinge', 'tip_hinge')
JOINTS_PER_TONGUE = len(JOINT_NAMES)
MOMENT_ARM = CONFIG['cable_moment_arm_m']


def xml_scene(segments=SEGMENTS, length=LENGTH):
    if segments < 2 or length <= 0:
        raise ValueError('Positive length and at least two segments required')
    step = length / segments
    anchor=CONFIG['anchor_half_span_m']
    density=CONFIG['effective_density_kg_m3']
    parts = ['''<mujoco model="French Kiss - tendon study">
    <compiler angle="radian" autolimits="true"/>
    <option timestep="0.005" integrator="implicitfast"/>
    <visual><global offwidth="1280" offheight="800"/><quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.4" diffuse="0.7 0.7 0.7"/></visual>
    <default><joint damping="3" armature="0.04" stiffness="28" range="-0.55 0.55"/>
    <geom friction="0.65 0.05 0.01" solref="0.04 1" solimp="0.85 0.95 0.003"/></default>
    <asset><texture type="skybox" builtin="gradient" rgb1=".035 .045 .075" rgb2=".12 .14 .20" width="512" height="3072"/><texture name="ground" type="2d" builtin="checker" rgb1="0.055 0.065 0.09" rgb2="0.075 0.085 0.11" width="256" height="256"/>
    <material name="stage" texture="ground" texrepeat="8 8" reflectance="0.15"/>
    <material name="rose" rgba="0.85 0.26 0.38 1" specular="0.65" shininess="0.6"/>
    <material name="pearl" rgba="0.52 0.39 0.72 1" specular="0.65" shininess="0.6"/></asset>
    <worldbody><light pos="0 -2 4" dir="0 0 -1"/><light pos="1 2 3" dir="-0.3 -0.5 -1" diffuse="0.5 0.4 0.45"/>
    <geom name="floor" type="plane" size="5 5 .1" material="stage"/>
    <camera name="front" pos="0 -3.7 2.2" xyaxes="1 0 0 0 .26 .96"/>
    <camera name="overhead" pos="0 0 4.8" xyaxes="1 0 0 0 1 0"/>''']
    for side, sign, quat, material in [('a', -1, '1 0 0 0', 'rose'), ('b', 1, '0 0 0 1', 'pearl')]:
        parts.append(f'<body name="{side}_base" pos="{sign*anchor} 0 1.25" quat="{quat}"><geom type="cylinder" pos="0 0 -.65" size=".14 .58" rgba=".12 .13 .17 1" contype="0" conaffinity="0"/>')
        for i in range(segments):
            width,thickness = tongue_profile((i+.5)/segments)
            parts.append(f'<body name="{side}_link{i}" pos="{0 if i==0 else step} 0 0">')
            joints = [(JOINT_NAMES[0], '0 0 1'), (JOINT_NAMES[1], '0 1 0')] if i == 0 else ([(JOINT_NAMES[2], '0 1 0')] if i == segments//2 else ([(JOINT_NAMES[3], '0 1 0')] if i == 3*segments//4 else []))
            for name,axis in joints:
                parts.append(f'<joint name="{side}_{name}" axis="{axis}"/>')
            parts.append(f'<geom name="{side}_segment{i}" type="ellipsoid" pos="{step/2} 0 0" size="{step*.61} {width} {thickness}" density="{density}" material="{material}" group="4"/>')
            parts.append(f'<geom name="{side}_rib{i}" type="capsule" fromto="{step/2} {-width*.7} 0 {step/2} {width*.7} 0" size=".007" rgba=".9 .8 .7 .35" contype="0" conaffinity="0" group="2" mass="0"/>')
            parts.append(f'<site name="{side}_marker{i}" pos="{step} 0 .04" size=".008" rgba="1 .85 .4 1" group="3"/>')
        parts.append(f'<site name="{side}_tip" pos="{step} 0 0" size=".012" rgba="1 .8 .6 1" group="3"/>')
        parts.extend(['</body>']*(segments+1))
    parts.append('</worldbody><deformable>')
    for side,sign,material in [('a',-1,'rose'),('b',1,'pearl')]:
        parts.append(tongue_skin(side,sign,length,segments,material))
    parts.append('</deformable><tendon>')
    for side in ('a','b'):
        for joint in JOINT_NAMES:
            for direction in (-1,1):
                name=f'{side}_{joint}_{direction}'
                parts.append(f'<fixed name="{name}"><joint joint="{side}_{joint}" coef="{direction*MOMENT_ARM}"/></fixed>')
    parts.append('</tendon><actuator>')
    for side in ('a','b'):
        for joint in JOINT_NAMES:
            for direction in (-1,1):
                name=f'{side}_{joint}_{direction}'
                parts.append(f'<motor name="{name}" tendon="{name}" ctrllimited="true" ctrlrange="-600 0" forcelimited="true" forcerange="-600 0"/>')
    parts.append('</actuator></mujoco>')
    return '\n'.join(parts).replace('-600 0',f'-{CONFIG["max_tension_N"]} 0').replace('range="-0.55 0.55"',f'range="-{CONFIG["joint_limit_rad"]} {CONFIG["joint_limit_rad"]}"').replace('stiffness="28"',f'stiffness="{CONFIG["joint_stiffness_Nm_rad"]}"').replace('timestep="0.005"', f'timestep="{CONFIG["timestep_s"]}"')


def make_model():
    return mujoco.MjModel.from_xml_string(xml_scene())


def joint_ids(model, side):
    return [model.joint(f'{side}_{name}').id for name in JOINT_NAMES]


def tip_pos(model, data, side):
    return data.site_xpos[model.site(f'{side}_tip').id].copy()
