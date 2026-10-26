# French Kiss - digital exhibition

Two broad, flat 1.2 m tongues in an offline MuJoCo simulation, with audience controls, recording/replay and independent experimental PPO agents. No physical hardware is needed.

## Start

Double-click **Launch-French-Kiss.bat**. It starts the **Reference / scripted** profile, informed by the supplied video: close adjustments, small curls, alternating initiative and short separation/reconnection. Press **V** to compare with the larger exploration/chase profile. Experimental PPO is opt-in with **T**, or `Launch-French-Kiss.bat --trained`.

From PowerShell in this directory:

```powershell
.venv\Scripts\python.exe -m scripts.demo
# Experimental raw PPO comparison:
.venv\Scripts\python.exe -m scripts.demo --trained
```

- Sliders: independent personalities for each tongue.
- V: switch **reference-inspired** / **larger exploration-chase** styles.
- T: switch between the active scripted interaction and experimental PPO (loads the local saved pair when first selected).
- Space: pause. R: reset. Escape: close.
- 1/2/3: camera views. Right-drag: orbit. Wheel: zoom.
- D: marker/rib debugging and synthetic stereo diagnostics.
- S: start/stop a state recording in `recordings/`.

## Joint test mode

Double-click **Test-Joints.bat**, or press **J** in the main viewer to enter/leave the test panel. It disables the agents while testing.

- Select tongue A or B and a joint. Drag any of its four angle sliders.
- **Minimum / Zero / Maximum** command the selected joint to its configured limits (currently -63.0 to +63.0 degrees) or neutral.
- **Sweep joint** repeatedly travels through minimum, maximum and zero.
- **Test all 8** centers the other joints, sweeps every joint on A then B once, and stops. A full sequence takes about 141 seconds, excluding pauses.
- **Center all** stops the sweep and sets all targets to zero. Selecting a joint or moving a slider also stops the sweep.
- **Physics on/off** switches between exact pose inspection and the force-driven tendon simulation. Target and actual angles are displayed separately.

The default **Pose preview** pauses contact dynamics and directly positions the mechanism, so you can inspect the complete configured range even when the two tongues intersect. It is a kinematic inspection, not evidence that the motors can achieve that pose. **Tendon physics** retains gravity, contact, actuator limits and slow commands; actual travel can be smaller than the target. The progress counter counts commanded sweeps, not successful physical-limit attainment. Normal controllers now use a wider +/-0.9 rad (about +/-52 degree) target range; joint tests use the full +/-1.1 rad (about +/-63 degree) range. Scripted amplitudes are doubled. Passive joint stiffness was reduced to permit the extra travel within the existing tendon-force bounds. Slow command speeds and playback remain enabled. New training uses the wider action range. Existing policies are replayed with the action scale stored in their training metadata; their outputs are no longer doubled, which previously caused separation. Those saved networks have not been retrained for the changed stiffness and remain experimental.

```powershell
.venv\Scripts\python.exe -m scripts.demo --joint-test
.venv\Scripts\python.exe -m scripts.demo --joint-test --sweep-all
```

Space pauses either mode; camera controls remain available. Recordings identify pose previews and suppress physical-force claims during replay.

## Current mechanism: four degrees of freedom per tongue

Exactly four independently actuated joints, in action-vector order:

1. **Base rotation** about the vertical axis (left/right yaw).
2. **Base hinge** for up/down bending at the mount.
3. **Middle hinge** 0.60 m along the tongue.
4. **Tip hinge** 0.90 m along the tongue, bending the final 0.30 m.

Eight visual/collision sections form three rigid bending spans; they do not add extra joints. There are eight total simulation coordinates and sixteen antagonistic tendon motors (two per joint). The observation has 27 floats: eight positions, eight velocities, six tip coordinates and five personality values. Each policy outputs four actions.

The mounts are now **1.98 m apart**, allowing the broad ends of the 1.2 m tongues to overlap and contact. Tip-to-tip separation can remain positive when the tongues overlap and their surfaces are touching; the contact-force HUD reports actual collision contacts.

The tongues are approximately 44 cm wide and 7 cm thick, with broad sides, rounded ends and a shallow upper groove. Exhibition playback runs at **0.3x speed**, preserving small physics steps; joint targets also slew slowly. The effective foam density is 120 kg/m3. Defaults live in `simulation-config.json`; these are artistic simulation parameters, not measured mechanical properties.

## Training and digital checks

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m scripts.train --steps 100000 --rounds 3 --output models/four-dof-experiment-02
.venv\Scripts\python.exe -m scripts.demo --policy-a models/four-dof-experiment-02/tongue_a_ppo.zip --policy-b models/four-dof-experiment-02/tongue_b_ppo.zip
.venv\Scripts\python.exe -m scripts.evaluate --headless --baseline --steps 6000
.venv\Scripts\python.exe -m scripts.evaluate --headless --model-dir models/four-dof-v1 --steps 6000
.venv\Scripts\python.exe -m scripts.demo --seconds 10 --record recordings/example.jsonl --screenshot example.png
.venv\Scripts\python.exe -m scripts.demo --replay recordings/example.jsonl
```

Two separate networks train alternately against frozen partner snapshots, with some scripted-partner episodes. Personalities vary during training. Rewards include actual contact forces, relative tip movement, novelty, withdrawal incentives and effort/aggression/static-contact penalties. These remain heuristic artistic objectives. Training steps are per agent per round. CPU training defaults to two Torch threads. CSV episode logs, checkpoint/configuration metadata and saved partner snapshots are written to the output folder. Use a new folder for a fresh experiment.

`models/four-dof-v1/` holds the current experimental policies. Old `models/contact-v1/` policies and previous 32-coordinate recordings belong to the old mechanism and cannot be used with four joints. Requested incompatible policies/recordings produce an explicit error. Generated model files and recordings remain local and are ignored by Git; archive them separately when sharing the installation.

## Implemented and remaining

Implemented: four-joint articulated mechanics, ideal tension-only tendon transmissions, broad skinned surfaces, live independent audience controls, single-window 3D viewer, slow playback, camera presets, contact telemetry, recording/replay, seeded PPO environments, frozen partner pools, synthetic stereo tracking with noise/dropout, and headless digital evaluation.

Approximate or planned: real spatial cable routing, slack/friction, calibrated tissue mechanics, reliable learned artistic personality/reciprocity, curriculum progression, image-based marker detection and occlusions, tracking-conditioned policies, video export and adaptive window sizing. The skin is visual; collisions use articulated rigid ellipsoids. Trained movement is explicitly labelled and is not assumed to outperform the scripted baseline. There is no hardware output path.

## Setup on another machine

Python 3.11 is supported. This PC already has a project-local runtime and dependencies.

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

`requirements-lock.txt` records the original tested Windows environment, including optional packages. Operation is offline after installation. `VALIDATION.md` records measured results and which mechanism they apply to.

Modules: `model.py` mechanics, `surface.py` visual skin, `simulation.py` physics/tendon commands, `agents.py` controllers, `rl_env.py` training, `tracking.py` synthetic perception. Entry points are `scripts/demo.py`, `train.py`, and `evaluate.py`.

## Active interaction and experimental policies

Both styles are scripted, not learned AI. The larger **Explore** style cycles through **Meet**, **Explore together**, **A pulls away / B follows**, **Explore apart**, **B pulls away / A follows**, and **Reunite**. The HUD names the active phase. Lead/follow roles swap on alternate cycles, and exploration paths vary between cycles rather than settling on one shared point. An 18-second simulation cycle takes about one minute at the slow 0.3x playback speed; substantial motion continues throughout.

The chaser follows the other tongue's actual moving tip. The controller combines changing 3D goals, tip Jacobian feedback, gravity compensation, curl/posture variation, and yielding during loaded contact. Audience desire, curiosity, confidence, playfulness and reciprocity influence retreat, exploration and pursuit. Contact is deliberately intermittent; maximizing time touching is no longer the exhibition objective.

A 90-second regression check requires both tips to explore at least 10 cm in every three-second simulation window and includes both chase roles, contact breaks and reunion. The headless evaluator now reports the minimum tip excursion in three-second windows, so motion quality is assessed alongside contact fraction. `interaction-motion-evaluation.json` records a measured longer run.

The raw PPO pair remains available for comparison but can settle or separate, especially after mechanical changes. Selecting it is explicitly labelled **Experimental PPO**. Saved policies now retain their training action scale from metadata rather than being silently remapped to a wider range. Missing metadata falls back to the current configuration. Learned behaviour still needs further training and artistic evaluation.

## Supplied video reference

The local `french-kiss-video-training.mp4` was inspected using an overview and one-second samples of the contact sequence at 24-35 seconds and pause/reconnection at 71-82 seconds. It is a 125.87-second, 30 FPS screen recording with player interactions/cuts. The interpretation focuses on close, continuously changing adjustments, distal curls, alternating initiative, and short withdrawal/reconnection. It does not infer accurate 3D tongue trajectories or physical forces from a single video.

`reference-motion-profile.json` records the source metadata, sampled sections, observations and hand-tuned movement parameters. The video itself is not copied into the project and is not needed at runtime. Local analysis thumbnails are ignored by Git.

**Reference / scripted** is the new default. Its phases are Approach, Close exploration, A initiates/B responds, Brief separation, B initiates/A responds, and Reconnect; roles alternate across cycles. It keeps the wide joint-test ranges while using more concentrated movements near the partner. At 0.3x speed a 15-second simulation cycle takes about 50 wall-clock seconds. The earlier large-motion style remains available with V or `--style explore`.

```powershell
.venv\Scripts\python.exe -m scripts.demo --style reference
.venv\Scripts\python.exe -m scripts.demo --style explore
.venv\Scripts\python.exe -m scripts.evaluate --headless --baseline --style reference --steps 18000
```

This is **reference-informed choreography**, not neural imitation learning or a policy trained on the clip. The existing experimental PPO weights were not changed by this work. A new 90-second regression checks continued motion, contact and reconnection in the reference style; the larger exploration style retains its separate movement tests. Results are recorded in `reference-style-evaluation.json`.
