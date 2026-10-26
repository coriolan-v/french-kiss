# Video-reference profile - 19 October 2026

Inspected the supplied 125.87-second, 1920x1080, 30 FPS screen recording using a 12-frame overview and one-second sequences at 24-35 s and 71-82 s. These show close repeated adjustment and distal curling, changes of initiative, and a short pause/separation followed by reconnection. Player interactions/cuts prevent interpreting the whole file as a single continuous motion capture sequence.

Implemented a hand-tuned **Reference / scripted** profile, now default, with close exploration, alternating initiator/responder, a brief separation and reconnection. V toggles the prior larger exploration/chase profile. `reference-motion-profile.json` records inspected intervals and tuning. No neural policy was trained from the video; no 3D reconstruction or force estimation is claimed.

- **37 tests passed**, including 90-second reference-style and exploration-style runs, anti-stagnation checks, actual-partner targeting, style switching, all five audience controls, contact recovery and all joint diagnostics.
- Reference-style 90-second evaluation at the exhibition's control rate: **87.44% contact**, also 87.44% within the provisional 0.1-35 N gentle threshold; peak solver force **28.15 N**; minimum three-second tip excursions **0.099 m for A / 0.112 m for B**; finite state. See `reference-style-evaluation.json`. Solver force values are not mechanically calibrated.
- Reference-style viewer rendering and recording exited successfully. Screenshot inspected. The source video remains in its original location; runtime uses only the motion-profile parameters. Analysis screenshots are local and Git-ignored.

Earlier entries describe preceding controller revisions.

# Changing exploration and chase phases - 19 October 2026

The previous contact-seeking controller could spend too long around a common pose. Contact fraction and small joint velocities did not adequately test whether the exhibition remained visually active.

- The scripted controller now cycles through meeting, exploring together, A retreat/B follow, exploring apart, B retreat/A follow, and reunion. Roles swap each cycle, paths vary between cycles, and the follower targets the partner's **actual** tip position. Loaded contact invokes yielding. The HUD names the current phase.
- **35 tests passed**, including a 90-second regression requiring both bodies to move their tips at least 10 cm in every three-second window; a partner-following/role-swap check; separated-start recovery; wider joint testing; and all five audience controls.
- Measured 90-second run: minimum three-second tip excursions **0.150 m for A / 0.226 m for B**; contact **58-59%**, gentle-threshold contact **57-59%** across the recorded checks; finite state. Peak aggregate solver contact force was about 87 N during a transient; these forces remain uncalibrated simulation values. Exact results and full spatial spans are in `interaction-motion-evaluation.json`; the standard evaluator output is in `choreography-evaluation.json`.
- Contact is intentionally intermittent. The recovery regression now checks recurring contact plus motion rather than demanding near-continuous touching, which conflicted with retreat and exploration.
- Normal slow playback and configured joint travel/force limits remain in place. An entire phase cycle takes 18 simulated seconds (about one minute of wall time at 0.3x playback). The exhibition remains explicitly labelled scripted; no new learned behaviour is claimed.

Earlier measurements below describe previous controller revisions.

# Separation/recovery fix - 19 October 2026

The previous default launch used raw PPO checkpoints trained before the amplitude/stiffness changes. Those networks had also been remapped from their trained +/-0.45 rad action scale to +/-0.9 rad. They could separate and converge to stationary poses. Increasing actuator range did not produce learned reciprocal behaviour.

The default launch now explicitly uses **Scripted interaction**: Jacobian tip feedback toward a moving shared encounter point, gravity/passive-spring compensation, personality-conditioned curl/exploration and brief withdrawal pulses. It is not described as learned AI. Raw PPO remains opt-in with T or --trained and is labelled Experimental PPO. Saved policies replay with their recorded training action scale; the same metadata scaling is used for evaluation and partner-pool replay.

- **33 tests passed**, including two deliberately separated-start recovery scenarios (default personalities and B desire 0.09), ongoing movement for both tongues, conditioning effects for all five controls on the actual feedback controller, and metadata action scaling.
- Recovery regression uses held commands updated every six 5 ms steps, matching the exhibition's approximately 33 Hz controller rate. Both recovery scenarios reestablish contact within six simulated seconds, sustain contact for over 80% of the last ten seconds and retain measurable motion on both bodies.
- A 60-second default-start evaluation at the same controller rate recorded **94.95% contact**, all within the provisional gentle-force threshold, peak solver contact force **12.97 N**, and mean absolute joint speed **0.04085 rad/s**. Exact output is in `active-interaction-evaluation.json`. These forces are not mechanically validated.
- Normal scripted exhibition rendering exited successfully; the screenshot was inspected. HUD shows Approaching/recovering or Touching/exploring. Slow playback remains 0.3x, so initial recovery takes longer in wall-clock time than simulation time.

All measurements below refer to earlier controller/mechanism revisions and remain historical.

# Increased-amplitude revision - 19 October 2026

- Joint travel doubled from +/-0.55 rad to **+/-1.1 rad** (about +/-63 degrees), for every axis on both tongues. Normal action scaling increased from +/-0.45 to **+/-0.9 rad** (about +/-52 degrees), including training, evaluation and loaded policy playback. Scripted amplitude gain is 2.0.
- Passive joint stiffness reduced from 28 to 8 Nm/rad so the larger travel is accessible within the existing tensile-force bounds. Command slew and slow exhibition playback are unchanged.
- **25 tests passed**: full-range kinematic sweeps, eight physical travel checks (both signs for every axis, with the partner separated to avoid collision obstruction), normal simulation stability and existing interface/controller tests.
- In physical tests all axes exceeded the old 0.55 rad travel magnitude in both directions while remaining finite and within joint/tension bounds. Actual travel during close tongue-on-tongue contact may be blocked; pose preview remains the full-range visual inspection.
- Normal trained rendering and joint-test rendering exited successfully. The diagnostic range display updates from the model limits. Existing policies are rescaled, not retrained for the changed travel/stiffness; earlier behaviour/contact measurements below are historical.

# Joint test mode validation

- **17 tests passed.** An automated kinematic sweep exercised all eight joints in isolation, verified both configured limits were reached exactly, checked nonselected joints remained centered, and verified reset on exit.
- A separate tendon-physics test verified that the diagnostic controller can command the full 0.55 rad joint limit while normal agents retain the 0.45 rad target limit, with finite simulation state.
- The joint-panel render, sweep startup, recording and separate pose-preview replay exited successfully. The panel screenshot was inspected visually.
- Joint test disables agents. Pose preview uses exact kinematic positioning and explicitly pauses contact response. Optional tendon physics reports actual versus requested angles; its travel may be collision/force limited. Sweep completion counts commands, not mechanically validated travel.

# Four-degree-of-freedom revision

Current mechanism: exactly four joints per tongue (base yaw, base pitch, middle pitch at 0.60 m, tip pitch at 0.90 m). Eight visual sections remain, with only these four kinematic freedoms. Mounts are 1.98 m apart. Slow 0.3x exhibition playback and 0.3 rad/s target slew remain enabled. Effective foam density is 120 kg/m3.

- **15 tests passed**, including the four joint axes and pivot-body locations, action dimensions, long-run stability, contact filtering, all personality controls, and Gymnasium behaviour.
- Both new PPO policies trained for **16,384 steps each**, in two alternating rounds, using seeds 7 and 8. Files are in `models/four-dof-v1/`; older mechanisms' policies are preserved separately and explicitly rejected by the exhibition loader.
- Trained viewer, eight-second recording, and separate replay checks exited successfully. Screenshot `four-dof-trained-preview.png` was inspected.
- A 30-second headless evaluation (6,000 physics steps) gave:

| Metric | Scripted | Four-DOF PPO pair |
| --- | ---: | ---: |
| Contact fraction | 99.50% | 99.53% |
| Gentle contact fraction (0.1-35 N) | 99.50% | 99.53% |
| Peak solver contact force | 8.09 N | 20.31 N |
| Mean absolute joint speed | 0.00663 rad/s | 0.00596 rad/s |
| Finite state | yes | yes |

Exact measurements are in `four-dof-baseline-evaluation.json` and `four-dof-learned-evaluation.json`. High contact reflects the closer geometry and can include largely static touching; it is not proof of learned reciprocity or artistic expressiveness. Both networks are short experimental training runs. Numerical forces remain unvalidated solver outputs.

The broad tongue ends now overlap in reach and touch along their surfaces. Positive tip-to-tip separation during this overlap is expected and does not mean there is no contact. The launcher selects only compatible four-DOF policies. Previous 32-coordinate recordings are rejected with an explicit compatibility message.

## Historical validation of the earlier 32-joint mechanism

Everything below describes the previous mechanism and is retained for provenance; it does not describe the current model.

# Development validation - 18 October 2026

This pass installed a workspace-local CPython 3.11.15 environment, installed dependencies, ran the original two tests (both passed), and launched the original native MuJoCo viewer without logged errors before changing the implementation.

## Final automated checks

- `python -m pytest -q`: **11 passed**. Includes 3,000-step stability, signed tendon response, tensile bounds, rejection of invalid actions, contact filtering, deterministic resets/truncation, calibrated stereo reconstruction/dropout, all five independent baseline controls, and learned action scaling.
- Gymnasium `check_env` passed for both sides during PPO training.
- Final training command was also smoke-tested after adding CSV episode logging: 512 steps, side A, output `models/final-pipeline-check`; checkpoint, metadata and episode CSV produced.
- Scripted renderer, trained renderer, synthetic tracking overlay, recording and replay all exited successfully in finite-duration checks. The final trained run recorded five seconds to `recordings/final-check.jsonl`, followed by a separate replay invocation. Screenshots were inspected visually.
- Exhibition rendering reached the 30 FPS target in short checks; this is not a hardware benchmark or a long-duration performance guarantee. Physics runs at 200 Hz; policies act about 33 times per second. Inference and training use CPU, with two Torch threads for the exhibition.

## Local neural networks

`models/contact-v1/tongue_a_ppo.zip` and `tongue_b_ppo.zip` are separate PPO networks trained for **32,768 steps each** (16,384 per round, two alternating rounds). A used seed 7 and B seed 8. First A learned against the scripted partner; later rounds used frozen partner checkpoints with 20% scripted episodes. The metadata includes the model configuration and partner paths. These models were trained on the final contact-capable geometry.

Earlier experimental checkpoints in `models/` were trained before the final anchor geometry. Use `models/contact-v1/` for the delivered exhibition. Do not infer compatibility or quality from old checkpoint filenames.

Changing each personality from 0 to 1 at the same initial observation changed each trained agent's deterministic action vector (L2 differences 0.19-0.45 across the ten checks). This verifies that the networks respond to conditioning. It does **not** prove that the resulting motion consistently expresses the named personality trait.

## 30-second digital evaluation

6,000 physics steps from the default reset and default personalities, holding actions for six steps. Exact output is in `baseline-evaluation.json` and `learned-evaluation.json`.

| Metric | Scripted baseline | Trained pair |
| --- | ---: | ---: |
| Any inter-tongue contact (>0.1 N) | 70.52% | 10.95% |
| Provisional gentle contact (0.1-35 N) | 70.50% | 10.82% |
| Peak aggregate contact force | 35.74 N | 37.81 N |
| Mean tip separation | 0.104 m | 0.130 m |
| Mean absolute joint speed | 0.014 rad/s | 0.718 rad/s |
| Finite final state | yes | yes |

The scripted baseline sustains considerably more contact. The learned pair moves more but has **not surpassed the baseline** at sustained gentle interaction. This is a functioning training and exhibition milestone, not a converged learned kissing behaviour. Multiple seeds, personality sweeps and longer artistic assessment remain necessary. These reported forces are solver outputs, not validated mechanical forces. Tip separation can remain positive during contact on the tongues' surfaces.

## Outstanding requirements

- Spatial cable routing, friction/slack, continuous tissue mechanics and material calibration. Current fixed-tendon moment arms and visual skin are explicit approximations.
- More training and reward tuning to obtain sustained reciprocal movement and reliable withdrawal/personality effects. No curriculum progression is implemented yet.
- True rendered-image marker detection, occlusion reasoning, and tracking-based agent observations. Current synthetic projections start from known 3D marker positions.
- Adaptive window sizing, video export and a training dashboard.
- Physical hardware and its safety/controller stack, intentionally deferred.

All testing above is digital. No hardware output interface was added.
