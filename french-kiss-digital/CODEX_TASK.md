# Codex brief — French Kiss

You are developing a fully offline, physically simulated interactive art installation: two 1.2 m giant tongue mechanisms, each with an autonomous embodied AI agent. The audience adjusts personality parameters that affect their behaviours. No hardware is required for the current phase. The system should be testable end-to-end on a mid-range Windows gaming PC.

Read README.md, inspect the existing MuJoCo simulation, and **first run the tests and the demo**. Fix any issues found. Keep the physics simulation and production motor-control safety separate.

Goals:
1. Robust MuJoCo digital twin with anatomical tongue-shaped soft approximation, realistic hinged backbone/ribs and tendon actuation, detailed joint limits and collisions.
2. Two independent neural-network policies with agent-specific behavioural conditioning (desire, curiosity, confidence, playfulness, reciprocity). Add actual PPO training with appropriate partner pools or a multi-agent RL library. The current scripts are only single-agent scaffolds.
3. Contact-based rewards that distinguish gentle reciprocal touching from aggressive or static collisions; meaningful novelty and withdrawal behaviours.
4. Live interactive app: 3D viewer + personality controls, training replay, camera debug overlays, status HUD, recordable demos. Offline on a mid-range PC.
5. Virtual camera observation and marker-based 3D tracking pipeline (hardware cameras optional later).
6. Clear module boundaries, runnable commands, tests, versioned config, logging and documentation.

Do not misrepresent scripted movement as AI-learned behaviour or pretend simulation is validated as mechanically accurate. Deliver tested incremental milestones rather than promising all features are complete.
