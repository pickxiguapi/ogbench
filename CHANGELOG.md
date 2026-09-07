# Changelog

## Release branch

- Added one canonical `LeWMPPController` implementing selectable subgoal targets, final-goal action-prior initialization, and MoH/terminal scoring.
- Consolidated MLP, Endpoint Flow, and LatentPath Flow behind one generator training and inference interface; every type supports bounded H25 and general full-future sampling.
- Added zero, policy-mode, and policy-mode-anchor CEM initialization while enforcing final-goal-only policy conditioning.
- Reduced action-prior training to the shared-all GCIQL-Chunk-AWR configuration used by the paper.
- Added standalone LeWM CEM and GCIQL-AWR-Chunk evaluation variants alongside the three LeWM++ ablations.
- Consolidated 142 dated and server-specific experiment scripts into six public launchers plus one shared helper.
- Removed archived implementations, internal reports, result snapshots, private server paths, staged planning, population guidance beyond the two paper modes, path-mean scoring, and obsolete visual-decoder/ACID studies.
- Added strict generator-family validation and provenance-separated output directories.
- Added release tests, cross-platform CPU JAX installation, a CUDA optional extra, and an end-to-end README.
