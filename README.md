# Track 2 Scattering Spectra CPU

Release `test_3-j5-unweighted-a0`: full contiguous supplied prefix, Battle-Lemarie Q=1/J<=5 (short prefixes J=4), uniform historical target statistics, alpha=0 (PSMC disabled). Same CPU synthesis, joint PCA, 64x1024 bank, 20000 candidates, 1000 joint draws, 200 iterations and tolerance 0.006.

`QFBENCH_SS_TARGET_HALF_LIFE=none` explicitly disables time weighting. No House calls, task data, fitted checkpoints, evaluation outcomes or cached trajectories are packaged. Numerical source files are unchanged from the prepared h252/alpha-zero release.

This fixed-setting Development ablation was selected by retrospective calibration under the published expected M0 normalization/cap 8. It improves calibration but worsens the already inspected holdout against h252/alpha-zero; it is not independently validated. See ARTIFACT_PROVENANCE.md for software and selection disclosure.

The public GHCR image is checked under offline/read-only/non-root CPU limits, including single, joint, monthly and transfer fixtures. The official toolkit ZIP contains an immutable image digest and signed team claim. Older image tags and ZIPs remain separate.
