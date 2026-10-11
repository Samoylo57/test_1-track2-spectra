# Track 2 Scattering Spectra CPU

Release `test_3-j5-h252-a0` disables PSMC (`alpha=0`) and uses the same unconditional branch evaluated in the cached paired experiment. Full contiguous prefix history, Battle-Lemarie Q=1/J<=5, target half-life 252 observations, 126 daily/20 monthly candidate context, bank 64x1024, 20,000 candidates, 1,000 joint draws, CPU/200 iterations are unchanged. Short prefixes use J=4.

Public registry: `ghcr.io/samoylo57/test_1-track2-spectra:test_3-j5-h252-a0`. Previous `test_1` and `test_2-j5-h252` images retain their immutable digests. Upload an officially signed descriptor ZIP bound to the new digest.

No task data, learned selector, stored forecasts, model checkpoint or House call is packaged. `category: api`, `models: []`. See `ARTIFACT_PROVENANCE.md` for source pins, retrospective selection evidence and limitations. CI builds linux/amd64, runs real forecasts under Development restrictions and official admissibility gates, verifies a same-seed rerun, then publishes.
