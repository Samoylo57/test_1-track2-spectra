# Track 2 Scattering Spectra + PSMC CPU

Release test_2-j5-h252 uses full supplied contiguous history, Battle-Lemarie scales capped at J=5, and historical target weights with half-life 252 input observations. Daily PSMC uses 126 observations; monthly PSMC uses 20. Short prefixes select a smaller supported J.

The image is built and validated by GitHub Actions before publication to ghcr.io/samoylo57/test_1-track2-spectra:test_2-j5-h252. It contains code and dependencies only. The upload ZIP is sealed separately using the official toolkit and the immutable registry digest. Category api, no House calls, no packaged fitted models. See ARTIFACT_PROVENANCE.md.
