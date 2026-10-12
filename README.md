# Fourth Track 2 submission

Full-history CPU Spectra, Battle-Lemarie J<=5, uniform target statistics,
no PSMC. A disclosed local non-neural gradient booster adjusts the centre;
width x1.5. 60 trees, depth 2, leaf minimum 60, L2=10, learning rate .05,
centre strength .5, shift capped at +/-1 original forecast SD.

Correction: median + SD*clip(.5*booster(features),-1,1)
            + 1.5*(draw - median).

No unit lookup, model server, training histories, labels, cached answers,
neural checkpoints or House calls are packaged. JSON trees are SHA-pinned
and disclosed in the submission descriptor, category api/access local.
See ARTIFACT_PROVENANCE.md for exact training and selection sources.

The last training target is 2019-10-23 and the last calibration-selection
target is 2020-07-21. As required by the artifact policy, all learned
adjustments (including width) require task cutoff strictly after both.
Earlier historical tasks retain unchanged Spectra. The centre also falls
back to zero for monthly/short/transfer inputs; eligible width stays x1.5.

The seven generator modules and pinned CPU dependencies are reused from
the immutable third image. CI runs the real non-root, offline, read-only
linux/amd64 image through four official input fixtures, checks numerical
corrections, official output gates and same-seed repeatability, then pushes
the new release tag without changing previous image digests.
