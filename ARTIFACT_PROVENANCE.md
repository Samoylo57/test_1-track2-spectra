# Test 4: Spectra + selected non-neural residual centre, width x1.5

Source date: 2026-10-12. Category api; House calls optional and unused.
The numerical Spectra generator is exactly the published third image,
immutable base digest cc016aeffd046cd164be90fa0c11fa2bd5fefbf801359f9f4009b43e04246402.
Full supplied contiguous history, Battle-Lemarie Q1, J<=5, uniform target
statistics, CPU 200 iterations, no PSMC, 1000 joint draws remain unchanged.

## Packaged learned artifact

models/center.json SHA256: 4381cf9ee199595055d9d3df180bde449db6d10a8d5f278b692960d0fb708952; own model/integration MIT.
HistGradientBoostingRegressor trained/exported with scikit-learn 1.7.2.
Squared error, 60 trees, depth 2, max 4 leaves, min_samples_leaf=60,
learning_rate=.05, L2=10, early_stopping=False, random_state=0.
Target=(actual-median_Spectra)/SD_Spectra, clipped +/-6 for training;
weight=1/(horizons*assets), normalized to sum N. Prediction delta is
SD_Spectra*clip(.5*booster(features),-1,1). Width is fixed 1.5.
No refit or network access occurs during inference. Model JSON stores only
split thresholds, leaf values, baseline and provenance; no training examples,
task answers, panel histories, caches, task IDs or asset lookup table.

## Training, selection and cutoffs

197 frozen CPU Spectra prefix forecasts produce 952 normalized residual
cells from the input panels of 54 calibration cards; 19 January origins
2001-2019, 18 series. Last training target: 2019-10-23.
Data source: Agenthon-2026/track2-forecasting-public public input panels,
pinned source 2299dab3d31f5e93c10df087af86ffb0b7bfe938, MIT repository;
underlying public data source attribution is retained by the organizers.
Training plan SHA256 4a2c8af3752bb2ba1292a6ec687be3f32a98652fc49d2b3a103266cd34ad291d.
Frozen forecast manifest SHA256 d2f9e0ba614e2199766337bd0898c0946ff84f89e350cb8eb0f167b9c20cf856.
Parameter selection SHA256 e76006a05340df0c4df3d5c3d6780a20dad9ed2c2126dcc68547b40b99280d6c.
Parameters and width were selected on 54 retrospective calibration cards,
last resolved calibration target 2020-07-21; the 19 control cards did not select
parameters. Historical data-release vintages have not been reconstructed;
the local results are retrospective, not independent historical validation.

All learned adjustments, including width, require task asof strictly after
both training and parameter-selection cutoff. Earlier tasks use unchanged
Spectra (centre 0, width 1). The centre additionally requires daily frequency,
J5, >=257 observations and contiguous current history; monthly/short/gapped
transfer inputs fall back to centre zero. After eligible dates the width
remains 1.5 even when the centre falls back. Dates are guards, never predictive
features. Features use only the current panel and current uncorrected draws.
Positive affine transforms retain joint draw identity and rank dependence.

This guard means the 54 historical calibration cards reproduce base Spectra
in the production agent; the research calibration improvements are not a
causal claim for those earlier dates. The 19 later control cards reproduce
the tested selected correction exactly. They were already examined and are
not a fresh independent evaluation.

## Pinned software and source provenance inherited from version 3



- Own integration, PSMC, foveal embedding and joint PCA extension: MIT, this
  workspace. Files are listed and SHA-256 hashed by `prepare_image.py`.
- Rudy Morel Scattering Spectra 2.0.3 numerical core: MIT, commit
  `48920143c4b702c220b22dd2486b9eb0e852592e`. Exact file hashes and upstream
  license are in `vendor/scatspectra-manifest.json` and `vendor/LICENSE-scatspectra.txt`.
  A minimal package initializer excludes the upstream S&P/VIX dataset frontend.
  Fixed Battle-Lemarie filters contain no pretrained neural weights. Only
  synthetic trajectory values are optimized to match prefix Phi1-Phi4 statistics.
- Reference method: Morel et al., arXiv:2308.01486v1 (2023-08-02), and the
  shadowing implementation at `751a800302e7482a1fc96820726e2141135cc8a0`.
  The latter is not packaged; PSMC and its embedding are our NumPy implementation.
- QFBench2 common 2.4.4, tag v2.4.4, source commit
  `68bc878a740eb0040e70ddf0feab98405dca1d6d`: CC-BY-NC-4.0 (third-party license
  retained in its wheel). Track2 scorer/helpers 3.1.0, source commit
  `2299dab3d31f5e93c10df087af86ffb0b7bfe938`: MIT;
  its source license is included in the preparation manifest. Official helpers
  resolve target semantics and monthly periods. These source distributions
  contain no fitted predictive model.
- Python 3.13, NumPy, SciPy, pandas, pyarrow, scikit-learn, threadpoolctl,
  PyTorch 2.8.0 CPU, matplotlib, termcolor and tqdm: software dependencies,
  their upstream licenses apply. Versions are pinned in requirements; installed
  versions are recorded inside the image. No neural checkpoint is loaded.

Source-code release dates are software provenance, not a training-data cutoff.
Source commits pin the numerical implementation; they supply no historical
market facts used by the estimator. The image preparation manifest hashes its
runtime files and official pure-Python wheels. The registry manifest digest
pins the complete built image including base OS and dependency layers.

## Data, fitting, selection and calibration

Only panel observations dated at or before the mounted card's as-of are used.
No data is downloaded at evaluation time, and no other unit is inspected.
There is no GBDT/Spectra selector. All available contiguous increments up to
the mounted unit's cutoff fit each Spectra bank; context is 126 daily steps or 20 monthly steps. Joint
cards estimate standardized PCA loadings from aligned increments in the same
prefix and synthesize independent latent Spectra paths; this is an explicit
extension, not the authors' multivariate scattering operator. Optional joint PSMC
conditions whole paths across all requested assets when enabled; it is disabled
in the current release. Short isolated transfer
anchors use the latest sufficiently long earlier contiguous segment for fitting
and context, and the current supplied level for anchoring. No return bridges
the gap; no missing recent path is filled in. The stale regime is a material
forecast-quality limitation, explicitly disclosed in each rationale.

When PSMC is enabled, kernel bandwidth uses generated past distances only,
targeting ESS 256 among 512 neighbors. No outcome labels are read during an
invocation's fitting or forecasting. Retrospective fixed-setting comparisons
are disclosed in the release sections above. Fixed settings
were explored on completed historical windows within supplied public prefixes;
those pilot results are research only, not official leaderboard scores or
independent final-test evidence. Task text is not used in this version.
Input file hashes, training dates, anchors, estimates and conditioning diagnostics
are included in each non-scored `forecast_rationale.md`.

## Optional historical target weighting

`--ss-target-half-life H` (research and submission CLIs) or
`QFBENCH_SS_TARGET_HALF_LIFE=H` (submission profile) selects normalized
exponential weights `2^(-age/H)` for historical target statistics. The parameter
is measured in input observations: business observations on daily panels and
months on monthly panels. `None` is the uniform default; an explicit CLI `none`
also overrides an environment setting.

The weights pool historical wavelet energies used for scale normalization and
the historical Phi1–Phi4 statistics through the pinned core's estimator API.
Wavelet filtering precedes weighting; input values are not multiplied by a
decay function. Generated bank statistics retain uniform pooling and are matched
to these historical targets. Prefix standardization, PCA, synthesis path length
and PSMC remain separately configured. The operator is own numerical code in
`scattering_generator.py`, with no modifications to vendored source files,
learned checkpoints, additional observations or future-label access.

This is a disclosed extension of the authors' uniform estimator, not a claim
that their published experiments used decay. Diagnostics record the half-life,
age unit, effective historical observation count and pooling of synthetic
statistics. Configuration-specific identities prevent forecasts from different
half-lives sharing a cache key in the accelerated paired test. Existing frozen
results are preserved. On 2026-10-11 the full-history J=5, half-life=252,
PSMC-context=126 configuration completed the same 75-card paired evaluation:
local composite error 1.14872 versus 1.24171 for the saved unweighted automatic-J
full-history run. All 126 component fits reached tolerance 0.006. Runtime was
894.31 versus 2619.79 seconds with the same 16-slot CPU budget and library
versions. J and weighting changed together; their individual effects are not
identified by this experiment. These are repeated public retrospective tests,
not official leaderboard results or independent final-test evidence.

A subsequent paired run on 2026-10-11 fixed J=5 and disabled historical time
weighting, with all numerical sources and other profile fields unchanged from
the weighted J=5 run. All 75 cards completed; all 126 component fits reached
tolerance. Its local mean composite error was 1.25757 (holdout 1.19410), and
wall time was 1014.36 seconds. Thus changing J alone, with uniform weights,
changed mean error by +1.28% and holdout error by -2.50% versus automatic J.
Adding half-life=252 weights at J=5 changed mean error by -8.66% and holdout
error by -14.78%. Weighted forecasts improved 34/75 cards and 12/19 holdout
cards; improvements in large errors have a substantial influence on the mean.
Mean iterations were 110.13 unweighted and 112.81 weighted; timing differences
between these two runs are subject to host load and filesystem variation.
The report postprocessor reads only completed runs, verifies paired outcomes
and M0, and leaves the numerical forecaster and frozen evaluator unchanged.

A fourth paired run on 2026-10-11 retained half-life=252 and changed only
the J configuration from fixed 5 to automatic. The same full historical inputs,
numerical sources, libraries, seeds, 16 CPU slots and 126-observation PSMC
context were used. All 75 forecasts completed without failures. Actual scale
counts were 124 components at J=7 and two at J=6. Local composite error was
1.17531 versus 1.14872 for weighted J=5 (+2.32%); holdout error was 1.15144
versus 1.01767 (+13.14%). Wall time was 3005.45 versus 894.31 seconds (3.36x).
The tolerance was attained by 8/126 components versus 126/126; mean iterations
were 198.10 versus 112.81. J changes the constraint count and the existing
bank edge exclusion. This compares the configured pipeline at 200 iterations,
not the two scale choices at equal achieved synthesis accuracy.

Without PSMC, mean generator error decreased from 1.12199 to 1.09221 while
holdout error increased from 0.99443 to 1.02580. Consequently the overall
deterioration concerns the final conditional pipeline. Adding weights at
automatic J improved overall error by 5.35% and holdout error by 5.99% versus
automatic J with uniform historical pooling. The four-run postprocessor checks
completion before reading scores and preserves earlier reports. Post-run
verification confirmed all 12 recorded numerical/evaluation source hashes,
all 18 pinned vendored source hashes, and the original/comparison report hashes.
These remain repeated public retrospective evaluations, not official scores.

## CPU execution

The 2026-10-11 test_2 image sets QFBENCH_SS_MAX_SCALES=5 and
QFBENCH_SS_TARGET_HALF_LIFE=252. Automatic J is capped using only the mounted
card's usable prefix length and the synthesis bank length; an explicit fixed-J
request retains strict validation and takes precedence. This chooses J=5 on
102/104 public input directories and J=4 on the short monthly NFP practice card
and the worked exemplar. The separate 75-card accuracy experiment remains J=5
on every card. The cap packages code, not forecasts, fitted banks or unit answers.
All source/input selection and scale-policy decisions appear in the rationale.

Full contiguous available history is used for fitting. Daily PSMC keeps 126
observations, and monthly PSMC keeps the existing 20 observations. The half-life
is 252 input observations in both cases: daily trading observations or monthly
observations respectively. This release introduces no House calls, language
model weights, static calibration coefficients or new learned artifact. It
retains the original 200-iteration approximation policy and seeds.

The optional `--ss-scales J` / `QFBENCH_SS_SCALES=J` fixes the number of
Battle-Lemarie scales in both historical and synthetic transforms. The default
remains the existing length-based automatic selection. Requested scales must
fit both signal lengths; unsupported requests fail rather than being clamped.
This is own configuration code around the unchanged vendored core. Changing
J invalidates a fitted bank and is recorded in diagnostics and cache identity.

`scattering_cpu.py` supplies execution adapters around the unchanged vendored
core: it removes identity channel and whole-batch gathers while retaining the
same scale selection, reductions and gradients. A bounded thread pool prepares
independent PCA-component banks with one native thread per component. The pool
is capped by the card's CPU grant, process affinity, Linux container CPU quota
and 16 workers. Nested multiprocessing evaluation workers use one component
worker unless the outer test scheduler grants an explicit CPU allocation; the
accelerated test scheduler reserves at most 16 slots across all concurrent
cards. A grant is local to one invocation and is reset on exit, including
failures. Component seeds and candidate sampling order are unchanged; entire
joint paths are still conditioned and resampled together.

The pool shares imported numerical code, uses no additional process, external
service, IPC arrays, compiler/JIT, temporary data files or packaged fit results.
Only the mounted unit prefix is used. Errors and interruptions cancel pending
components; active optimizations stop at their next objective evaluation.
Requested/effective workers and bank preparation timing are disclosed in
rationale diagnostics. Use `--ss-workers 1` or `QFBENCH_SS_WORKERS=1` to select
sequential execution. GPU remains disabled in this image.

For predictable CPU runtime, the submission profile bounds each synthesis at 200
L-BFGS-B iterations, with target relative error 0.006. Finite approximate banks
are retained at the iteration limit, with actual per-path matching errors and
convergence flags disclosed in the rationale. The research core's default remains
strict (600 iterations, failure if tolerance is not reached). No other generator
is substituted; an approximate match is a material statistical limitation.

The software reads only the task's mounted input for forecasting. Dependency
imports read packaged code; optional caches use `/tmp`. Three required deliverables
use `/output`. Runtime seed is honored; CPU is used even when a GPU is granted.
