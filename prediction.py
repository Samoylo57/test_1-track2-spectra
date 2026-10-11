"""Run generator + PSMC forecasts and single-series historical holdout tests.

Forecast mode writes the competition's three artifacts. Backtest mode uses only
older observations inside one supplied panel, with a fresh fit at every cutoff.
The core generator and PSMC modules have no knowledge of cards or file formats.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import tomllib
from typing import Any

import numpy as np
import pandas as pd
from qfbench2_common.scoring.crps import crps_ensemble
from qfbench2_track_forecasting.horizons import monthly_horizon_steps
from qfbench2_track_forecasting.limits import ParseLimits
from qfbench2_track_forecasting.targets import log_return_steps

from generators import GENERATOR_NAMES, PathGenerator, make_generator
from psmc import PSMC

BASE = Path(__file__).resolve().parent
KIT = BASE.parent / "track2-pilot-kit"


@dataclass(frozen=True)
class ForecastConfig:
    generator: str = "stationary-bootstrap"
    student_df: float = 5.0
    training_window: int | None = 300  # None uses the entire contiguous input prefix.
    context_length: int = 20
    block_length: float = 20.0
    n_candidates: int = 20_000
    top_k: int = 512
    ess_fraction: float = 0.5
    bandwidth: float | None = None
    embedding: str = "identity"
    foveal_alpha: float = 1.15
    foveal_beta: float = .9
    ss_bank_size: int = 64
    ss_bank_length: int = 1024
    ss_max_path_length: int = 512
    ss_batch_size: int = 64
    ss_iterations: int = 600
    ss_tolerance: float = .006
    ss_seed: int = 12345
    ss_device: str = "cpu"
    ss_domain: str = "increments"
    ss_strict_tolerance: bool = True
    ss_cpu_fastpath: bool = True
    ss_workers: int = 1
    ss_target_half_life: float | None = None
    ss_scales: int | None = None
    n_draws: int = 1000
    seed: int = 42
    level_transform: str = "difference"
    use_psmc: bool = True

    def __post_init__(self) -> None:
        if self.context_length < 2 or (self.training_window is not None and
                self.training_window < max(30, self.context_length)):
            raise ValueError("training_window must be >= max(30, context_length), context >= 2")
        if self.n_candidates < self.top_k or self.top_k < 1 or self.n_draws < 2:
            raise ValueError("require n_candidates >= top_k >= 1 and n_draws >= 2")
        if self.level_transform not in {"difference", "log"}:
            raise ValueError("level_transform must be difference or log")
        if not isinstance(self.ss_workers, int) or not 1 <= self.ss_workers <= 16:
            raise ValueError("ss_workers must be an integer in [1,16]")
        if self.ss_target_half_life is not None and (isinstance(self.ss_target_half_life, (bool, np.bool_)) or
                not np.isfinite(self.ss_target_half_life) or self.ss_target_half_life <= 0):
            raise ValueError('ss_target_half_life must be finite and positive, or None')
        if self.ss_scales is not None and (isinstance(self.ss_scales, (bool, np.bool_)) or
                not isinstance(self.ss_scales, (int, np.integer)) or self.ss_scales < 1):
            raise ValueError('ss_scales must be a positive integer or None')
        self.create_generator()
        self.create_psmc()

    def create_generator(self) -> PathGenerator:
        return make_generator(self.generator, self.block_length, student_df=self.student_df,
            scattering_options={"bank_size": self.ss_bank_size, "bank_length": self.ss_bank_length,
                "max_path_length": self.ss_max_path_length, "batch_size": self.ss_batch_size,
                "max_iterations": self.ss_iterations, "tolerance": self.ss_tolerance,
                "seed": self.ss_seed, "device": self.ss_device, "domain": self.ss_domain,
                "strict_tolerance": self.ss_strict_tolerance, "cpu_fastpath": self.ss_cpu_fastpath,
                "target_half_life": self.ss_target_half_life, "scales": self.ss_scales})

    def create_psmc(self) -> PSMC:
        return PSMC(self.top_k, self.ess_fraction, self.bandwidth, self.embedding,
                    self.foveal_alpha, self.foveal_beta)


@dataclass(frozen=True)
class Prediction:
    samples: np.ndarray
    unconditional_samples: np.ndarray
    diagnostics: dict[str, Any]


def historical_steps(
    series: pd.Series, target_type: str, frequency: str, level_transform: str
) -> pd.Series:
    """Transform the latest uninterrupted history segment; never bridge a gap."""
    if len(series) < 3 or not np.isfinite(series.to_numpy()).all():
        raise ValueError("need finite history with at least three observations")
    dates = pd.DatetimeIndex(series.index)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("history dates must be unique and increasing")
    if frequency == "monthly":
        periods = dates.to_period("M")
        if periods.has_duplicates or np.median(np.diff(periods.asi8)) != 1:
            raise ValueError("monthly history must have unique monthly observations")
        gaps = np.r_[False, np.diff(periods.asi8) != 1]
    else:
        days = np.diff(dates.asi8) / pd.Timedelta(days=1).value
        gaps = np.r_[False, days > max(5.0, 10.0 * float(np.median(days)))]
    if gaps.any():
        series = series.iloc[int(np.flatnonzero(gaps)[-1]) :]
    if target_type == "log_return":
        if frequency == "monthly":
            raise ValueError("monthly log-return targets are outside this baseline")
        return pd.Series(log_return_steps(series.to_numpy()), index=series.index)
    if target_type != "level":
        raise ValueError(f"unsupported target type: {target_type}")
    if level_transform == "log":
        if (series <= 0).any():
            raise ValueError("log level transformation requires strictly positive levels")
        series = np.log(series)
    return series.diff().iloc[1:]


def forecast_history(
    series: pd.Series | pd.DataFrame,
    horizon_steps: list[int],
    *,
    target_type: str = "level",
    frequency: str = "daily",
    config: ForecastConfig = ForecastConfig(),
    fitted_generator: PathGenerator | None = None,
) -> Prediction:
    """Generate past+future, shadow the recent past, and extract joint horizons."""
    if not horizon_steps or any(type(h) is not int or h <= 0 for h in horizon_steps):
        raise ValueError("horizon_steps must be positive integer sampling steps")
    joint = isinstance(series, pd.DataFrame)
    if joint and config.generator not in {"student-t", "scattering-spectra"}:
        raise ValueError("only student-t and scattering-spectra support joint targets")
    if joint and frequency == "monthly":
        raise ValueError("joint monthly targets are outside this version")
    increments = (
        pd.DataFrame({asset: historical_steps(series[asset], target_type, frequency, config.level_transform)
                      for asset in series.columns}).dropna()
        if joint else historical_steps(series, target_type, frequency, config.level_transform)
    )
    training = increments if config.training_window is None else increments.iloc[-config.training_window :]
    if len(training) < max(30, config.context_length):
        raise ValueError(
            f"only {len(training)} recent contiguous increments; need "
            f"{max(30, config.context_length)}. A sparse transfer anchor cannot supply a PSMC past."
        )
    observed = training.to_numpy()[-config.context_length :]
    generation_seed, shadow_seed, control_seed = np.random.SeedSequence(config.seed).spawn(3)
    generator = fitted_generator or config.create_generator()
    if joint and config.generator == "scattering-spectra" and fitted_generator is None:
        from joint_scattering import JointScatteringGenerator
        from dataclasses import fields
        options = {f.name: getattr(generator, f.name) for f in fields(generator) if f.init}
        generator = JointScatteringGenerator(options, workers=config.ss_workers)
    generator.fit(training.to_numpy())
    candidates = generator.sample(
        config.n_candidates,
        config.context_length + max(horizon_steps),
        np.random.default_rng(generation_seed),
    )
    control_indices = np.random.default_rng(control_seed).integers(
        config.n_candidates, size=config.n_draws
    )
    control_future = candidates[control_indices, config.context_length :]
    diagnostics: dict[str, Any] = {
        "config": asdict(config),
        "training_rows": len(training),
        "training_first_date": str(training.index[0].date()),
        "training_last_date": str(training.index[-1].date()),
        "context_first_date": str(training.index[-config.context_length].date()),
        "step_mean": training.mean().to_dict() if joint else float(training.mean()),
        "step_sd": training.std().to_dict() if joint else float(training.std()),
        "horizon_steps": horizon_steps,
        "frequency": frequency,
        "target_type": target_type,
        "text_used": False,
        "generator_parameters": generator.diagnostics() if hasattr(generator, "diagnostics") else {},
    }
    if config.use_psmc:
        result = config.create_psmc().condition(
            observed, candidates, config.n_draws, np.random.default_rng(shadow_seed)
        )
        future = result.future_steps
        diagnostics["psmc"] = result.diagnostics()
    else:
        future = control_future
        diagnostics["psmc"] = None
    anchor = (np.zeros(len(series.columns)) if target_type == "log_return" else series.iloc[-1].to_numpy()) if joint else (
        0.0 if target_type == "log_return" else float(series.iloc[-1])
    )

    def decode(paths: np.ndarray) -> np.ndarray:
        totals = paths.cumsum(axis=1)[:, np.asarray(horizon_steps) - 1]
        if target_type == "log_return":
            return totals
        if config.level_transform == "log":
            return anchor * np.exp(totals)
        return anchor + totals

    samples, control = decode(future), decode(control_future)
    if not np.isfinite(samples).all() or not np.isfinite(control).all():
        raise ValueError("forecast contains non-finite values; revise the generator or transformation")
    diagnostics["anchor"] = anchor.tolist() if joint else anchor
    if joint:
        diagnostics["asset_order"] = list(series.columns)
    diagnostics["quantiles"] = np.quantile(samples, [0.05, 0.5, 0.95], axis=0).tolist()
    return Prediction(samples, control, diagnostics)


def public_unit(unit_id: str) -> Path:
    """Locate only this requested unit in the downloaded public repository."""
    if Path(unit_id).name != unit_id or "\\" in unit_id or unit_id in {".", ".."}:
        raise ValueError("unit_id must be a directory name")
    repositories = sorted((KIT / "sources").glob("track2-forecasting-public-*"))
    if len(repositories) != 1:
        raise ValueError("need exactly one downloaded Track 2 repository under track2-pilot-kit/sources")
    unit = repositories[0] / "units" / unit_id
    if not (unit / "card.toml").is_file():
        raise ValueError(f"public unit not found: {unit_id}")
    return unit


def load_task(args: argparse.Namespace, *, allow_joint: bool = False) -> tuple[dict[str, Any], pd.Series | pd.DataFrame, Path, str, str]:
    if args.unit_id:
        if args.panels or args.card:
            raise ValueError("use --unit-id or explicit --panels/--card inputs")
        root = public_unit(args.unit_id)
        panels = root / "panels" if (root / "panels").is_dir() else root
        card_path = root / "card.toml"
    else:
        if args.panels is None:
            raise ValueError("supply --unit-id or --panels")
        panels = args.panels.resolve()
        candidates = [panels / "card.toml", panels.parent / "card.toml"]
        card_path = args.card or next((p for p in candidates if p.is_file()), None)
        if card_path is None:
            raise ValueError("card.toml not found; supply --card")
    card = tomllib.loads(card_path.read_text(encoding="utf-8"))
    assets = card["targets"]["asset_ids"]
    if not assets or (len(assets) != 1 and not allow_joint):
        raise ValueError(f"this baseline supports one target series; this task requests {len(assets)}")
    asof = args.asof or card["provenance"]["data_cutoff"]
    date.fromisoformat(asof)
    if asof != card["provenance"]["data_cutoff"]:
        raise ValueError("--asof must match the task cutoff; use backtest for earlier historical cutoffs")
    files = sorted(panels.glob("*.parquet"))
    if not files and panels.name == "panels":
        files = sorted(panels.parent.glob("*.parquet"))
    matches = {asset: [] for asset in assets}
    for path in files:
        frame = pd.read_parquet(path)
        col = next((c for c in ("asset", "asset_id") if c in frame.columns), None)
        if col is None:
            continue
        rows = frame.loc[frame[col].astype(str).isin(assets)].copy()
        if rows.empty:
            continue
        rows["date"] = pd.to_datetime(rows["date"], errors="raise")
        rows = rows.loc[rows["date"] <= pd.Timestamp(asof)].sort_values("date")
        for asset, group in rows.groupby(col):
            matches[str(asset)].append((path, group.set_index("date")["value"].astype(float)))
    for asset, found in matches.items():
        if len(found) != 1:
            raise ValueError(f"expected one panel containing {asset} before the cutoff, found {len(found)}")
    source = matches[assets[0]][0][0]
    if len(assets) == 1:
        series = matches[assets[0]][0][1]
    else:
        series = pd.concat({asset: matches[asset][0][1] for asset in assets}, axis=1, join="inner").dropna()
        if series.empty:
            raise ValueError("joint targets have no aligned complete observations")
    frequency = card["targets"].get(
        "target_frequency", card.get("metadata", {}).get("target_frequency", "daily")
    )
    # Older cards call daily month-ahead targets 'monthly'. Detect the actual
    # daily cadence, but refuse a conflict with explicit monthly-period metadata.
    spec_path = card_path.parent / "forecast_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    if frequency == "monthly" and len(series) >= 30:
        gaps = np.diff(series.index.asi8) / pd.Timedelta(days=1).value
        per_month = pd.Series(1, index=series.index.to_period("M")).groupby(level=0).sum()
        if np.median(gaps) <= 3 and per_month.median() >= 8:
            explicit = any(
                "observation_periods" in obj.get("targets", {})
                or any("observation_period" in q for q in obj.get("questions", []))
                for obj in (card, spec)
            )
            if explicit:
                raise ValueError("monthly target metadata conflicts with daily observations")
            frequency = "daily"
    if frequency not in {"daily", "business_daily", "monthly"}:
        raise ValueError(f"unsupported target frequency: {frequency}")
    if len(assets) > 1 and frequency == "monthly":
        raise ValueError("joint monthly targets are outside this version")
    return card, series, source, frequency, asof


def resolve_horizon_steps(
    card: dict[str, Any], series: pd.Series, source: Path, frequency: str, asof: str
) -> list[int]:
    horizons = card["targets"]["horizons"]
    if frequency != "monthly":
        return list(horizons)
    root = source.parent.parent if source.parent.name == "panels" else source.parent
    spec_path = root / "forecast_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else None
    asset = card["targets"]["asset_ids"][0]
    matrix = monthly_horizon_steps(
        [asset], horizons, {asset: str(series.index[-1].date())},
        asof=asof, card=card, forecast_spec=spec,
    )
    return [int(h) for h in matrix[0]]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def default_output_dir(unit_id: str, generator_name: str) -> Path:
    root = BASE / "outputs"
    if generator_name != "stationary-bootstrap":
        root /= generator_name
    return root / unit_id


def run_forecast(args: argparse.Namespace, config: ForecastConfig) -> None:
    card, series, source, frequency, asof = load_task(args, allow_joint=config.generator in {"student-t", "scattering-spectra"})
    floor = max(ParseLimits().min_draws, card.get("scoring", {}).get("params", {}).get("n_draws_min", 0))
    if not floor <= config.n_draws <= ParseLimits().max_draws:
        raise ValueError(f"n_draws must be in [{floor}, {ParseLimits().max_draws}]")
    steps = resolve_horizon_steps(card, series, source, frequency, asof)
    prediction = forecast_history(
        series, steps, target_type=card["targets"]["target_type"], frequency=frequency, config=config
    )
    unit_id = card["task"]["id"]
    assets = card["targets"]["asset_ids"]
    horizons = card["targets"]["horizons"]
    out = (args.out or default_output_dir(unit_id, config.generator) / "forecast.parquet").resolve()
    if out.name != "forecast.parquet":
        raise ValueError("--out must name forecast.parquet")
    report_stem = unit_id if config.generator == "stationary-bootstrap" else f"{config.generator}-{unit_id}"
    diagnostics_path = (args.report or BASE / "reports" / f"{report_stem}-diagnostics.json").resolve()
    if diagnostics_path.parent == out.parent or diagnostics_path.suffix != ".json":
        raise ValueError("put a .json diagnostics report outside the competition artifact directory")
    artifact_names = {"forecast.parquet", "forecast_meta.json", "forecast_rationale.md"}
    if out.parent.exists() and any(p.name not in artifact_names for p in out.parent.iterdir()):
        raise ValueError("output directory must contain only the three competition artifacts")
    out.parent.mkdir(parents=True, exist_ok=True)
    cube = prediction.samples.transpose(0, 2, 1) if prediction.samples.ndim == 3 else prediction.samples[:, None, :]
    pd.DataFrame({
        "draw": np.repeat(np.arange(config.n_draws, dtype=np.int64), len(assets) * len(horizons)),
        "asset": np.tile(np.repeat(assets, len(horizons)), config.n_draws),
        "horizon": np.tile(np.asarray(horizons, dtype=np.int64), config.n_draws * len(assets)),
        "value": cube.reshape(-1),
    }).to_parquet(out, index=False)
    method = f"{config.generator} {'with ' + config.embedding + ' PSMC' if config.use_psmc else 'unconditional'}; no text"
    write_json(out.parent / "forecast_meta.json", {
        "unit_id": unit_id, "asof": asof, "representation": "samples",
        "asset_ids": assets, "horizons": horizons, "n_draws": config.n_draws,
        "target": card["targets"]["target_type"],
        "rationale": {"file": "forecast_rationale.md", "method": method},
    })
    diag = prediction.diagnostics | {
        "unit_id": unit_id, "asof": asof, "asset_ids": assets, "horizon_keys": horizons,
        "panel_file": str(source.resolve()),
        "panel_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "accuracy_scored": False,
    }
    rationale = (
        f"# Statistical baseline: {unit_id}\n\n"
        f"Forecast {', '.join(assets)} as of {asof}, using {method}.\n\n"
        "The generator is fitted only on the latest contiguous segment at or before the cutoff. "
        "A candidate's past and future share one continuous generation process. "
        "Student-t increments are independent across time and use df-corrected covariance. "
        "GARCH uses simulated standardized historical residuals and a persistent variance state. "
        "Quantile GBDT uses lagged increment features, fitted quantiles and recursive inverse-CDF sampling. "
        "Bootstrap variants resample history and restart instead of wrapping at its end. "
        "Scattering Spectra uses the authors' fixed Battle-Lemarie wavelets and Phi1-Phi4 "
        "statistics, optimizing synthetic trajectory values with bounded L-BFGS-B. "
        "Its finite bank is fitted on this unit's prefix; no bundled market datasets or "
        "pretrained neural weights enter the numerical core. "
        "The selected generator and its fitted parameters are recorded below.\n\n"
        "PSMC compares the generated past increments to the observed recent increments. "
        "The foveal option first maps past increments to weighted cumulative changes at "
        "geometrically spaced lags, retaining the repeated short lags from the paper. "
        "It selects nearest pasts, weights them with a Gaussian kernel and resamples whole "
        "future continuations. Adaptive bandwidth uses only past distances and targets "
        "the configured effective sample size. No future values select paths or weights. "
        "When PSMC is disabled, candidates are sampled uniformly.\n\n"
        "All horizons in a draw come from the same continuation. Level targets add cumulative "
        "changes to the last observed level (or exponentiate log changes when requested). "
        "Factor log-return targets sum log(1+r) and start at zero. Monthly targets use explicit "
        "observation periods and retain the authored horizon keys.\n\n"
        "The frozen text corpus and LLMs are not used in this first numerical baseline. "
        "No claim of improvement over M0 is made; hidden outcomes are unavailable.\n\n"
        "Configuration and conditioning diagnostics:\n\n```json\n"
        + json.dumps(prediction.diagnostics, indent=2, allow_nan=False) + "\n```\n"
    )
    (out.parent / "forecast_rationale.md").write_text(rationale, encoding="utf-8")
    write_json(diagnostics_path, diag)
    print(f"Forecast: {out}")
    print(f"{config.n_draws} joint draws x {len(assets)} asset(s) x {len(horizons)} horizon(s); keys={horizons}, steps={steps}")
    if config.use_psmc:
        print(f"PSMC ESS: {diag['psmc']['effective_sample_size']:.1f}; report: {diagnostics_path}")


def run_backtest(args: argparse.Namespace, config: ForecastConfig) -> None:
    card, series, source, frequency, asof = load_task(args)
    horizon = args.backtest_horizon or (3 if frequency == "monthly" else 21)
    if horizon < 1 or args.windows < 1:
        raise ValueError("backtest horizon and windows must be positive")
    first = len(series) - args.windows * horizon
    if first < max(31, config.context_length + 1):
        raise ValueError("not enough history for the requested holdout windows")
    target_type = card["targets"]["target_type"]
    rows = []
    for window, cutoff in enumerate(range(first, len(series) - horizon + 1, horizon)):
        prefix = series.iloc[:cutoff]
        holdout_dates = series.index[cutoff - 1 : cutoff + horizon]
        if frequency == "monthly":
            uninterrupted = np.all(np.diff(holdout_dates.to_period("M").asi8) == 1)
        else:
            historical_gaps = np.diff(prefix.index.asi8) / pd.Timedelta(days=1).value
            holdout_gaps = np.diff(holdout_dates.asi8) / pd.Timedelta(days=1).value
            uninterrupted = np.all(holdout_gaps <= max(5.0, 10.0 * float(np.median(historical_gaps))))
        if not uninterrupted:
            raise ValueError("a holdout crosses a missing-history gap; choose a recent contiguous window")
        prediction = forecast_history(
            prefix, [horizon], target_type=target_type, frequency=frequency,
            config=replace(config, seed=config.seed + window, use_psmc=True),
        )
        actual = (
            float(log_return_steps(series.iloc[cutoff : cutoff + horizon].to_numpy()).sum())
            if target_type == "log_return" else float(series.iloc[cutoff + horizon - 1])
        )
        rows.append({
            "cutoff": str(prefix.index[-1].date()),
            "holdout_end": str(series.index[cutoff + horizon - 1].date()),
            "psmc_crps": float(crps_ensemble(prediction.samples[:, 0], actual)),
            "unconditional_crps": float(crps_ensemble(prediction.unconditional_samples[:, 0], actual)),
            "psmc_90pct_covers": bool(np.quantile(prediction.samples[:, 0], 0.05) <= actual <= np.quantile(prediction.samples[:, 0], 0.95)),
            "psmc_ess": prediction.diagnostics["psmc"]["effective_sample_size"],
            "training_last_date": prediction.diagnostics["training_last_date"],
        })
    report = {
        "unit_id": card["task"]["id"], "asset": card["targets"]["asset_ids"][0],
        "source_cutoff": asof, "horizon_in_panel_observations": horizon,
        "config": asdict(config), "windows": rows,
        "mean_psmc_crps": float(np.mean([r["psmc_crps"] for r in rows])),
        "mean_unconditional_crps": float(np.mean([r["unconditional_crps"] for r in rows])),
        "note": "Raw marginal CRPS on historical holdouts inside one supplied panel. Lower is better. "
        "This is not the competition score or an M0 comparison. Panel revisions are not reconstructed at historical cutoffs.",
        "official_score": False, "panel_file": str(source.resolve()),
    }
    report_stem = card['task']['id'] if config.generator == "stationary-bootstrap" else f"{config.generator}-{card['task']['id']}"
    out = (args.report or BASE / "reports" / f"{report_stem}-backtest.json").resolve()
    write_json(out, report)
    print(f"Historical holdouts: {len(rows)}; PSMC CRPS={report['mean_psmc_crps']:.6g}; "
          f"unconditional CRPS={report['mean_unconditional_crps']:.6g}")
    print(f"Report: {out}")


def parse_training_window(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("training window must be an integer or 'all'") from exc


def parse_target_half_life(value: str) -> float | None:
    if value.lower() == 'none':
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('target half-life must be positive or none') from exc
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('target half-life must be finite and positive or none')
    return result


def parse_scales(value: str) -> int | None:
    if value.lower() in {'auto', 'none'}:
        return None
    if not value.isascii() or not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError('Spectra scales must be a positive integer or auto')
    return int(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["forecast", "backtest"])
    parser.add_argument("--unit-id", help="downloaded public unit; Student-t also supports joint daily targets")
    parser.add_argument("--panels", type=Path)
    parser.add_argument("--card", type=Path)
    parser.add_argument("--text", type=Path, help="accepted for the competition CLI; unused in V1")
    parser.add_argument("--asof")
    parser.add_argument("--out", type=Path, help="absolute forecast.parquet destination")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--generator", choices=GENERATOR_NAMES, default="stationary-bootstrap")
    parser.add_argument("--student-df", type=float, default=5.0)
    parser.add_argument("--training-window", type=parse_training_window, default=300,
                        help="recent increment count, or 'all' for the entire contiguous input prefix")
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--block-length", type=float, default=20)
    parser.add_argument("--n-candidates", type=int, default=20_000)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--ess-fraction", type=float, default=0.5)
    parser.add_argument("--bandwidth", type=float)
    parser.add_argument("--embedding", choices=["identity", "foveal"], default="identity")
    parser.add_argument("--foveal-alpha", type=float, default=1.15)
    parser.add_argument("--foveal-beta", type=float, default=.9)
    parser.add_argument("--ss-bank-size", type=int, default=64)
    parser.add_argument("--ss-bank-length", type=int, default=1024)
    parser.add_argument("--ss-max-path-length", type=int, default=512)
    parser.add_argument("--ss-batch-size", type=int, default=64)
    parser.add_argument("--ss-iterations", type=int, default=600)
    parser.add_argument("--ss-tolerance", type=float, default=.006)
    parser.add_argument("--ss-seed", type=int, default=12345)
    parser.add_argument("--ss-device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--ss-domain", choices=["increments", "levels"], default="increments")
    parser.add_argument("--ss-strict-tolerance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ss-cpu-fastpath", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ss-workers", type=int, default=1,
                        help="CPU workers for joint Spectra components, 1..16; nested process pools use 1")
    parser.add_argument('--ss-target-half-life', type=parse_target_half_life,
                        help='historical target-statistic half-life in observations; none disables (default)')
    parser.add_argument('--ss-scales', type=parse_scales,
                        help='fixed number of Battle-Lemarie scales J, or auto (default)')
    parser.add_argument("--n-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level-transform", choices=["difference", "log"], default="difference")
    parser.add_argument("--no-psmc", action="store_true", help="unconditional control forecast")
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--backtest-horizon", type=int, help="holdout horizon in panel observations")
    args = parser.parse_args(argv)
    try:
        config = ForecastConfig(**{
            key: getattr(args, key) for key in ForecastConfig.__dataclass_fields__ if key != "use_psmc"
        }, use_psmc=not args.no_psmc)
        if args.mode == "backtest":
            run_backtest(args, config)
        else:
            run_forecast(args, config)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
