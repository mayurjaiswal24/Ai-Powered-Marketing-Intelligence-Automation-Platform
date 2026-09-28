"""Weekly forecast (U6): spend, leads, conversions and revenue for the next 4-8 weeks.

Honest and simple on purpose (pandas + numpy only, fully deterministic):
  - History = weekly totals of full weeks only (an incomplete first or last week is left out, as
    on the trend charts). At least FORECAST_MIN_WEEKS (26) full weeks are needed.
  - Four candidate methods:
      last value                       next weeks = the latest week
      4-week average                   next weeks = mean of the last 4 weeks
      4-week average + 12-week trend   the 4-week average plus the weekly slope of a straight
                                       line fitted to the last 12 weeks
      Holt's linear smoothing          level + trend smoothing; alpha/beta picked from a small
                                       grid by the lowest one-week-ahead squared error
  - Rolling-origin backtest over the last 12 weeks: pretend it is each of those weeks, forecast
    the following weeks with only the data known then, and compare with what really happened.
    The method with the lowest WAPE (sum of absolute errors / sum of actuals) is chosen per metric.
  - Range = 10th-90th percentile of the chosen method's past relative errors at each horizon, from
    every past week with at least 12 weeks of history before it (the last 12 weeks alone give too
    few errors: on the sample only about 4 in 10 real weeks then fell inside the range). About 8
    in 10 future weeks should fall inside it if the future behaves like the past.
Limits: one year of history cannot capture yearly seasonality, and incidents (tracking outages,
budget typos) are not predicted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analytics.common import channel_key
from analytics.kpis import KPI_REGISTRY, safe_divide
from config import settings as cfg

FORECAST_METRICS = ["spend", "leads", "conversions", "revenue"]
METHODS = {"last_value": "Last week's value", "ma4": "4-week average",
           "ma4_trend": "4-week average + 12-week trend", "holt": "Holt's linear smoothing"}
MA_WEEKS, TREND_WEEKS = 4, 12
HOLT_ALPHAS = (0.2, 0.4, 0.6, 0.8)
HOLT_BETAS = (0.05, 0.1, 0.2, 0.3)
LIMITS_NOTE = ("One year of data cannot capture yearly seasonality (for example a yearly admissions "
               "peak), and sudden incidents such as tracking outages or budget typos are not predicted.")


# ---------------------------------------------------------------------------------------------
# Weekly history
# ---------------------------------------------------------------------------------------------

def weekly_history(df: pd.DataFrame, channel: str | None = None) -> pd.DataFrame:
    """Weekly totals (week start, one column per metric) of full weeks only; weeks with no rows
    inside the range count as zero (nothing happened). `channel` limits the rows first."""
    if df is None or df.empty or "date" not in df or df["date"].notna().sum() == 0:
        return pd.DataFrame()
    rows = df[df["date"].notna()]
    if channel:
        key = channel_key(rows)
        if key is None:
            return pd.DataFrame()
        rows = rows[rows[key].astype(str) == str(channel)]
        if rows.empty:
            return pd.DataFrame()
    dates = pd.to_datetime(rows["date"]).dt.normalize()
    all_dates = pd.to_datetime(df["date"].dropna()).dt.normalize()
    week = dates - pd.to_timedelta(dates.dt.weekday, unit="D")          # Monday, as in the data's week_start
    metrics = [m for m in FORECAST_METRICS if m in rows and rows[m].notna().any()]
    if not metrics:
        return pd.DataFrame()
    sums = rows[metrics].apply(pd.to_numeric, errors="coerce").groupby(week).sum(min_count=1)
    # Days of data per week come from the whole file, so a channel with a quiet day is not "partial".
    all_weeks = all_dates - pd.to_timedelta(all_dates.dt.weekday, unit="D")
    days = all_dates.groupby(all_weeks).nunique()
    weeks = pd.date_range(days.index.min(), days.index.max(), freq="7D")
    out = sums.reindex(weeks).fillna(0.0)
    out["days"] = days.reindex(weeks).fillna(0).astype(int)
    # Full weeks only: an incomplete first or last week would look like a collapse.
    if len(out) and out["days"].iloc[0] < 7:
        out = out.iloc[1:]
    if len(out) and out["days"].iloc[-1] < 7:
        out = out.iloc[:-1]
    out.index.name = "period"
    return out.reset_index()


# ---------------------------------------------------------------------------------------------
# Methods: each takes the history (numpy array) and returns `horizon` forecasts (never below 0)
# ---------------------------------------------------------------------------------------------

def _last_value(y: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(y[-1], horizon)


def _ma4(y: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(y[-MA_WEEKS:].mean(), horizon)


def _ma4_trend(y: np.ndarray, horizon: int) -> np.ndarray:
    """The 4-week average is centred 1.5 weeks before the latest week, so the forecast h weeks
    ahead is the average plus (h + 1.5) weekly steps of the 12-week trend."""
    recent = y[-TREND_WEEKS:]
    slope = np.polyfit(np.arange(len(recent)), recent, 1)[0] if len(recent) >= 2 else 0.0
    level = y[-MA_WEEKS:].mean()
    return level + slope * (np.arange(1, horizon + 1) + (min(MA_WEEKS, len(y)) - 1) / 2)


def _holt_fit(y: np.ndarray, alpha: float, beta: float) -> tuple[float, float, float]:
    level, trend, sse = y[0], (y[1] - y[0]) if len(y) > 1 else 0.0, 0.0
    for value in y[1:]:
        predicted = level + trend
        sse += (value - predicted) ** 2
        new_level = alpha * value + (1 - alpha) * predicted
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
    return level, trend, sse


def _holt(y: np.ndarray, horizon: int) -> np.ndarray:
    best = None
    for a in HOLT_ALPHAS:                     # fixed order + strict "<": always the same choice
        for b in HOLT_BETAS:
            level, trend, sse = _holt_fit(y, a, b)
            if best is None or sse < best[2]:
                best = (level, trend, sse)
    return best[0] + best[1] * np.arange(1, horizon + 1)


_METHOD_FUNCS = {"last_value": _last_value, "ma4": _ma4, "ma4_trend": _ma4_trend, "holt": _holt}


def predict(y, method: str, horizon: int) -> np.ndarray:
    return np.clip(_METHOD_FUNCS[method](np.asarray(y, dtype=float), horizon), 0.0, None)


# ---------------------------------------------------------------------------------------------
# Backtest, method choice and range
# ---------------------------------------------------------------------------------------------

def backtest(y, method: str, horizon: int, weeks: int | None = None) -> pd.DataFrame:
    """Rolling origin: for each of the last `weeks` weeks as a starting point, forecast up to
    `horizon` weeks with only the earlier data and record actual vs forecast per horizon."""
    weeks = weeks or cfg.FORECAST_BACKTEST_WEEKS
    y = np.asarray(y, dtype=float)
    n, rows = len(y), []
    for origin in range(max(MA_WEEKS, n - weeks), n):
        steps = min(horizon, n - origin)
        fc = predict(y[:origin], method, steps)
        for h in range(steps):
            rows.append((h + 1, y[origin + h], fc[h]))
    return pd.DataFrame(rows, columns=["horizon", "actual", "forecast"])


def wape(bt: pd.DataFrame) -> float | None:
    """Weighted absolute percentage error: sum |actual - forecast| / sum |actual| x 100."""
    if bt is None or bt.empty:
        return None
    return safe_divide((bt["actual"] - bt["forecast"]).abs().sum(), bt["actual"].abs().sum(), 100.0)


def error_band(bt: pd.DataFrame, horizon: int) -> list[tuple[float, float]]:
    """(low, high) relative error per horizon: 10th and 90th percentile of (actual - forecast) /
    forecast. A horizon with fewer than 3 past errors uses all horizons together."""
    lo_p, hi_p = cfg.FORECAST_RANGE_PERCENTILES
    rel = bt[bt["forecast"] > 0].assign(rel=lambda t: (t["actual"] - t["forecast"]) / t["forecast"])
    pooled = rel["rel"].to_numpy()
    out = []
    for h in range(1, horizon + 1):
        r = rel.loc[rel["horizon"] == h, "rel"].to_numpy()
        r = r if len(r) >= 3 else pooled
        out.append((float(np.percentile(r, lo_p)), float(np.percentile(r, hi_p))) if len(r) else (0.0, 0.0))
    return out


@dataclass
class MetricForecast:
    metric: str
    method: str
    wape: float | None                      # chosen method's backtest error, %
    wapes: dict[str, float | None]          # every method's backtest error, %
    history: pd.DataFrame                   # period, actual
    forecast: pd.DataFrame                  # period, horizon, forecast, low, high

    @property
    def method_label(self) -> str:
        return METHODS[self.method]

    @property
    def accuracy_pct(self) -> float | None:
        return None if self.wape is None else max(0.0, 100.0 - self.wape)


def forecast_metric(history: pd.DataFrame, metric: str, horizon: int) -> MetricForecast:
    y = history[metric].astype(float).to_numpy()
    wapes = {m: wape(backtest(y, m, horizon)) for m in METHODS}
    scored = [(w, i, m) for i, (m, w) in enumerate(wapes.items()) if w is not None]
    method = min(scored)[2] if scored else "last_value"      # ties -> the simpler method (list order)
    point = predict(y, method, horizon)
    band = error_band(backtest(y, method, horizon, weeks=len(y) - TREND_WEEKS), horizon)
    last = pd.Timestamp(history["period"].iloc[-1])
    periods = [last + pd.Timedelta(weeks=h) for h in range(1, horizon + 1)]
    low = [max(0.0, min(p, p * (1 + lo))) for p, (lo, _) in zip(point, band)]
    high = [max(p, p * (1 + hi)) for p, (_, hi) in zip(point, band)]
    fc = pd.DataFrame({"period": periods, "horizon": range(1, horizon + 1), "forecast": point,
                       "low": low, "high": high})
    return MetricForecast(metric, method, wapes[method], wapes,
                          history[["period", metric]].rename(columns={metric: "actual"}), fc)


@dataclass
class ForecastResult:
    available: bool
    reason: str = ""
    channel: str | None = None
    horizon: int = 8
    weeks: int = 0                           # full weeks of history used
    metrics: dict[str, MetricForecast] = field(default_factory=dict)


def forecast_analysis(df: pd.DataFrame, horizon: int | None = None, channel: str | None = None) -> ForecastResult:
    """Forecast every available metric (overall, or for one channel)."""
    horizon = int(horizon or cfg.FORECAST_HORIZON_WEEKS)
    horizon = max(1, min(horizon, cfg.FORECAST_MAX_HORIZON_WEEKS))
    history = weekly_history(df, channel)
    if history.empty:
        return ForecastResult(False, "The forecast needs a date column and at least one of spend, leads, "
                                     "conversions or revenue.", channel, horizon)
    weeks = len(history)
    if weeks < cfg.FORECAST_MIN_WEEKS:
        return ForecastResult(False, f"The forecast needs at least {cfg.FORECAST_MIN_WEEKS} full weeks of data; "
                                     f"this data has {weeks}.", channel, horizon, weeks)
    metrics = {m: forecast_metric(history, m, horizon)
               for m in FORECAST_METRICS if m in history}
    return ForecastResult(True, "", channel, horizon, weeks, metrics)


def range_coverage(df: pd.DataFrame, horizon: int | None = None, channel: str | None = None,
                   weeks: int | None = None) -> float | None:
    """Honesty check: pretend it is each of the last `weeks` weeks, forecast the following weeks
    with only the data known then (method, forecast and range all refitted) and return the share
    (%) of the real weeks that fell inside the shaded range, all metrics together."""
    horizon = int(horizon or cfg.FORECAST_HORIZON_WEEKS)
    weeks = int(weeks or cfg.FORECAST_COVERAGE_WEEKS)
    history = weekly_history(df, channel)
    n = len(history)
    if n < cfg.FORECAST_MIN_WEEKS or n - weeks < TREND_WEEKS + MA_WEEKS:    # too little to replay
        return None
    inside = total = 0
    for m in [c for c in FORECAST_METRICS if c in history]:
        actual_all = history[m].to_numpy(dtype=float)
        for origin in range(n - weeks, n):
            steps = min(horizon, n - origin)
            fc = forecast_metric(history.iloc[:origin], m, steps).forecast
            actual = actual_all[origin:origin + steps]
            inside += int(((actual >= fc["low"].to_numpy() - 1e-9) & (actual <= fc["high"].to_numpy() + 1e-9)).sum())
            total += steps
    return safe_divide(inside, total, 100.0)


# ---------------------------------------------------------------------------------------------
# Plain English and display tables
# ---------------------------------------------------------------------------------------------

def accuracy_sentence(mf: MetricForecast) -> str:
    from utils.formatting import format_value, title_case
    label = title_case(KPI_REGISTRY[mf.metric].label)
    if mf.wape is None:
        return f"{label}: no activity in the last 12 weeks, so accuracy cannot be measured."
    return (f"{label}: {mf.method_label} was the most accurate of {len(METHODS)} methods tested on the last "
            f"{cfg.FORECAST_BACKTEST_WEEKS} weeks; typical error {format_value(mf.wape, 'percent')} of the "
            "actual weekly value.")


def summary_table(result: ForecastResult) -> pd.DataFrame:
    """One row per metric: method, typical error, next-N-weeks total and the average week's range."""
    from utils.formatting import format_value, title_case
    if result is None or not result.available:
        return pd.DataFrame()
    rows = []
    for m, mf in result.metrics.items():
        fmt = KPI_REGISTRY[m].fmt
        fc = mf.forecast
        rows.append({"Metric": title_case(KPI_REGISTRY[m].label), "Method": mf.method_label,
                     "Typical Error": format_value(mf.wape, "percent"),
                     f"Next {result.horizon} Weeks": format_value(fc["forecast"].sum(), fmt, compact=True),
                     "Weekly Range": f"{format_value(fc['low'].mean(), fmt, compact=True)} – "
                                     f"{format_value(fc['high'].mean(), fmt, compact=True)}"})
    return pd.DataFrame(rows)


def weekly_table(mf: MetricForecast) -> pd.DataFrame:
    from utils.formatting import format_date, format_value
    fmt = KPI_REGISTRY[mf.metric].fmt
    fc = mf.forecast
    return pd.DataFrame({"Week Starting": [format_date(p) for p in fc["period"]],
                         "Forecast": [format_value(v, fmt, compact=fmt == "money") for v in fc["forecast"]],
                         "Low (10th Percentile)": [format_value(v, fmt, compact=fmt == "money") for v in fc["low"]],
                         "High (90th Percentile)": [format_value(v, fmt, compact=fmt == "money") for v in fc["high"]]})


__all__ = ["ForecastResult", "MetricForecast", "forecast_analysis", "forecast_metric", "weekly_history",
           "backtest", "wape", "predict", "range_coverage", "METHODS", "FORECAST_METRICS", "LIMITS_NOTE"]
