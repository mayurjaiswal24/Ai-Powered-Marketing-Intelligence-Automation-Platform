"""Anomaly detection: weeks (and days) that are genuinely unusual, not just large.

WEEKLY METHOD (per campaign, per channel and per region; metrics: spend, revenue, clicks, CPC,
CPL, lead-qualification rate, lead-to-conversion rate):

  1. Only full weeks (Mon-Sun, all 7 days present for that entity) are used, so a campaign
     starting on a Thursday does not look like a spend drop.
  2. Seasonality is removed by comparing each entity with THE REST OF THE BUSINESS in the same
     week: for spend we look at "this campaign's spend / everyone else's spend", for CPL at
     "this campaign's CPL / everyone else's CPL". January or Diwali lifts everyone, so the
     ratio stays steady; a broken landing page hits one campaign, so its ratio jumps.
     (Alternative: compare with the same week last year. Needs 2+ years of data, which a
     typical upload does not have.)
  3. Baseline = median of that ratio over the previous 8 full weeks (the median ignores the
     odd bad week). Spread = MAD (median absolute deviation), floored at 5% of the baseline.
  4. Robust score = (this week - baseline) / (1.4826 x MAD). A week is flagged only if the score
     is at least 3.5 AND the change is at least 30% AND the metric has enough volume (e.g. at
     least 20 leads a week for CPL). All thresholds live in config/settings.py.
  5. Consecutive flagged weeks for the same entity and metric are merged into one anomaly.

DAILY RULE (tracking outages): spend continues (>= 25% of normal) but clicks fall below 5% of
the previous 28 days' median. Days are grouped by platform, because an outage usually hits
every campaign that uses the same tracking tag.

Every anomaly records: metric, entity, period, baseline (expected), observed, % change, score,
direction, sentiment and a plain-English description.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analytics.common import campaign_key
from analytics.kpis import KPI_REGISTRY, ratio_parts
from config.settings import (ANOMALY_BASELINE_WEEKS, ANOMALY_HIGH_SCORE, ANOMALY_MIN_HISTORY_WEEKS,
                             ANOMALY_MIN_PCT_CHANGE, ANOMALY_MIN_VOLUME, ANOMALY_SCALE_FLOOR,
                             ANOMALY_SCORE_THRESHOLD, OUTAGE_BASELINE_DAYS, OUTAGE_CLICK_SHARE,
                             OUTAGE_MIN_BASELINE_CLICKS, OUTAGE_MIN_SPEND_SHARE)
from utils.formatting import format_date, format_value

ADDITIVE_METRICS = ["spend", "revenue", "clicks"]
RATIO_METRICS = ["cpc", "cpl", "lead_qualification_rate", "lead_to_conversion_rate"]
MAD_TO_SD = 1.4826   # makes the MAD comparable to a standard deviation for normal data

ANOMALY_COLUMNS = ["metric", "entity_type", "entity", "grain", "period_start", "period_end",
                   "baseline", "observed", "pct_change", "score", "direction", "sentiment",
                   "severity", "method", "description", "details_json"]


# ---------------------------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------------------------

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """All weekly and daily anomalies for a cleaned dataset (needs `date` and `week_start`)."""
    if "date" not in df or df["date"].isna().all():
        return pd.DataFrame(columns=ANOMALY_COLUMNS)
    data = df[df["date"].notna()]
    found: list[dict] = []
    for entity_type, column in _entity_columns(data):
        found += weekly_anomalies(data, column, entity_type)
    found += tracking_outages(data)
    out = pd.DataFrame(found, columns=ANOMALY_COLUMNS)
    if out.empty:
        return out
    return out.sort_values(["period_start", "entity_type", "entity", "metric"]).reset_index(drop=True)


def _entity_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    pairs = []
    key = campaign_key(df)
    if key:
        pairs.append(("campaign", key))
    for entity_type in ("channel", "region"):
        if entity_type in df and df[entity_type].notna().any():
            pairs.append((entity_type, entity_type))
    return pairs


# ---------------------------------------------------------------------------------------------
# Weekly robust detection
# ---------------------------------------------------------------------------------------------

def _metric_parts(df: pd.DataFrame, metric: str):
    """Row-level (numerator, denominator or None, scale) using the KPI engine's definitions."""
    if metric in ADDITIVE_METRICS:
        if metric not in df or df[metric].isna().all():
            return None
        return df[metric].astype(float), None, 1.0
    parts = ratio_parts(df, metric)
    if parts is None:
        return None
    num, den, scale, _ = parts
    return num, den, scale


# The count behind each metric. Counts have natural "small numbers" noise: with about 9
# conversions a week, 5 or 14 happen by pure chance. Spend has no such noise.
NOISE_COUNT = {"revenue": "conversions", "clicks": "clicks", "cpc": "clicks", "cpl": "leads",
               "lead_qualification_rate": "qualified_leads",
               "lead_to_conversion_rate": "conversions"}
from config.settings import ANOMALY_WINDOWS as WINDOWS  # noqa: E402  (single weeks + 3-week stretches)
from config.settings import ANOMALY_MIN_ENTITIES_FOR_COMMON_MOVE as MIN_ENTITIES_FOR_COMMON_MOVE  # noqa: E402


def _stable_composition(df: pd.DataFrame, entity: pd.Series, week: pd.Series,
                        index: pd.Index) -> pd.DataFrame:
    """For channels/regions: True where the same campaigns ran for the whole of the current week
    and the baseline weeks before it (like-for-like). A campaign starting or stopping changes
    the totals without anything being wrong, so those windows are not judged."""
    key = campaign_key(df)
    days = df.groupby([entity, df[key].astype(str), week])["date"].nunique()
    # Status per campaign-week: 0 = not running, 1 = running all 7 days, 2 = partial week.
    status = days.map(lambda d: 1 if d == 7 else 2).unstack([0, 1]).reindex(index).fillna(0)
    span = ANOMALY_BASELINE_WEEKS + 1
    rolling = status.rolling(span, min_periods=1)
    steady = (rolling.max() == rolling.min()) & (status != 2)
    return steady.T.groupby(level=0).all().T


def weekly_anomalies(df: pd.DataFrame, entity_col: str, entity_type: str) -> list[dict]:
    week = df["week_start"]
    entity = df[entity_col].astype(str)
    full = df.groupby([entity, week])["date"].nunique().unstack(0) == 7
    key = campaign_key(df)
    if entity_type != "campaign" and key is not None:
        full &= _stable_composition(df, entity, week, full.index).reindex(
            index=full.index, columns=full.columns, fill_value=False)
        # A channel/region with a single campaign would only repeat that campaign's alerts.
        n_campaigns = df.groupby(entity)[key].nunique()
        full = full[[c for c in full.columns if n_campaigns.get(c, 0) > 1]]
    if full.shape[1] == 0:
        return []

    def by_entity_week(series: pd.Series) -> pd.DataFrame:
        table = series.groupby([entity, week]).sum(min_count=1).unstack(0)
        return table.reindex(index=full.index, columns=full.columns)

    flagged: list[dict] = []
    for metric in ADDITIVE_METRICS + RATIO_METRICS:
        parts = _metric_parts(df, metric)
        if parts is None:
            continue
        num, den, scale = parts
        count_col = NOISE_COUNT.get(metric)
        tables = {"num": by_entity_week(num),
                  "den": by_entity_week(den) if den is not None else None,
                  "count": (by_entity_week(df[count_col].astype(float))
                            if count_col and count_col in df else None)}
        for window in WINDOWS:
            for item in _judge(tables, scale, full, metric, window):
                item.update(entity_type=entity_type, metric=metric)
                flagged.append(item)
    return _merge_periods(flagged)


def _baseline(table: pd.DataFrame, shift: int, func=None) -> pd.DataFrame:
    """Statistic of each entity's previous ANOMALY_BASELINE_WEEKS weeks (NaN weeks skipped),
    ending just before the window being judged."""
    rolling = table.shift(1).rolling(ANOMALY_BASELINE_WEEKS, min_periods=ANOMALY_MIN_HISTORY_WEEKS)
    stat = rolling.median() if func is None else rolling.apply(func, raw=True)
    return stat.shift(shift)


def _common_move(r: pd.DataFrame) -> pd.Series:
    """The typical relative change across entities in each period (median, so one entity's
    spike cannot drag it). 1.0 when there are too few entities to judge."""
    enough = r.notna().sum(axis=1) >= MIN_ENTITIES_FOR_COMMON_MOVE
    return r.median(axis=1).where(enough, 1.0)


def _ratio(num, den, scale):
    return num if den is None else num / den.where(den != 0) * scale


def _judge(t: dict, scale: float, full: pd.DataFrame, metric: str, window: int) -> list[dict]:
    """Judge every entity's weeks (window=1) or 3-week stretches (window=3) at once.

    relative change r = this period / the entity's own baseline (median of previous 8 weeks)
    common move  f = median of r across all entities that period (seasonality, market moves)
    adjusted     a = r / f   (1.0 = moved exactly like everyone else)
    """
    shift = window - 1
    value1 = _ratio(t["num"], t["den"], scale)
    ok1 = full & value1.notna()
    base = _baseline(value1.where(ok1), shift)

    def wsum(x):
        return None if x is None else x.rolling(window, min_periods=window).sum()

    value = _ratio(wsum(t["num"]), wsum(t["den"]), scale)
    if t["den"] is None:
        value = value / window     # totals are compared as a weekly average
    ok = (ok1.astype(float).rolling(window, min_periods=window).min() == 1) & base.notna() & (base != 0)
    volume = t["num"] if t["den"] is None else t["den"]
    ok &= _baseline(volume.where(ok1), shift) >= ANOMALY_MIN_VOLUME.get(metric, 0)

    r = (value / base).where(ok)
    common = _common_move(r)
    a = r.div(common, axis=0)

    # Robust score: distance from 1.0 in units of the entity's usual week-to-week wobble,
    # measured on the same seasonality-adjusted scale (single weeks, before this window).
    r1 = (value1 / _baseline(value1.where(ok1), 0)).where(ok1)
    a1 = r1.div(_common_move(r1), axis=0)
    mad = _baseline(a1, shift, lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))))
    spread = np.maximum(MAD_TO_SD * mad / np.sqrt(window), ANOMALY_SCALE_FLOOR)
    score = (a - 1) / spread
    if t["count"] is not None:
        # Square-root rule for counts: z = 2 x (sqrt(observed) - sqrt(expected)). Both tests
        # must agree, so a change has to beat both the usual wobble AND small-number chance.
        expected_count = _baseline(t["count"].where(ok1), shift) * window
        count_z = 2 * (np.sqrt((a * expected_count).clip(lower=0)) - np.sqrt(expected_count))
        score = np.sign(score) * np.minimum(score.abs(), count_z.abs())

    pct = (a - 1) * 100
    expected = base * common.to_numpy()[:, None]
    flag = ok & (score.abs() >= ANOMALY_SCORE_THRESHOLD) & (pct.abs() >= ANOMALY_MIN_PCT_CHANGE)
    if window > 1:
        # A 3-week stretch counts only if it is SUSTAINED: at least 2 of its 3 weeks moved the
        # same way by half the minimum change or more. Otherwise one spike week would stretch
        # into a 3-week anomaly. (2 of 3, not 3 of 3: with small counts one week can dip less
        # by chance.)
        week_pct = (a1 - 1) * 100
        half = ANOMALY_MIN_PCT_CHANGE / 2
        need = window - 1
        up_weeks = (week_pct >= half).astype(float).rolling(window, min_periods=window).sum()
        down_weeks = (week_pct <= -half).astype(float).rolling(window, min_periods=window).sum()
        flag &= ((pct > 0) & (up_weeks >= need)) | ((pct < 0) & (down_weeks >= need))
    weeks = list(value.index)
    out = []
    stacked = flag.fillna(False).astype(bool).stack()
    for w, name in stacked[stacked].index:
        start = weeks[weeks.index(w) - shift]
        out.append({"entity": name, "start": start, "end": w + pd.Timedelta(days=6),
                    "window": window, "baseline": float(expected.at[w, name]),
                    "observed": float(value.at[w, name]), "pct_change": float(pct.at[w, name]),
                    "score": float(score.at[w, name]), "common_move_pct": float((common[w] - 1) * 100)})
    return out


def _merge_periods(items: list[dict]) -> list[dict]:
    """Merge overlapping or back-to-back flagged periods (same entity, metric and direction)
    into one anomaly. The reported numbers are those of the strongest single period."""
    groups: dict[tuple, list[dict]] = {}
    for it in items:
        key = (it["entity_type"], it["entity"], it["metric"], it["pct_change"] > 0)
        groups.setdefault(key, []).append(it)
    anomalies = []
    for members in groups.values():
        members.sort(key=lambda r: r["start"])
        cluster = [members[0]]
        for it in members[1:]:
            if it["start"] <= max(c["end"] for c in cluster) + pd.Timedelta(days=1):
                cluster.append(it)
            else:
                anomalies.append(_weekly_record(cluster))
                cluster = [it]
        anomalies.append(_weekly_record(cluster))
    return anomalies


def _weekly_record(cluster: list[dict]) -> dict:
    peak = max(cluster, key=lambda r: abs(r["score"]))
    start = min(c["start"] for c in cluster)
    end = max(c["end"] for c in cluster)
    n_weeks = ((end - start).days + 1) // 7
    details = {"periods": [{"start": c["start"].strftime("%Y-%m-%d"),
                            "end": c["end"].strftime("%Y-%m-%d"), "weeks": c["window"],
                            "observed": round(c["observed"], 4), "expected": round(c["baseline"], 4),
                            "pct_change": round(c["pct_change"], 1), "score": round(c["score"], 2),
                            "typical_change_all_entities_pct": round(c["common_move_pct"], 1)}
                           for c in sorted(cluster, key=lambda c: (c["start"], c["window"]))],
               "strongest": {"start": peak["start"].strftime("%Y-%m-%d"), "weeks": peak["window"]}}
    return _record(peak["metric"], peak["entity_type"], peak["entity"], "week", start, end,
                   peak["baseline"], peak["observed"], peak["pct_change"], peak["score"],
                   "weekly_robust_score", details, n_periods=n_weeks)


# ---------------------------------------------------------------------------------------------
# Daily tracking-outage rule
# ---------------------------------------------------------------------------------------------

def tracking_outages(df: pd.DataFrame) -> list[dict]:
    key = campaign_key(df)
    if key is None or "spend" not in df or "clicks" not in df:
        return []
    group_col = "platform" if "platform" in df and df["platform"].notna().any() else (
        "channel" if "channel" in df and df["channel"].notna().any() else None)
    daily = (df.groupby([key, "date"])
               .agg(spend=("spend", "sum"), clicks=("clicks", lambda s: s.sum(min_count=1)),
                    leads=("leads", "sum") if "leads" in df else ("clicks", "size"))
               .reset_index().sort_values([key, "date"]))
    if group_col:
        owner = df.groupby(key)[group_col].agg(lambda s: s.mode().iat[0] if s.notna().any() else "(none)")
        daily["group"] = daily[key].map(owner)
    else:
        daily["group"] = daily[key]

    hits = []
    for name, part in daily.groupby(key):
        part = part.set_index("date").asfreq("D")
        base_clicks = part["clicks"].rolling(f"{OUTAGE_BASELINE_DAYS}D", min_periods=7).median().shift(1)
        base_spend = part["spend"].rolling(f"{OUTAGE_BASELINE_DAYS}D", min_periods=7).median().shift(1)
        flag = ((part["clicks"].notna()) & (part["spend"] > 0)
                & (base_clicks >= OUTAGE_MIN_BASELINE_CLICKS)
                & (part["clicks"] <= OUTAGE_CLICK_SHARE * base_clicks)
                & (part["spend"] >= OUTAGE_MIN_SPEND_SHARE * base_spend))
        for day in part.index[flag.fillna(False)]:
            hits.append({"group": part.at[day, "group"], "campaign": name, "date": day,
                         "clicks": float(part.at[day, "clicks"]), "baseline": float(base_clicks[day]),
                         "spend": float(part.at[day, "spend"]), "leads": float(part.at[day, "leads"])})
    if not hits:
        return []

    # One anomaly per platform and run of consecutive days.
    hits_df = pd.DataFrame(hits).sort_values(["group", "date"])
    records = []
    for group, part in hits_df.groupby("group"):
        dates = sorted(part["date"].unique())
        runs, run = [], [dates[0]]
        for d in dates[1:]:
            if d - run[-1] == pd.Timedelta(days=1):
                run.append(d)
            else:
                runs.append(run)
                run = [d]
        runs.append(run)
        for run in runs:
            sub = part[part["date"].isin(run)]
            observed = sub.groupby("date")["clicks"].sum().mean()
            baseline = sub.groupby("date")["baseline"].sum().mean()
            details = {"campaigns": sorted(sub["campaign"].unique().tolist()),
                       "days": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in run],
                       "spend_during_outage": round(float(sub["spend"].sum()), 2),
                       "leads_during_outage": float(sub["leads"].sum())}
            records.append(_record("clicks", "platform" if group_col == "platform" else
                                   (group_col or "campaign"), str(group), "day", pd.Timestamp(run[0]),
                                   pd.Timestamp(run[-1]), baseline, observed,
                                   (observed / baseline - 1) * 100 if baseline else None, None,
                                   "tracking_outage_rule", details, n_periods=len(run)))
    return records


# ---------------------------------------------------------------------------------------------
# Record building and plain-English description
# ---------------------------------------------------------------------------------------------

METRIC_LABELS = {"spend": "Spend", "revenue": "Revenue", "clicks": "Clicks", "cpc": "CPC",
                 "cpl": "CPL", "lead_qualification_rate": "Lead qualification rate",
                 "lead_to_conversion_rate": "Lead-to-conversion rate"}


def _record(metric, entity_type, entity, grain, start, end, baseline, observed, pct, score,
            method, details, n_periods=1) -> dict:
    direction = "up" if (pct or 0) > 0 else "down"
    better = KPI_REGISTRY[metric].higher_is_better if metric in KPI_REGISTRY else None
    if method == "tracking_outage_rule":
        sentiment = "negative"
    elif better is None:
        sentiment = "check"
    else:
        sentiment = "positive" if (direction == "up") == better else "negative"
    if score is None:
        severity = "high"
    else:
        severity = "high" if abs(score) >= ANOMALY_HIGH_SCORE and abs(pct) >= 50 else "medium"
    fmt = KPI_REGISTRY[metric].fmt if metric in KPI_REGISTRY else "count"
    label = METRIC_LABELS.get(metric, metric)

    when = (f"the week of {format_date(start)}" if grain == "week" and n_periods == 1 else
            f"{n_periods} weeks from {format_date(start)} to {format_date(end)}" if grain == "week"
            else f"{format_date(start)}" if start == end else f"{format_date(start)} to {format_date(end)}")
    if method == "tracking_outage_rule":
        description = (f"Possible tracking outage on {entity}: on {when} clicks fell to "
                       f"{format_value(observed, 'count')} a day (normally about "
                       f"{format_value(baseline, 'count')}) while spend continued "
                       f"({format_value(details['spend_during_outage'], 'money')}). "
                       f"Campaigns affected: {len(details['campaigns'])}.")
    else:
        verb = "rose" if direction == "up" else "fell"
        numbers = (f"{format_value(observed, fmt)} vs about {format_value(baseline, fmt)} expected "
                   "from its recent weeks and the rest of the business")
        strongest = details.get("strongest")
        if n_periods == 1 or not strongest:
            description = f"{label} for {entity} ({entity_type}) {verb} {abs(pct):.0f}% in {when}: {numbers}."
        else:
            peak_start = pd.Timestamp(strongest["start"])
            peak = (f"the week of {format_date(peak_start)}" if strongest["weeks"] == 1 else
                    f"the {strongest['weeks']} weeks from {format_date(peak_start)}")
            description = (f"{label} for {entity} ({entity_type}) was unusually "
                           f"{'high' if direction == 'up' else 'low'} over {when}. Strongest in "
                           f"{peak}: {verb} {abs(pct):.0f}%, {numbers}.")
    return {"metric": metric, "entity_type": entity_type, "entity": entity, "grain": grain,
            "period_start": pd.Timestamp(start), "period_end": pd.Timestamp(end),
            "baseline": baseline, "observed": observed, "pct_change": pct, "score": score,
            "direction": direction, "sentiment": sentiment, "severity": severity, "method": method,
            "description": description, "details_json": json.dumps(details)}
