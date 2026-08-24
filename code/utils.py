"""Shared helpers for the QSS 45 caregiving project.

Imported at the top of every script. Nothing here touches the filesystem
except `paths()`, which resolves locations relative to the repository root so
that no script contains a hardcoded path.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_STATE = 20260818

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def paths():
    """Return the project directories, resolved from this file's location.

    Works whether a script is launched from repo root or from code/, which
    is why no script hardcodes an absolute path.
    """
    root = Path(__file__).resolve().parent.parent
    d = {
        "root": root,
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "figures": root / "output" / "figures",
        "tables": root / "output" / "tables",
    }
    for k in ("processed", "figures", "tables"):
        d[k].mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# House plotting style
# ---------------------------------------------------------------------------

# Re-exported from figstyle so the two projects cannot drift to different
# palettes. Change a color there and every figure in both repos follows.
import figstyle
from figstyle import (BLUE, ORANGE, TEAL, GREY, LIGHT, MID, DARK, AMBER,
                      SLATE, INK, GRID, RULE, PAPER,
                      INCOME_COLORS, CATEGORY_COLORS)

# Applied on import rather than per script. The QSS 20 scripts each call
# figstyle.use_paper_style() themselves; here the call lives in the module every
# script already imports, so a script cannot forget it and silently write a
# figure at screen resolution with a clipped title.
figstyle.use_paper_style()


def style_axis(ax, axis="y"):
    """Light grid behind the data, no top or right spine."""
    import figstyle
    return figstyle.style_axis(ax, axis=axis)


def caption(fig, text, size=7.6):
    """Source note placed below the figure box. See figstyle.caption."""
    import figstyle
    figstyle.caption(fig, text, size=size)


def suptitle(fig, text, size=13.5):
    import figstyle
    figstyle.suptitle(fig, text, size=size)


def panel_label(ax, letter, dx=-0.02, dy=1.12):
    import figstyle
    figstyle.panel(ax, letter)


def savefig(fig, name, top=None):
    """Write PNG and PDF using the shared layout engine.

    The engine places captions outside the axes box and saves with a tight
    bounding box, which is what stops titles being clipped and captions landing
    on tick labels.
    """
    import figstyle
    p = paths()["figures"] / name
    figstyle.save(fig, p)
    print("  wrote", p.relative_to(paths()["root"]))


def savetable(df, name, index=False):
    p = paths()["tables"] / name
    df.to_csv(p, index=index)
    print(f"  wrote {p.relative_to(paths()['root'])}")


# ---------------------------------------------------------------------------
# HRS cleaning
# ---------------------------------------------------------------------------

# HRS stores "don't know" / "refused" as reserved numeric codes, not as blanks
DK_RF = {
    "one_digit": [8, 9, -8],
    "two_digit": [98, 99, -8],
    "five_digit": [99998, 99999],
}


def clean_codes(series, missing):
    """Blank out HRS reserved codes so they are not read as real values."""
    return series.where(~series.isin(missing))


def difficulty(series):
    """Recode an HRS ADL/IADL item to 1 = has difficulty, 0 = does not.

    1 = yes and 6 = can't do both count as difficulty; 5 = no and 7 = doesn't
    do count as no difficulty; 8/9/-8 are don't-know / refused.
    """
    s = clean_codes(series, DK_RF["one_digit"])
    return s.map({1: 1, 6: 1, 5: 0, 7: 0})


def merge_report(left, right, on, how, label):
    """Merge two frames and print row counts before and after.

    The QSS 45 repo rubric asks for diagnostic output around every merge; this
    makes that automatic rather than something to remember.
    """
    before = len(left)
    out = left.merge(right, on=on, how=how)
    matched = out[right.columns.difference(pd.Index(on if isinstance(on, list) else [on]))[0]].notna().sum()
    print(f"  merge [{label}]: {before} rows in -> {len(out)} rows out, "
          f"{matched} matched a value on {on} ({how} join)")
    if len(out) != before and how == "left":
        print(f"    WARNING: left join changed row count by {len(out) - before} "
              f"- check for duplicate keys in the right frame")
    return out


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "age", "female", "hispanic", "years_school",
    "sep_divorced", "widowed", "never_married",
    "lives_with_partner", "lives_with_relative", "lives_alone",
    "nursing_home", "proxy", "adl_count", "iadl_count", "self_rated_health",
    "stroke", "alzheimers", "dementia", "psych", "diabetes", "heart",
    "word_recall", "rate_memory", "n_children", "kids_within_10mi",
    "total_care_hours",
]


def build_features(df, extra=None, drop=None):
    """Assemble the model matrix.

    Missingness is handled with an explicit indicator plus median imputation
    rather than letting a booster route NaNs down its own default branches.
    That matters here because the missingness is not at random -- the word
    recall test is missing precisely for the cognitively impaired and
    proxy-interviewed respondents whose care arrangements are most distinctive
    -- and native NaN handling would fold the fact of missingness into the SHAP
    attribution of the parent feature.
    """
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    if extra:
        cols = cols + [c for c in extra if c in df.columns]
    if drop:
        cols = [c for c in cols if c not in drop]
    X = df[cols].copy()

    race = pd.get_dummies(
        df["race"].map({1: "white", 2: "black", 7: "other"}),
        prefix="race", dtype=float)
    if "race_white" in race.columns:
        race = race.drop(columns=["race_white"])  # omitted reference category
    X = pd.concat([X, race], axis=1)

    for col in list(X.columns):
        if X[col].isna().mean() > 0.01:
            X[col + "_missing"] = X[col].isna().astype(float)
    X = X.fillna(X.median())
    return X


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def weighted_mean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = ~np.isnan(x) & ~np.isnan(w)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))


def brier(y_true, p):
    return float(np.mean((np.asarray(p) - np.asarray(y_true)) ** 2))


# ---------------------------------------------------------------------------
# Publication figure style
# ---------------------------------------------------------------------------

def use_paper_style():
    """Apply the shared print-ready matplotlib style."""
    import figstyle
    figstyle.use_paper_style()


def annotate_bars(ax, bars, values, fmt="{:.2f}", pad=0.01, fontsize=9,
                  inside_threshold=None, orient="v"):
    """Label bars without letting the text collide with the bar or the axis.

    If `inside_threshold` is given, bars shorter than it get their label
    outside and taller ones get it inside in white, which is what stops long
    labels running off the end of a wide bar.
    """
    span = (ax.get_ylim()[1] - ax.get_ylim()[0]) if orient == "v" else \
           (ax.get_xlim()[1] - ax.get_xlim()[0])
    for b, v in zip(bars, values):
        inside = inside_threshold is not None and abs(v) >= inside_threshold
        if orient == "v":
            x = b.get_x() + b.get_width() / 2
            y = v - pad * span if inside else v + pad * span
            ax.text(x, y, fmt.format(v), ha="center",
                    va="top" if inside else "bottom",
                    color="white" if inside else "#222222", fontsize=fontsize,
                    fontweight="semibold", zorder=6)
        else:
            y = b.get_y() + b.get_height() / 2
            x = v - pad * span if inside else v + pad * span
            ax.text(x, y, fmt.format(v), va="center",
                    ha="right" if inside else "left",
                    color="white" if inside else "#222222", fontsize=fontsize,
                    fontweight="semibold", zorder=6)


def panel_label(ax, letter, dx=-0.08, dy=1.06):
    """Bold panel letter in the corner, PNAS-style."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")
