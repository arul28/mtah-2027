"""Convert the open DFL/Sportec IDSSE dataset (Bassek et al. 2025) into per-match
labelled dev-set files.

Outputs per match (in OUT_DIR):
    <match_id>.npz          tracking arrays (see README in the report / below)
    <match_id>_events.csv   kloppy event dataframe (all columns)
    <match_id>_meta.json    team info, coordinate system, possession statistics

Input: the three IDSSE XML files per match, saved locally in RAW_DIR as
    <match_id>_meta.xml, <match_id>_tracking.xml, <match_id>_event.xml
The exact URLs are printed by this script when a file is missing (they are the
figshare URLs kloppy itself uses). Those URLs sit behind an AWS WAF bot
challenge, so they have to be downloaded with a normal browser.

Usage:
    python convert_dfl.py                  # all 7 matches
    python convert_dfl.py J03WPY J03WMX    # selected matches
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from kloppy import sportec
from kloppy import __version__ as KLOPPY_VERSION
from kloppy.domain import BallState, Ground
from kloppy._providers.sportec import get_IDSSE_url
from kloppy.infra.serializers.tracking.sportec.deserializer import (
    GAME_SECTION_TO_PERIOD_ID,
)
from lxml import etree

ROOT = Path(os.environ.get("MTAH_EXT_DIR", Path(__file__).resolve().parent.parent / "external_data"))
RAW_DIR = ROOT / "dfl" / "raw"
OUT_DIR = ROOT / "dfl"

MATCH_IDS = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]
COORD_SYSTEM = "metrica"  # origin top-left, y top->bottom, normalised 0..1
NOT_AVAILABLE = -1


# ---------------------------------------------------------------------------
# Raw XML: ball BallStatus / BallPossession per frame (kloppy only keeps the
# derived owning team, and maps every non-1 possession value to away).
# ---------------------------------------------------------------------------
def parse_raw_ball(tracking_xml: Path) -> dict:
    """Return {(period, frame_N): (ball_status, ball_possession, xy_missing)}.

    ball_status / ball_possession are the raw integers, or -1 if the attribute
    is absent. xy_missing is 1 if the ball X or Y attribute is absent (kloppy
    silently substitutes 0 in that case).
    """
    out: dict = {}
    period = None
    is_ball = False
    context = etree.iterparse(
        str(tracking_xml), events=("start", "end"), huge_tree=True
    )
    for event, elem in context:
        if event == "start" and elem.tag == "FrameSet":
            period = GAME_SECTION_TO_PERIOD_ID.get(elem.get("GameSection"))
            is_ball = (elem.get("TeamId") or "").upper() == "BALL"
        elif event == "end" and elem.tag == "Frame":
            if is_ball and period is not None and elem.get("N") is not None:
                out[(period, int(elem.get("N")))] = (
                    _int_attr(elem.get("BallStatus")),
                    _int_attr(elem.get("BallPossession")),
                    int(elem.get("X") is None or elem.get("Y") is None),
                )
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
    return out


def _int_attr(value):
    if value is None:
        return NOT_AVAILABLE
    try:
        return int(float(value))
    except ValueError:
        return NOT_AVAILABLE


# ---------------------------------------------------------------------------
# Possession statistics
# ---------------------------------------------------------------------------
def run_stats(labels: np.ndarray, periods: np.ndarray, fps: float, target: str):
    """Count maximal runs of `target` and their mean duration (s).

    Runs are computed within each period so they cannot merge across the
    half-time break.
    """
    lengths = []
    for p in np.unique(periods):
        lab = labels[periods == p]
        if lab.size == 0:
            continue
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [lab.size]))
        for s, e in zip(starts, ends):
            if lab[s] == target:
                lengths.append(e - s)
    lengths = np.asarray(lengths, dtype=float)
    mean_s = float(lengths.mean() / fps) if lengths.size else None
    return int(lengths.size), mean_s


def possession_summary(labels, periods, fps):
    n_h, mean_h = run_stats(labels, periods, fps, "H")
    n_a, mean_a = run_stats(labels, periods, fps, "A")
    # mean duration over all H and A runs combined
    combined = []
    for p in np.unique(periods):
        lab = labels[periods == p]
        if lab.size == 0:
            continue
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [lab.size]))
        for s, e in zip(starts, ends):
            if lab[s] in ("H", "A"):
                combined.append(e - s)
    mean_all = float(np.mean(combined) / fps) if combined else None
    return {
        "runs_H": n_h,
        "runs_A": n_a,
        "runs_total": n_h + n_a,
        "mean_possession_s_H": mean_h,
        "mean_possession_s_A": mean_a,
        "mean_possession_s_all": mean_all,
    }


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert(match_id: str) -> dict:
    t0 = time.time()
    paths = {
        "meta": RAW_DIR / f"{match_id}_meta.xml",
        "tracking": RAW_DIR / f"{match_id}_tracking.xml",
        "event": RAW_DIR / f"{match_id}_event.xml",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        urls = {k: get_IDSSE_url(match_id, k) for k in missing}
        raise FileNotFoundError(
            f"{match_id}: missing {missing}. Download and save as "
            + "; ".join(f"{paths[k]} <- {urls[k]}" for k in missing)
        )

    # ---- tracking (no only_alive filter, no sample rate) -------------------
    ds = sportec.load_tracking(
        meta_data=paths["meta"],
        raw_data=paths["tracking"],
        sample_rate=None,
        limit=None,
        coordinates=None,
        only_alive=False,
    )
    orientation_loaded = ds.metadata.orientation.name
    ds = ds.transform(to_coordinate_system=COORD_SYSTEM)
    meta_cs = ds.metadata.coordinate_system
    pitch_length = float(meta_cs.pitch_length)
    pitch_width = float(meta_cs.pitch_width)
    fps = float(ds.metadata.frame_rate)

    teams = list(ds.metadata.teams)
    home = next(t for t in teams if t.ground == Ground.HOME)
    away = next(t for t in teams if t.ground == Ground.AWAY)

    records = sorted(ds.records, key=lambda f: (f.period.id, f.frame_id))
    T = len(records)

    # distinct players observed per side, ordered by metadata squad order
    order = {
        side: {p.player_id: i for i, p in enumerate(team.players)}
        for side, team in ((Ground.HOME, home), (Ground.AWAY, away))
    }
    seen = {Ground.HOME: set(), Ground.AWAY: set()}
    for f in records:
        for pl in f.players_data:
            seen[pl.team.ground].add(pl.player_id)
    cols = {
        side: sorted(seen[side], key=lambda pid: order[side].get(pid, 10**6))
        for side in (Ground.HOME, Ground.AWAY)
    }
    col_idx = {
        side: {pid: i for i, pid in enumerate(cols[side])}
        for side in cols
    }
    Nh, Na = len(cols[Ground.HOME]), len(cols[Ground.AWAY])

    period = np.empty(T, dtype=np.int8)
    frame_id = np.empty(T, dtype=np.int64)
    ball = np.full((T, 3), np.nan, dtype=np.float32)
    home_xy = np.full((T, Nh, 2), np.nan, dtype=np.float32)
    away_xy = np.full((T, Na, 2), np.nan, dtype=np.float32)
    home_sp = np.full((T, Nh), np.nan, dtype=np.float32)
    away_sp = np.full((T, Na), np.nan, dtype=np.float32)
    ball_state = np.full(T, NOT_AVAILABLE, dtype=np.int8)
    owner = np.full(T, NOT_AVAILABLE, dtype=np.int8)
    n_ball_none = 0
    n_owner_none = 0

    for i, f in enumerate(records):
        period[i] = f.period.id
        frame_id[i] = f.frame_id
        bc = f.ball_coordinates
        if bc is not None:
            ball[i] = (bc.x, bc.y, bc.z)
        else:
            n_ball_none += 1
        if f.ball_state == BallState.ALIVE:
            ball_state[i] = 1
        elif f.ball_state == BallState.DEAD:
            ball_state[i] = 0
        if f.ball_owning_team is None:
            n_owner_none += 1
        else:
            owner[i] = 0 if f.ball_owning_team.ground == Ground.HOME else 1
        for pl, pd in f.players_data.items():
            if pl.team.ground == Ground.HOME:
                j = col_idx[Ground.HOME][pl.player_id]
                home_xy[i, j] = (pd.coordinates.x, pd.coordinates.y)
                home_sp[i, j] = pd.speed
            else:
                j = col_idx[Ground.AWAY][pl.player_id]
                away_xy[i, j] = (pd.coordinates.x, pd.coordinates.y)
                away_sp[i, j] = pd.speed

    # ---- raw ball flags from XML, aligned on (period, frame_id) ------------
    raw = parse_raw_ball(paths["tracking"])
    ball_status_raw = np.full(T, NOT_AVAILABLE, dtype=np.int8)
    ball_possession_raw = np.full(T, NOT_AVAILABLE, dtype=np.int8)
    raw_ball_missing_xy = 0
    n_no_raw_frame = 0
    for i in range(T):
        rec = raw.get((int(period[i]), int(frame_id[i])))
        if rec is None:
            n_no_raw_frame += 1
            continue
        ball_status_raw[i] = rec[0]
        ball_possession_raw[i] = rec[1]
        raw_ball_missing_xy += rec[2]
    del raw
    gc.collect()

    # ---- derived statistics -------------------------------------------------
    alive = ball_state == 1
    n_alive = int(alive.sum())
    lab_owner = np.where(owner == 0, "H", np.where(owner == 1, "A", "U"))
    lab_owner = np.where(alive, lab_owner, "D")
    lab_raw = np.where(
        ball_possession_raw == 1,
        "H",
        np.where(ball_possession_raw == 2, "A", "U"),
    )
    lab_raw = np.where(alive, lab_raw, "D")

    # raw vs kloppy consistency and raw coverage while alive
    raw_missing_alive = int(
        (alive & (ball_possession_raw == NOT_AVAILABLE)).sum()
    )
    raw_zero_alive = int((alive & (ball_possession_raw == 0)).sum())
    raw_other_alive = int(
        (alive & ~np.isin(ball_possession_raw, [0, 1, 2, NOT_AVAILABLE])).sum()
    )
    kloppy_vs_raw = {
        "alive_raw_home_kloppy_home": int(
            (alive & (ball_possession_raw == 1) & (owner == 0)).sum()
        ),
        "alive_raw_away_kloppy_away": int(
            (alive & (ball_possession_raw == 2) & (owner == 1)).sum()
        ),
        "alive_raw_not_home_kloppy_away": int(
            (alive & (ball_possession_raw != 1) & (owner == 1)).sum()
        ),
        "alive_raw_not_home_but_raw_present": int(
            (
                alive
                & (ball_possession_raw != 1)
                & (ball_possession_raw != NOT_AVAILABLE)
                & (owner == 1)
            ).sum()
        ),
    }

    home_alive = int((alive & (owner == 0)).sum())
    away_alive = int((alive & (owner == 1)).sum())

    meta = {
        "match_id": match_id,
        "source": "local files in " + str(RAW_DIR),
        "kloppy_version": KLOPPY_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "home_team": {
            "team_id": home.team_id,
            "name": home.name,
            "ground": "HOME",
            "n_players_observed": Nh,
            "player_ids_in_column_order": cols[Ground.HOME],
        },
        "away_team": {
            "team_id": away.team_id,
            "name": away.name,
            "ground": "AWAY",
            "n_players_observed": Na,
            "player_ids_in_column_order": cols[Ground.AWAY],
        },
        "home_is_teams_index0": teams[0].ground == Ground.HOME,
        "coordinate_system": COORD_SYSTEM,
        "coordinate_description": (
            "kloppy MetricaCoordinateSystem: normalised 0..1, origin top-left, "
            "x along length (left->right), y along width (top->bottom)"
        ),
        "orientation_loaded_by_kloppy": orientation_loaded,
        "orientation_used": ds.metadata.orientation.name,
        "orientation_note": (
            "Not changed; kept as set by kloppy's sportec loader "
            "(HOME_AWAY or AWAY_HOME, from the first period's attacking "
            "direction)."
        ),
        "z_units": "metres (kloppy passes z through unchanged)",
        "speed_units": "m/s, raw Sportec S attribute via kloppy PlayerData.speed",
        "pitch_length_m": pitch_length,
        "pitch_width_m": pitch_width,
        "fps": fps,
        "n_frames_total": T,
        "n_frames_per_period": {
            str(int(p)): int((period == p).sum()) for p in np.unique(period)
        },
        "frames_without_kloppy_ball_coordinates": n_ball_none,
        "frames_with_ball_state_unknown": int((ball_state == -1).sum()),
        "frames_with_owner_None": n_owner_none,
        "frames_without_raw_ball_frame": n_no_raw_frame,
        "raw_ball_xy_attr_missing_frames": raw_ball_missing_xy,
        "share_alive": float(n_alive / T) if T else None,
        "share_alive_owned_by_home": (
            float(home_alive / n_alive) if n_alive else None
        ),
        "alive_frames": n_alive,
        "alive_owned_home": home_alive,
        "alive_owned_away": away_alive,
        "alive_owner_unknown": int((alive & (owner == NOT_AVAILABLE)).sum()),
        "possession_kloppy": possession_summary(
            np.asarray(lab_owner), period, fps
        ),
        "possession_raw_BallPossession": possession_summary(
            np.asarray(lab_raw), period, fps
        ),
        "raw_BallPossession_alive_missing": raw_missing_alive,
        "raw_BallPossession_alive_zero": raw_zero_alive,
        "raw_BallPossession_alive_other": raw_other_alive,
        "raw_vs_kloppy_alive": kloppy_vs_raw,
        "runtime_s": round(time.time() - t0, 1),
    }

    # ---- write npz -----------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / f"{match_id}.npz",
        period=period,
        frame_id=frame_id,
        ball=ball,
        home=home_xy,
        away=away_xy,
        home_speed=home_sp,
        away_speed=away_sp,
        ball_state=ball_state,
        ball_owning_team=owner,
        ball_possession_raw=ball_possession_raw,
        ball_status_raw=ball_status_raw,
        pitch=np.array([pitch_length, pitch_width], dtype=np.float32),
    )

    # ---- events ---------------------------------------------------------------
    del records, ds
    gc.collect()
    ev = sportec.load_event(event_data=paths["event"], meta_data=paths["meta"])
    ev = ev.transform(to_coordinate_system=COORD_SYSTEM)
    ev.to_df().to_csv(OUT_DIR / f"{match_id}_events.csv", index=False)
    meta["n_events"] = len(ev.records)
    del ev
    gc.collect()

    with open(OUT_DIR / f"{match_id}_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    meta["runtime_s"] = round(time.time() - t0, 1)
    return meta


def main(argv):
    match_ids = argv or MATCH_IDS
    failures = {}
    for mid in match_ids:
        try:
            m = convert(mid)
            print(f"[ok] {mid}: frames={m['n_frames_total']} "
                  f"alive={m['share_alive']:.3f} "
                  f"runs H/A={m['possession_kloppy']['runs_H']}/"
                  f"{m['possession_kloppy']['runs_A']} "
                  f"({m['runtime_s']}s)")
        except Exception as exc:  # keep going, record why
            failures[mid] = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {mid}: {failures[mid]}", file=sys.stderr)
        gc.collect()
    if failures:
        print("Failures:", json.dumps(failures, indent=2), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
