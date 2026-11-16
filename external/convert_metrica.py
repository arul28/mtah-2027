"""Convert Metrica Sports sample games 1-3 into per-game tensors, events and meta.

Inputs : C:/Users/arul2/mtah-ext/metrica/raw/data  (git clone of metrica-sports/sample-data)
Outputs: C:/Users/arul2/mtah-ext/metrica/G{1,2,3}.npz
         C:/Users/arul2/mtah-ext/metrica/G{1,2,3}_events.csv
         C:/Users/arul2/mtah-ext/metrica/G{1,2,3}_meta.json

Games 1-2: RawTrackingData CSV (Home/Away files, 3 header lines, x,y pairs, Ball last)
           + RawEventsData CSV.
Game 3   : EPTS (metadata XML + tracking txt) + events JSON.
Run with:  python -I convert_metrica.py
"""

from __future__ import annotations

import bisect
import csv
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(os.environ.get("MTAH_EXT_DIR", Path(__file__).resolve().parent.parent / "external_data")) / "metrica" / "raw" / "data"
OUT = Path(os.environ.get("MTAH_EXT_DIR", Path(__file__).resolve().parent.parent / "external_data")) / "metrica"

FPS = 25.0
PITCH_LEN, PITCH_WID = 105.0, 68.0
POSSESSION_TYPES = {"PASS", "CARRY", "SET PIECE", "SHOT", "RECOVERY"}
DEAD_TRIGGER_TYPES = {"BALL OUT", "FAULT RECEIVED"}
EVENT_COLS = [
    "Team", "Type", "Subtype", "Period", "Start Frame", "Start Time [s]",
    "End Frame", "End Time [s]", "From", "To", "Start X", "Start Y", "End X", "End Y",
]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_csv_tracking(home_path: Path, away_path: Path, game: str):
    """Load the two Metrica CSV tracking files. Returns a dict of raw arrays.

    The team file is identified by its row-1 label ('Home'/'Away'), not by its name.
    """
    teams = {}
    for path in (home_path, away_path):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            team_row = next(reader)
            num_row = next(reader)
            name_row = next(reader)
        label = team_row[3]
        ncols = len(name_row)
        assert name_row[ncols - 2] == "Ball", (path, name_row[-3:])
        player_cols = list(range(3, ncols - 2, 2))
        keys = [name_row[c].replace(" ", "") for c in player_cols]
        data = pd.read_csv(path, header=None, skiprows=3, dtype=np.float64).to_numpy()
        assert data.shape[1] == ncols, (data.shape, ncols)
        teams[label] = dict(
            path=path, keys=keys, cols=player_cols, data=data, ncols=ncols,
            numbers=[num_row[c] for c in player_cols],
        )
    assert set(teams) == {"Home", "Away"}, teams.keys()

    h, a = teams["Home"], teams["Away"]
    # both files must describe the same frames
    assert np.array_equal(h["data"][:, 1], a["data"][:, 1]), "frame mismatch"
    assert np.array_equal(h["data"][:, 0], a["data"][:, 0]), "period mismatch"

    def xy(team):
        d = team["data"]
        cols = team["cols"]
        return np.stack([d[:, cols], d[:, [c + 1 for c in cols]]], axis=-1).astype(np.float32)

    ball_h = h["data"][:, [h["ncols"] - 2, h["ncols"] - 1]]
    ball_a = a["data"][:, [a["ncols"] - 2, a["ncols"] - 1]]
    if not np.allclose(ball_h, ball_a, equal_nan=True):
        print(f"[{game}] WARNING ball columns differ between Home and Away files")

    return dict(
        period=h["data"][:, 0].astype(np.int8),
        frame_id=h["data"][:, 1].astype(np.int64),
        ball_xy=ball_h.astype(np.float32),
        home=xy(h),
        away=xy(a),
        home_keys=h["keys"],
        away_keys=a["keys"],
        home_file=str(h["path"]),
        away_file=str(a["path"]),
        home_numbers=h["numbers"],
        away_numbers=a["numbers"],
    )


def load_csv_events(path: Path) -> pd.DataFrame:
    ev = pd.read_csv(path)
    assert list(ev.columns) == EVENT_COLS, ev.columns
    ev["From ID"] = np.nan
    ev["To ID"] = np.nan
    return ev


def load_epts(meta_path: Path, txt_path: Path, game: str):
    """Parse EPTS metadata + tracking. Returns dict like load_csv_tracking."""
    root = ET.parse(meta_path).getroot()

    gp = {
        p.findtext("Name"): (p.findtext("Value") or "").strip()
        for p in root.findall("Metadata/GlobalConfig/ProviderGlobalParameters/ProviderParameter")
    }
    fh1 = (int(gp["first_half_start"]), int(gp["first_half_end"]))
    fh2 = (int(gp["second_half_start"]), int(gp["second_half_end"]))
    fps = float(root.findtext("Metadata/GlobalConfig/FrameRate"))
    assert fps == FPS, fps

    session = root.find("Metadata/Sessions/Session")
    field = session.find("FieldSize")
    width, height = float(field.findtext("Width")), float(field.findtext("Height"))
    assert (width, height) == (PITCH_LEN, PITCH_WID), (width, height)
    score = session.find("MatchParameters/Score")
    home_id, away_id = score.get("idLocalTeam"), score.get("idVisitingTeam")

    players = {}
    for p in root.findall("Metadata/Players/Player"):
        players[p.get("id")] = dict(team=p.get("teamId"), shirt=int(p.findtext("ShirtNumber")))
    chan = {}
    for pc in root.findall("Metadata/PlayerChannels/PlayerChannel"):
        chan[pc.get("id")] = (pc.get("channelId"), pc.get("playerId"))

    specs = []
    for spec in root.findall("DataFormatSpecifications/DataFormatSpecification"):
        slots = []
        for child in spec:
            if child.tag == "SplitRegister" and child.get("separator") == ";":
                for slot in child:
                    refs = [r.get("playerChannelId") for r in slot.iter("PlayerChannelRef")]
                    assert len(refs) == 2
                    (cx, px), (cy, py) = chan[refs[0]], chan[refs[1]]
                    assert cx == "x" and cy == "y" and px == py, (refs, cx, cy)
                    slots.append(px)
        assert len(slots) == len(set(slots)), "player repeated within one spec"
        specs.append(dict(start=int(spec.get("startFrame")), end=int(spec.get("endFrame")), slots=slots))
    specs.sort(key=lambda s: s["start"])
    spec_starts = [s["start"] for s in specs]

    home_players = sorted((p for p in players if players[p]["team"] == home_id), key=lambda p: players[p]["shirt"])
    away_players = sorted((p for p in players if players[p]["team"] == away_id), key=lambda p: players[p]["shirt"])
    all_players = home_players + away_players
    gidx = {p: i for i, p in enumerate(all_players)}
    spec_cols = [np.array([gidx[p] for p in s["slots"]]) for s in specs]

    with open(txt_path, encoding="ascii") as f:
        lines = [ln for ln in (l.strip() for l in f) if ln]
    T = len(lines)
    P = np.full((T, len(all_players), 2), np.nan, dtype=np.float32)
    frame_id = np.empty(T, dtype=np.int64)
    ball = np.full((T, 2), np.nan, dtype=np.float32)
    for r, line in enumerate(lines):
        head, slot_s, ball_s = line.split(":")
        fr = int(head)
        si = bisect.bisect_right(spec_starts, fr) - 1
        spec = specs[si]
        assert spec["start"] <= fr <= spec["end"], (fr, spec["start"], spec["end"])
        toks = slot_s.split(";")
        assert len(toks) == len(spec["slots"]), (fr, len(toks))
        vals = np.full((len(toks), 2), np.nan, dtype=np.float32)
        for k, tok in enumerate(toks):
            if tok:
                a_, b_ = tok.split(",")
                vals[k, 0] = float(a_)
                vals[k, 1] = float(b_)
        P[r, spec_cols[si]] = vals
        frame_id[r] = fr
        if ball_s:
            bx, by = ball_s.split(",")
            ball[r] = (float(bx), float(by))
    assert np.all(np.diff(frame_id) == 1), "EPTS frames not contiguous"

    period = np.where(
        (frame_id >= fh1[0]) & (frame_id <= fh1[1]), 1,
        np.where((frame_id >= fh2[0]) & (frame_id <= fh2[1]), 2, 0),
    ).astype(np.int8)
    assert np.all(period > 0), "frames outside both halves"

    nh = len(home_players)
    return dict(
        period=period,
        frame_id=frame_id,
        ball_xy=ball,
        home=P[:, :nh, :],
        away=P[:, nh:, :],
        home_keys=home_players,
        away_keys=away_players,
        home_file=str(txt_path),
        away_file=str(txt_path),
        home_numbers=[str(players[p]["shirt"]) for p in home_players],
        away_numbers=[str(players[p]["shirt"]) for p in away_players],
        home_team_id=home_id,
        away_team_id=away_id,
    )


def load_json_events(path: Path, home_id: str, away_id: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rows = []
    for e in d["data"]:
        team_id = (e.get("team") or {}).get("id")
        if team_id == home_id:
            team = "Home"
        elif team_id == away_id:
            team = "Away"
        else:
            raise ValueError(f"unknown team id {team_id}")
        subs = e.get("subtypes")
        if isinstance(subs, dict):
            subs = [subs]
        subtype = "|".join(s["name"] for s in subs) if subs else np.nan
        frm = e.get("from") or {}
        to = e.get("to") or {}
        st, en = e["start"], e["end"]
        rows.append({
            "Team": team,
            "Type": e["type"]["name"],
            "Subtype": subtype,
            "Period": e["period"],
            "Start Frame": st["frame"],
            "Start Time [s]": st["time"],
            "End Frame": en["frame"],
            "End Time [s]": en["time"],
            "From": frm.get("name", np.nan),
            "To": to.get("name", np.nan),
            "Start X": np.nan if st.get("x") is None else st["x"],
            "Start Y": np.nan if st.get("y") is None else st["y"],
            "End X": np.nan if en.get("x") is None else en["x"],
            "End Y": np.nan if en.get("y") is None else en["y"],
            "From ID": frm.get("id", np.nan),
            "To ID": to.get("id", np.nan),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Derived quantities
# ----------------------------------------------------------------------------

def compute_speed(pos: np.ndarray, period: np.ndarray) -> np.ndarray:
    """pos (T,N,2) normalised -> speed (T,N) m/s. Computed per period segment."""
    out = np.full(pos.shape[:2], np.nan, dtype=np.float32)
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        assert np.all(np.diff(rows) == 1), "period rows not contiguous"
        seg = pos[rows].astype(np.float64)
        x, y = seg[..., 0], seg[..., 1]
        dx = np.gradient(x, axis=0)  # central differences (per frame)
        dy = np.gradient(y, axis=0)
        v = np.hypot(dx * PITCH_LEN, dy * PITCH_WID) * FPS
        v = np.array(pd.DataFrame(v).rolling(5, center=True, min_periods=1).mean().to_numpy(), copy=True)
        v[np.isnan(x) | np.isnan(y)] = np.nan
        out[rows] = v
    return out


def build_labels(frame_id: np.ndarray, period: np.ndarray, ev: pd.DataFrame) -> np.ndarray:
    """0 home poss, 1 away poss, 2 dead/unknown, per the agreed rules."""
    label = np.full(len(frame_id), 2, dtype=np.int8)
    team_code = {"Home": 0, "Away": 1}
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        fp = frame_id[rows]
        n = len(rows)
        lo = lambda v: int(np.searchsorted(fp, v, side="left"))   # first row with frame >= v
        hi = lambda v: int(np.searchsorted(fp, v, side="right"))  # first row with frame > v
        e = ev[ev["Period"] == p].sort_values("Start Frame", kind="stable")
        lab = np.full(n, 2, dtype=np.int8)

        # 1) possession: team of most recent possession event (by Start Frame)
        poss = e[e["Type"].isin(POSSESSION_TYPES)]
        ps = poss["Start Frame"].to_numpy()
        pt = poss["Team"].map(team_code).to_numpy()
        for i in range(len(poss)):
            a = lo(ps[i])
            b = lo(ps[i + 1]) if i + 1 < len(poss) else n
            lab[a:b] = pt[i]

        # 2) dead ball: End Frame of trigger -> Start Frame of next SET PIECE
        sp_starts = np.sort(e.loc[e["Type"] == "SET PIECE", "Start Frame"].to_numpy())
        is_off = e["Subtype"].fillna("").str.contains("OFFSIDE", regex=False)
        trig = e[e["Type"].isin(DEAD_TRIGGER_TYPES) | is_off]
        for end_f in trig["End Frame"].to_numpy():
            nxt = sp_starts[sp_starts >= end_f]
            a = lo(end_f)
            b = lo(nxt[0]) if len(nxt) else n
            lab[a:b] = 2

        # 3) period start until first SET PIECE start (kick off) -> dead
        if len(sp_starts):
            lab[: lo(sp_starts[0])] = 2
        # 4) after last event of the period -> dead
        last_end = max(int(e["Start Frame"].max()), int(e["End Frame"].max()))
        lab[hi(last_end):] = 2

        label[rows] = lab
    return label


def possession_stats(label: np.ndarray, period: np.ndarray):
    durs_s, durs_alt_s = [], []
    per_period = {}
    for p in np.unique(period):
        l = label[period == p]
        # maximal runs of identical label in {0,1} (a team switch starts a new possession)
        cuts = np.flatnonzero(np.diff(l) != 0) + 1
        starts = np.r_[0, cuts]
        ends = np.r_[cuts, len(l)]
        runs = [e_ - s_ for s_, e_ in zip(starts, ends) if l[s_] != 2]
        durs_s += [r / FPS for r in runs]
        # alternative: runs of non-dead frames split only by dead ball
        alive = (l != 2).astype(np.int8)
        c2 = np.flatnonzero(np.diff(alive) != 0) + 1
        s2 = np.r_[0, c2]
        e2 = np.r_[c2, len(l)]
        alt = [e_ - s_ for s_, e_ in zip(s2, e2) if alive[s_]]
        durs_alt_s += [r / FPS for r in alt]
        per_period[int(p)] = len(runs)
    d = np.array(durs_s) if durs_s else np.array([np.nan])
    alive_frames = int((label != 2).sum())
    minutes = alive_frames / FPS / 60.0
    return dict(
        n_possessions=len(durs_s),
        n_possessions_per_period=per_period,
        mean_possession_s=float(np.nanmean(d)),
        median_possession_s=float(np.nanmedian(d)),
        possessions_per_alive_min=len(durs_s) / minutes,
        n_possessions_if_team_switch_not_split=len(durs_alt_s),
    )


def check_mapping(ev: pd.DataFrame, frame_id, home, away, home_keys, away_keys, game: str):
    """Check that events' From player sits where the event says, in the team file
    the event's Team points to, and that coordinates share a frame (not flipped)."""
    hk = {k: i for i, k in enumerate(home_keys)}
    ak = {k: i for i, k in enumerate(away_keys)}
    n = found = 0
    found_other = 0
    dists = []
    for r in ev.itertuples(index=False):
        key = r.from_key
        if not isinstance(key, str) or np.isnan(r.start_x) or np.isnan(r.start_y):
            continue
        n += 1
        own_idx, own_arr = (hk, home) if r.Team == "Home" else (ak, away)
        oth_idx, _ = (ak, away) if r.Team == "Home" else (hk, home)
        if key in oth_idx:
            found_other += 1
        if key not in own_idx:
            continue
        found += 1
        row = int(np.searchsorted(frame_id, r.StartF))
        if row >= len(frame_id) or frame_id[row] != r.StartF:
            continue
        p = own_arr[row, own_idx[key]]
        if np.isnan(p).any():
            continue
        dists.append(float(np.hypot(p[0] - r.start_x, p[1] - r.start_y)))
    d = np.array(dists)
    res = dict(
        events_with_coords=n,
        from_player_in_labelled_team_file=found,
        from_player_in_opposite_team_file=found_other,
        median_dist_from_player_to_event_start=float(np.median(d)),
        share_dist_below_0_03=float((d < 0.03).mean()),
    )
    print(f"[{game}] mapping check: {res}")
    return res


# ----------------------------------------------------------------------------
# Per game
# ----------------------------------------------------------------------------

def process(game: str, trk: dict, ev: pd.DataFrame, source: str, events_out: pd.DataFrame):
    frame_id, period = trk["frame_id"], trk["period"]
    assert np.all(np.diff(frame_id) > 0), "frame ids not increasing"

    ball = np.full((len(frame_id), 3), np.nan, dtype=np.float32)
    ball[:, 0:2] = trk["ball_xy"]
    home, away = trk["home"], trk["away"]
    home_speed = compute_speed(home, period)
    away_speed = compute_speed(away, period)

    # event helpers used for labels and checks
    ev = ev.copy()
    ev["StartF"] = ev["Start Frame"].astype(np.int64)
    ev["Start X"] = ev["Start X"].astype(float)
    ev["start_x"] = ev["Start X"]
    ev["start_y"] = ev["Start Y"].astype(float)
    ev["from_key"] = ev["From ID"].where(ev["From ID"].notna(), ev["From"].astype(str).str.replace(" ", ""))
    ev.loc[ev["From"].isna() & ev["From ID"].isna(), "from_key"] = np.nan

    label = build_labels(frame_id, period, ev)
    stats = possession_stats(label, period)
    mapping = check_mapping(ev, frame_id, home, away, trk["home_keys"], trk["away_keys"], game)

    npz_path = OUT / f"{game}.npz"
    np.savez_compressed(
        npz_path,
        period=period,
        frame_id=frame_id,
        ball=ball,
        home=home.astype(np.float32),
        away=away.astype(np.float32),
        home_speed=home_speed,
        away_speed=away_speed,
        pitch=np.array([PITCH_LEN, PITCH_WID], dtype=np.float32),
        label=label,
    )
    events_out.to_csv(OUT / f"{game}_events.csv", index=False)

    T = len(frame_id)
    alive = int((label != 2).sum())
    meta = dict(
        game=game,
        source=source,
        fps=FPS,
        pitch=[PITCH_LEN, PITCH_WID],
        n_frames=T,
        n_frames_per_period={str(int(p)): int((period == p).sum()) for p in np.unique(period)},
        label_frames=dict(home=int((label == 0).sum()), away=int((label == 1).sum()), dead=int((label == 2).sum())),
        alive_share=alive / T,
        n_possessions=stats["n_possessions"],
        n_possessions_per_period=stats["n_possessions_per_period"],
        mean_possession_s=stats["mean_possession_s"],
        median_possession_s=stats["median_possession_s"],
        possessions_per_alive_min=stats["possessions_per_alive_min"],
        n_possessions_if_team_switch_not_split=stats["n_possessions_if_team_switch_not_split"],
        n_events=int(len(events_out)),
        home_file=trk["home_file"],
        away_file=trk["away_file"],
        home_player_keys=trk["home_keys"],
        away_player_keys=trk["away_keys"],
        home_shirt_numbers=trk["home_numbers"],
        away_shirt_numbers=trk["away_numbers"],
        mapping_check=mapping,
        home_players_with_data=int((~np.isnan(home).all(axis=(0, 2))).sum()),
        away_players_with_data=int((~np.isnan(away).all(axis=(0, 2))).sum()),
    )
    (OUT / f"{game}_meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    print(f"[{game}] wrote {npz_path} T={T} alive_share={meta['alive_share']:.4f} "
          f"poss={stats['n_possessions']} mean={stats['mean_possession_s']:.2f}s "
          f"median={stats['median_possession_s']:.2f}s per_min={stats['possessions_per_alive_min']:.3f}")
    return meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Games 1 and 2: CSV
    for n in ("1", "2"):
        game = f"G{n}"
        d = RAW / f"Sample_Game_{n}"
        trk = load_csv_tracking(
            d / f"Sample_Game_{n}_RawTrackingData_Home_Team.csv",
            d / f"Sample_Game_{n}_RawTrackingData_Away_Team.csv",
            game,
        )
        ev_raw = load_csv_events(d / f"Sample_Game_{n}_RawEventsData.csv")
        # events CSV written exactly as read (14 raw columns)
        process(game, trk, ev_raw, f"Metrica sample-data Sample_Game_{n} (CSV)", ev_raw[EVENT_COLS])

    # Game 3: EPTS + JSON. Events CSV keeps the 14 raw columns plus From ID / To ID.
    d = RAW / "Sample_Game_3"
    trk = load_epts(d / "Sample_Game_3_metadata.xml", d / "Sample_Game_3_tracking.txt", "G3")
    ev3 = load_json_events(d / "Sample_Game_3_events.json", trk["home_team_id"], trk["away_team_id"])
    process("G3", trk, ev3, "Metrica sample-data Sample_Game_3 (EPTS + JSON)", ev3.copy())


if __name__ == "__main__":
    main()
