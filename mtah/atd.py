"""Load the raw ATD into dense arrays, cached as .npz under cache/.

Arrays, indexed by video frame (ATD frame_id - 1):
  players  float32 (T, S, 3)   x, y normalised 0-1, speed m/s; NaN where the slot is empty
  ball     float32 (T, 3)      x, y normalised 0-1, speed (mostly NaN)
  team     int8    (S,)        0 home, 1 away, per track slot
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402

CACHE = os.path.join(ROOT, "cache")


def slot_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def parse(raw_path, n_slots):
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32,
                        na_values=["NaN"]).values
    frame_id = table[:, 0].astype(np.int64)
    players = table[:, 1:1 + 3 * n_slots].reshape(len(table), n_slots, 3)
    ball = table[:, 1 + 3 * n_slots:4 + 3 * n_slots]
    return frame_id, players, ball


def load(source="tactical", game=GAME, refresh=False):
    path = os.path.join(CACHE, "atd_{}_{}.npz".format(game, source))
    if os.path.exists(path) and not refresh:
        data = np.load(path)
        return {key: data[key] for key in data.files}
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, source=source)
    team = slot_teams(meta_path)
    frame_id, players, ball = parse(raw_path, len(team))
    n = int(frame_id.max())
    full_players = np.full((n, len(team), 3), np.nan, dtype=np.float32)
    full_ball = np.full((n, 3), np.nan, dtype=np.float32)
    full_players[frame_id - 1] = players
    full_ball[frame_id - 1] = ball
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, players=full_players, ball=full_ball, team=team)
    return {"players": full_players, "ball": full_ball, "team": team}


if __name__ == "__main__":
    d = load(refresh=True)
    print({k: v.shape for k, v in d.items()})
    print("ball present", np.mean(~np.isnan(d["ball"][:, 0])))
