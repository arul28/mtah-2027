"""Smart Tagging as per-frame evidence arrays for the competition match.

Metrica's automatic tags are noisy phases, not possessions, so they are only ever used as soft
evidence. In-possession phases name the team with the ball; defending phases name the team
without it.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "starter"))

ATTACKING = {"build up", "progression", "creating chances", "attacking transition",
             "ball in final third", "ball in the box", "players in the box", "long ball"}
DEFENDING = {"defending in attacking third", "defending in middle third",
             "defending in defensive third", "defensive transition"}
HOME_NAME, AWAY_NAME = "barcelona", "atletic club"


def load(n_frames=199319):
    import load_smart_tagging as st
    events = st.load()
    team_vote = np.zeros(n_frames, np.float32)   # +1 per tag saying home has it, -1 away
    setpiece = np.zeros(n_frames, np.float32)
    kickoff_centre = []
    for e in events:
        code = e["code"].strip().lower()
        team = e["tags"].get("Team", "N/A").strip().lower()
        a, b = max(0, e["start_frame"]), min(n_frames, e["end_frame"])
        side = 1.0 if team == HOME_NAME else -1.0 if team == AWAY_NAME else 0.0
        if code in ATTACKING:
            team_vote[a:b] += side
        elif code in DEFENDING:
            team_vote[a:b] -= side
        elif code == "set pieces":
            setpiece[a:b] = 1.0
            if "kick off" in e["tags"].get("Type", "").lower():
                kickoff_centre.append((a + b) // 2)
    return {"team_vote": team_vote, "setpiece": setpiece, "kickoffs": np.array(kickoff_centre)}
