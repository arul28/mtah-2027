"""The one command that regenerates the submission file.

    python -m mtah.final                     # writes submissions/<NAME>.csv

Everything that decides the output is in CONFIG below; each value was chosen on the external
dev games (see NOTES.md), never on the public 5 minutes. Runs are deterministic: fixed seeds,
deterministic LightGBM, and torch deterministic algorithms (the GRU trains on CPU by default,
where that is guaranteed; MTAH_DEVICE=cuda is faster but only reproducible on the same GPU stack).
"""

import hashlib
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mtah import decode, dev, experiment, match, rules, seq  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = os.environ.get("MTAH_TEAM", "submission")

CONFIG = {
    "train_seeds": (0, 1, 2),     # degraded copies of each dev game
    "dfl_tau": 0.0,               # stickiness of the DFL training labels (s)
    "skillcorner": False,         # add the 20 SkillCorner games to training
    "symmetries": 1,              # 1 none, 2 home/away swap, 4 swap and width mirror (mtah/symm.py)
    "vball": True,             # match ball corrected by the tactical video (mtah/vball.py)
    "ball_smooth": 0.0,           # dev ball made as smooth as the ATD's (degrade.BALL_SMOOTH)
    "ball_sharp": False,          # ... except on video-trusted runs (degrade.BALL_SHARP)
    "tags": True,                 # Smart Tagging features (simulated on dev games)
    "gru_models": 6,
    "gru_epochs": 20,
    "gru_hidden": 128,
    "unet_models": 4,             # 1-D U-Net seeds added to the blend (mtah/seq4.py)
    "lgb_weight": 0.25,
    "unet_weight": 0.25,          # the GRU takes the rest
    "c_turn": 10.0,
    "c_dead": 40.0,
    "bias_d": 0.5,
}


def set_determinism():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    experiment.PARAMS.update(deterministic=True, force_row_wise=True, seed=0)


def training_set(cfg):
    from mtah import degrade
    degrade.BALL_SMOOTH, degrade.BALL_SHARP = cfg["ball_smooth"], cfg["ball_sharp"]
    games = dev.available() + list(dev.METRICA)
    Xs, ys = [], []
    for s in cfg["train_seeds"]:
        for g in games:
            _, X, y, p = dev.load_match(g, seed=s)
            if cfg["tags"]:
                X = np.concatenate([X, dev.tag_block(y, p, 100 + s)[1]], 1)
            Xs.append(X)
            ys.append(y if g in dev.METRICA else dev.sticky(y, p, cfg["dfl_tau"]))
    for g in (dev.skillcorner() if cfg["skillcorner"] else []):
        _, X, y, p = dev.load_match(g, seed=0)
        if cfg["tags"]:
            X = np.concatenate([X, dev.tag_block(np.where(y < 0, dev.D, y), p, 100)[1]], 1)
        Xs.append(X)
        ys.append(y)
    return Xs, ys


def video_ball():
    """Build the video caches if missing: blobs (~10 min on 20 processes), then the ball."""
    from mtah import blobs, vball
    if not os.path.exists(blobs.OUT):
        blobs.main()
    if not os.path.exists(vball.OUT):
        vball.main()


def probabilities(cfg):
    set_determinism()
    if cfg["vball"] or cfg["ball_sharp"]:
        video_ball()
    seq.DEVICE = os.environ.get("MTAH_DEVICE", "cpu")
    Xs, ys = training_set(cfg)
    match.VBALL = cfg["vball"]
    names, Xm = match.load_features()
    names = list(names)
    if cfg["tags"]:
        Xm = np.concatenate([Xm, match.tag_block()[1]], 1)
    runs = [(first, last + 1) for _, first, last in match.HALVES]
    if cfg["symmetries"] > 1:
        from mtah import experiment8, symm
        vs = symm.variants(names + (list(match.tag_block()[0]) if cfg["tags"] else []),
                           cfg["symmetries"])
        p_lgb = experiment8.tta(experiment8.fit_lgb(Xs, ys, vs).predict, Xm, vs)
    else:
        vs = None
        p_lgb = experiment.fit(Xs, ys).predict(Xm)
    grus = []
    for k in range(cfg["gru_models"]):
        model = seq.train(Xs, ys, epochs=cfg["gru_epochs"], seed=k, hidden=cfg["gru_hidden"],
                          variants=vs)
        if vs:
            grus.append(experiment8.tta(lambda Z: seq.predict(model, Z, runs), Xm, vs))
        else:
            grus.append(seq.predict(model, Xm, runs))
    p_gru = np.mean(grus, 0)
    w = cfg["lgb_weight"]
    proba = w * p_lgb + (1 - w) * p_gru
    if cfg["unet_models"]:
        from mtah import seq4
        unets = [seq4.predict(seq4.train(Xs, ys, epochs=cfg["gru_epochs"], seed=100 + k,
                                         hidden=cfg["gru_hidden"]), Xm, runs)
                 for k in range(cfg["unet_models"])]
        wu = cfg["unet_weight"]
        proba = (1 - w - wu) * p_gru + wu * np.mean(unets, 0) + w * p_lgb
    return rules.apply_prekickoff_dead(proba, match.HALVES), runs


def main():
    cfg = CONFIG
    proba, runs = probabilities(cfg)
    # Rounded before decoding so that last-bit float differences between machines cannot move a
    # decision; 1e-4 is far below anything the decoder costs can resolve.
    proba = np.round(proba.astype(np.float64), 4)
    np.save(os.path.join(ROOT, "cache", "proba_final_{}.npy".format(NAME)), proba)
    path = decode.viterbi(proba, cfg["c_turn"], cfg["c_dead"], (0, 0, cfg["bias_d"]), runs)
    graded = match.graded_mask(len(proba))
    os.makedirs(os.path.join(ROOT, "submissions"), exist_ok=True)
    out = os.path.join(ROOT, "submissions", NAME + ".csv")
    from mtah.predict import write_submission
    write_submission(out, path, graded)
    digest = hashlib.sha256(open(out, "rb").read()).hexdigest()[:16]
    print("wrote", out, "sha256", digest)


if __name__ == "__main__":
    main()
