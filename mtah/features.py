"""Per-frame features that do not depend on track identity.

The ATD has 46 anonymous slots that fragment and swap, so everything here is computed over the
*set* of a team's visible players: nearest distances, counts, centroids, spreads, speeds. The same
code runs on the external dev games (other matches, clean tracking we degrade on purpose) and on
the competition match, so a model trained there sees the same inputs here.

Coordinates come in normalised 0-1 and are converted to metres centred on the pitch, then flipped
per period so the home team always attacks +x.
"""

import numpy as np

FPS = 25
VERSION = "f6"  # bump when features change; it is part of every feature cache filename
# (f7 = block-to-ball distance, goal-side counts and players near the ball's projected position was
#  tested on the same protocol: +0.001 at (10,40,0.5) but -0.002 at (15,40) and (15,80), 7/10 then
#  2-3/10 games up. f8 = stoppage evidence (everyone standing, the ball not moving, out of play)
#  was 0.7493 vs 0.7505 at (10,40,0.5) and 0.7518 vs 0.7551 at (15,40), with G3 clearly worse.
#  Both are inside noise and negative where it matters, so not adopted; set VERSION to reproduce.)


def to_metres(xy, pitch):
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = (xy[..., 0] - 0.5) * pitch[0]
    out[..., 1] = (xy[..., 1] - 0.5) * pitch[1]
    return out


def home_attack_sign(home_x, period):
    """+1 where home attacks +x, per frame. Decided per period from where home's deepest player
    (normally the goalkeeper) sits: the side its extreme player hugs is the side it defends."""
    sign = np.ones(len(period), dtype=np.float32)
    import warnings
    for p in np.unique(period):
        m = period == p
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # an all-NaN row is a frame with nobody tracked
            lo = np.nanmin(home_x[m], axis=1)
            hi = np.nanmax(home_x[m], axis=1)
        # the goalkeeper keeps an extreme near its own goal line most of the half
        defends_left = np.nanmedian(lo) + np.nanmedian(hi) > 0  # hugs -x more than +x
        sign[m] = 1.0 if defends_left else -1.0
    return sign


def nanstat(fn, a, axis):
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(a, axis=axis)


def rolling_mean(x, half, causal=None):
    """Centred (or past-only / future-only) moving mean ignoring NaN. half in frames."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    n = len(x)
    i = np.arange(n)
    if causal == "past":
        lo, hi = np.maximum(0, i - half), i + 1
    elif causal == "future":
        lo, hi = i, np.minimum(n, i + half + 1)
    else:
        lo, hi = np.maximum(0, i - half), np.minimum(n, i + half + 1)
    cnt = cn[hi] - cn[lo]
    with np.errstate(all="ignore"):
        return np.where(cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan).astype(np.float32)


def frames_since(mask):
    """Frames since mask was last True (0 when True). Large where never."""
    idx = np.where(mask, np.arange(len(mask)), -10 ** 6)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(len(mask)) - last, 10 ** 4).astype(np.float32)


def ffill(a):
    a = a.copy()
    ok = ~np.isnan(a[:, 0])
    idx = np.where(ok, np.arange(len(a)), 0)
    idx = np.maximum.accumulate(idx)
    out = a[idx]
    out[~np.maximum.accumulate(ok)] = np.nan
    return out


def kick_features(Bm, h_dmin, a_dmin, present):
    """Kicks and receptions read off the ball's speed, and which side made them.

    A kick is the ball accelerating sharply away from a player, a reception the ball slowing
    sharply next to one. The side of the nearest player (within 3 m) at that moment is credited.
    "Two passes in a row by the same side" is the challenge's own proxy for controlled
    possession, so the last kicker / receiver and their agreement are exposed directly.
    """
    T = len(Bm)
    v = np.full(T, np.nan, np.float32)
    v[2:] = np.hypot(Bm[2:, 0] - Bm[:-2, 0], Bm[2:, 1] - Bm[:-2, 1]) * (FPS / 2.0)
    v = np.minimum(v, 40.0)
    dv = np.full(T, np.nan, np.float32)
    dv[4:] = v[4:] - v[:-4]
    near_side = np.where(h_dmin < a_dmin, 1.0, -1.0)
    near_ok = present & (np.fmin(h_dmin, a_dmin) < 3.0)
    kick = near_ok & (dv > 5.0) & (v > 8.0)
    recv = near_ok & (dv < -5.0)
    out = {}
    for name, ev in (("kick", kick), ("recv", recv)):
        # keep the first frame of each burst
        ev = ev & ~np.concatenate(([False], ev[:-1]))
        side = np.where(ev, near_side, 0.0)
        idx = np.where(ev, np.arange(T), 0)
        idx = np.maximum.accumulate(idx)
        last = np.where(np.maximum.accumulate(ev), side[idx], 0.0)
        out["last_" + name] = last.astype(np.float32)
        out["since_" + name] = frames_since(ev)
        for half in (50, 125):
            out["{}_bal_p{}".format(name, half)] = rolling_mean(side, half, "past")
            out["{}_bal_f{}".format(name, half)] = rolling_mean(side, half, "future")
    out["kick_recv_agree"] = out["last_kick"] * out["last_recv"]
    if VERSION >= "f5":
        # which way the ball travels: home attacks +x, so a team in possession mostly moves it
        # towards its own +x (home) or -x (away)
        vx = np.full(T, np.nan, np.float32)
        vx[2:-2] = (Bm[4:, 0] - Bm[:-4, 0]) * (FPS / 4.0)
        vx = np.clip(vx, -40.0, 40.0)
        for half in (12, 50):
            out["ball_vx_c%d" % half] = rolling_mean(vx, half)
            out["ball_vx_p%d" % half] = rolling_mean(vx, half, "past")
            out["ball_vx_f%d" % half] = rolling_mean(vx, half, "future")
        # direction of each kick relative to the kicker's attack: + forward, - backwards
        ahead = rolling_mean(vx, 5, "future")
        kick1 = kick & ~np.concatenate(([False], kick[:-1]))
        fwd = np.where(kick1, np.sign(np.nan_to_num(ahead)) * near_side, 0.0)
        idx = np.maximum.accumulate(np.where(kick1, np.arange(T), 0))
        out["last_kick_fwd"] = np.where(np.maximum.accumulate(kick1), fwd[idx], 0.0).astype(
            np.float32)
        out["last_kick_dir"] = np.where(np.maximum.accumulate(kick1),
                                        np.sign(np.nan_to_num(ahead))[idx], 0.0).astype(np.float32)
    return out


def intercept_features(Bm, P, present):
    """Who gets to the ball first: closest approach under the ball's current velocity.

    For each player, relative position r = p - b and relative velocity u = v_p - v_b; the closest
    approach happens at t* = -<r,u>/|u|^2 with distance |r + u t*|. Aggregated per team: the
    minimum approach distance and its time, plus how many players sit on the ball's path.
    """
    T = len(Bm)
    vb = np.zeros((T, 2))
    vb[3:-3] = (Bm[6:] - Bm[:-6]) / 6.0 * np.array([1.0, 1.0])
    vb[~present] = 0.0
    vp = np.zeros_like(P)
    vp[2:] = (P[2:] - P[:-2]) / 2.0
    r = P - Bm[:, None, :]
    u = vp - vb[:, None, :]
    uu = (u ** 2).sum(-1)
    ru = (r * u).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.clip(-ru / np.maximum(uu, 1e-9), 0.0, 50.0)
    close = np.hypot(*(r + u * t[..., None]).transpose(2, 0, 1))
    close = np.where(np.isnan(r[..., 0]) | ~present[:, None], np.inf, close)
    t = np.where(np.isinf(close), np.inf, t)
    k = np.argmin(close, 1)
    dmin = np.take_along_axis(close, k[:, None], 1)[:, 0]
    tmin = np.take_along_axis(t, k[:, None], 1)[:, 0]
    sp = np.hypot(vb[:, 0], vb[:, 1])
    moving = present & (sp > 6.0 / FPS)
    dirx = np.where(moving, vb[:, 0] / np.maximum(sp, 1e-9), 0.0)
    diry = np.where(moving, vb[:, 1] / np.maximum(sp, 1e-9), 0.0)
    fwd = r[..., 0] * dirx[:, None] + r[..., 1] * diry[:, None]
    lat = np.abs(r[..., 0] * diry[:, None] - r[..., 1] * dirx[:, None])
    corridor = np.sum((fwd > 1.0) & (fwd < 20.0) & (lat < 4.0) & ~np.isnan(r[..., 0]), 1)
    return {"cap": np.where(present, dmin, np.nan),
            "cap_t": np.where(present, np.minimum(tmin, 60.0), np.nan),
            "corridor": np.where(present, corridor, np.nan).astype(np.float32)}


def frame_features(home, away, home_spd, away_spd, ball, period, pitch):
    """home/away (T,N,2) normalised xy with NaN; *_spd (T,N) m/s; ball (T,2) normalised.

    Returns (names, float32 matrix (T, F)).
    """
    H = to_metres(home, pitch)
    A = to_metres(away, pitch)
    Bm = to_metres(ball, pitch)
    sign = home_attack_sign(H[..., 0], period)
    H[..., 0] *= sign[:, None]
    A[..., 0] *= sign[:, None]
    Bm[:, 0] *= sign
    T = len(Bm)
    half_len, half_wid = pitch[0] / 2.0, pitch[1] / 2.0
    f = {}

    present = ~np.isnan(Bm[:, 0])
    f["ball_present"] = present.astype(np.float32)
    f["ball_x"] = Bm[:, 0]
    f["ball_y"] = Bm[:, 1]
    f["ball_out"] = np.maximum(np.abs(Bm[:, 0]) - half_len, np.abs(Bm[:, 1]) - half_wid)
    v = np.full(T, np.nan, np.float32)
    v[1:] = np.hypot(np.diff(Bm[:, 0]), np.diff(Bm[:, 1])) * FPS
    v = np.minimum(v, 45.0)
    f["ball_v"] = rolling_mean(v, 2)
    f["ball_v_1s"] = rolling_mean(v, 12)
    f["ball_since"] = frames_since(present)
    f["ball_until"] = frames_since(present[::-1])[::-1]
    lastB = ffill(Bm)
    f["ball_last_x"] = lastB[:, 0]
    f["ball_last_y"] = lastB[:, 1]

    for name, P, S in (("h", H, home_spd), ("a", A, away_spd)):
        ok = ~np.isnan(P[..., 0])
        f[name + "_n"] = ok.sum(1).astype(np.float32)
        f[name + "_cx"] = nanstat(np.nanmean, P[..., 0], 1)
        f[name + "_cy"] = nanstat(np.nanmean, P[..., 1], 1)
        f[name + "_sx"] = nanstat(np.nanstd, P[..., 0], 1)
        f[name + "_sy"] = nanstat(np.nanstd, P[..., 1], 1)
        f[name + "_maxx"] = nanstat(np.nanmax, P[..., 0], 1)
        f[name + "_minx"] = nanstat(np.nanmin, P[..., 0], 1)
        f[name + "_spd"] = nanstat(np.nanmean, S, 1)
        f[name + "_spd_max"] = nanstat(np.nanmax, S, 1)
        f[name + "_n_fast"] = np.nansum(S > 4.0, axis=1).astype(np.float32)
        # distances to the ball, both live and to its last known position
        for tag, Bp in (("", Bm), ("_last", lastB)):
            d = np.hypot(P[..., 0] - Bp[:, None, 0], P[..., 1] - Bp[:, None, 1])
            f[name + "_dmin" + tag] = nanstat(np.nanmin, d, 1)
            if tag == "":
                srt = np.sort(np.where(np.isnan(d), np.inf, d), axis=1)
                second = srt[:, 1] if srt.shape[1] > 1 else np.full(T, np.inf)
                f[name + "_d2"] = np.where(np.isinf(second), np.nan, second).astype(np.float32)
                f[name + "_n3"] = np.sum(d < 3.0, axis=1).astype(np.float32)
                f[name + "_n10"] = np.sum(d < 10.0, axis=1).astype(np.float32)
                f[name + "_n10"][~present] = np.nan
                f[name + "_n3"][~present] = np.nan

    for name, P, S in (("h", H, home_spd), ("a", A, away_spd)):
        # team shape relative to its own attacking direction (home attacks +x, away -x)
        own = P[..., 0] if name == "h" else -P[..., 0]
        f[name + "_n_att_third"] = np.sum(own > half_len / 3.0, axis=1).astype(np.float32)
        f[name + "_n_def_third"] = np.sum(own < -half_len / 3.0, axis=1).astype(np.float32)
        cx = f[name + "_cx"] if name == "h" else -f[name + "_cx"]
        vx = np.full(T, np.nan, np.float32)
        vx[12:] = (cx[12:] - cx[:-12]) * (FPS / 12.0)
        f[name + "_advance"] = vx  # m/s the block moves towards the opponent goal
        # players standing still: set pieces freeze most of the pitch
        f[name + "_n_still"] = np.nansum(S < 0.8, axis=1).astype(np.float32)
        if VERSION >= "f2":
            # penalty boxes: corners and wide free kicks pack them, open play rarely does
            box_x = half_len - 16.5
            in_y = np.abs(P[..., 1]) < 20.16
            f[name + "_n_opp_box"] = np.sum((own > box_x) & in_y, axis=1).astype(np.float32)
            f[name + "_n_own_box"] = np.sum((own < -box_x) & in_y, axis=1).astype(np.float32)
            # the three nearest to the last known ball, and how fast the nearest is closing in
            d = np.hypot(P[..., 0] - lastB[:, None, 0], P[..., 1] - lastB[:, None, 1])
            dd = np.where(np.isnan(d), np.inf, d)
            order = np.argsort(dd, axis=1)
            srt = np.take_along_axis(dd, order, 1)
            for k in range(3):
                f["{}_dlast{}".format(name, k + 1)] = np.where(
                    np.isinf(srt[:, k]), np.nan, srt[:, k]).astype(np.float32)
            nearest_spd = np.take_along_axis(np.nan_to_num(S, nan=np.nan), order[:, :1], 1)[:, 0]
            f[name + "_near_spd"] = np.where(np.isinf(srt[:, 0]), np.nan, nearest_spd)
            dmin = f[name + "_dlast1"]
            close = np.full(T, np.nan, np.float32)
            close[5:] = (dmin[:-5] - dmin[5:]) * (FPS / 5.0)
            f[name + "_closing"] = close

    # who last had the ball at their feet: possession persists until the other side controls it
    for radius in (1.5, 3.0):
        near_h = present & (f["h_dmin"] < radius) & (f["h_dmin"] < f["a_dmin"])
        near_a = present & (f["a_dmin"] < radius) & (f["a_dmin"] < f["h_dmin"])
        sh, sa = frames_since(near_h), frames_since(near_a)
        uh, ua = frames_since(near_h[::-1])[::-1], frames_since(near_a[::-1])[::-1]
        tag = str(radius).replace(".", "")
        f["since_h_" + tag] = np.minimum(sh, 1500)
        f["since_a_" + tag] = np.minimum(sa, 1500)
        f["since_diff_" + tag] = np.clip(sa - sh, -1500, 1500)
        f["until_diff_" + tag] = np.clip(ua - uh, -1500, 1500)
        # share of close control by home over the last / next few seconds
        for half in (25, 75):
            ph = rolling_mean(near_h.astype(np.float32), half, "past")
            pa = rolling_mean(near_a.astype(np.float32), half, "past")
            fh = rolling_mean(near_h.astype(np.float32), half, "future")
            fa = rolling_mean(near_a.astype(np.float32), half, "future")
            f["ctrl_past_{}_{}".format(tag, half)] = ph - pa
            f["ctrl_fut_{}_{}".format(tag, half)] = fh - fa

    if VERSION >= "f2":
        # a ball that sits still (restart spot) vs one that is moving, over 2 s
        bx = rolling_mean(Bm[:, 0], 25)
        by = rolling_mean(Bm[:, 1], 25)
        bx2 = rolling_mean(Bm[:, 0] ** 2, 25)
        by2 = rolling_mean(Bm[:, 1] ** 2, 25)
        f["ball_wander"] = np.sqrt(np.maximum(bx2 - bx ** 2, 0) + np.maximum(by2 - by ** 2, 0))
        f["ball_edge"] = np.minimum(half_len - np.abs(lastB[:, 0]), half_wid - np.abs(lastB[:, 1]))
        f["box_diff"] = (f["h_n_opp_box"] + f["a_n_own_box"]) - (f["a_n_opp_box"] + f["h_n_own_box"])
        f["dlast_diff"] = f["a_dlast1"] - f["h_dlast1"]
        f["closing_diff"] = f["h_closing"] - f["a_closing"]

    if VERSION >= "f4":
        f.update(kick_features(Bm, f["h_dmin"], f["a_dmin"], present))

    if VERSION >= "f6":
        # who is first to the ball: closest approach under the ball's current velocity
        ih = intercept_features(Bm, H, present)
        ia = intercept_features(Bm, A, present)
        f["h_cap"] = ih["cap"]
        f["a_cap"] = ia["cap"]
        f["cap_diff"] = np.where(present, np.minimum(ia["cap"], 80) - np.minimum(ih["cap"], 80),
                                 np.nan)
        f["h_cap_t"] = ih["cap_t"]
        f["a_cap_t"] = ia["cap_t"]
        f["cap_t_diff"] = np.where(present, np.minimum(ia["cap_t"], 60) - np.minimum(
            ih["cap_t"], 60), np.nan)
        f["h_corridor"] = ih["corridor"]
        f["a_corridor"] = ia["corridor"]
        f["corridor_diff"] = f["h_corridor"] - f["a_corridor"]
        # ball in flight: fast and away from every player
        flight = present & (f["ball_v"] > 8.0) & (np.fmin(f["h_dmin"], f["a_dmin"]) > 2.5)
        f["flight"] = flight.astype(np.float32)
        f["since_flight"] = frames_since(flight)
        # acceleration of the ball (kicks show up here even when the kick detector misses)
        acc = np.full(T, np.nan, np.float32)
        acc[1:-1] = np.hypot(*(np.diff(Bm, 2, axis=0)).T)
        f["ball_acc"] = rolling_mean(acc, 2)
        f["ball_acc12"] = rolling_mean(acc, 12)
        # a kick that has not been answered by a reception yet: the pass is in the air
        out_k = f["last_kick"]
        f["kick_unanswered"] = (f["since_kick"] < f["since_recv"]).astype(np.float32) * out_k
        for half in (250, 500):
            f["last_kick_l%d" % half] = rolling_mean(f["last_kick"], half, "past")
            f["last_recv_l%d" % half] = rolling_mean(f["last_recv"], half, "past")

    if VERSION >= "f7":
        # where the ball is relative to each team's block, and to where it is going
        bf = ffill(Bm)
        vbx = np.zeros((T, 2))
        vbx[3:-3] = (Bm[6:] - Bm[:-6]) / 6.0
        vbx[~present] = 0.0
        fut = bf + vbx * 25.0
        for name, P in (("h", H), ("a", A)):
            own = P[..., 0] if name == "h" else -P[..., 0]
            bown = bf[:, 0] if name == "h" else -bf[:, 0]
            f[name + "_bdist"] = np.hypot(f[name + "_cx"] - bf[:, 0], f[name + "_cy"] - bf[:, 1])
            # players between the ball and this team's own goal, and ahead of it
            f[name + "_n_goalside"] = np.sum(own < bown[:, None], axis=1).astype(np.float32)
            f[name + "_n_fut"] = np.sum(
                np.hypot(P[..., 0] - fut[:, None, 0], P[..., 1] - fut[:, None, 1]) < 8.0,
                axis=1).astype(np.float32)
        f["bdist_diff"] = f["a_bdist"] - f["h_bdist"]
        f["goalside_diff"] = f["h_n_goalside"] - f["a_n_goalside"]
        f["n_fut_diff"] = f["h_n_fut"] - f["a_n_fut"]

    if VERSION >= "f8":
        # stoppage evidence: everyone standing, the ball not moving, the ball out of play
        n_still_all = np.nan_to_num(f["h_n_still"]) + np.nan_to_num(f["a_n_still"])
        for name, S in (("h", home_spd), ("a", away_spd)):
            f[name + "_n_walk"] = np.nansum(S < 1.5, axis=1).astype(np.float32)
        f["n_walk_all"] = f["h_n_walk"] + f["a_n_walk"]
        f["n_still_all"] = n_still_all
        calm = n_still_all >= 12
        f["calm"] = calm.astype(np.float32)
        f["calm_run"] = np.minimum(frames_since(~calm), 1500)
        f["calm_after"] = np.minimum(frames_since(calm[::-1])[::-1], 1500)
        # a ball that is not visibly moving (absent counts: it is held, or off the pitch)
        ball_slow = np.nan_to_num(f["ball_v"], nan=0.0) < 1.0
        f["ball_static_run"] = np.minimum(frames_since(~ball_slow), 1500)
        out_play = f["ball_out"] > 0.0
        f["ball_out_run"] = np.minimum(frames_since(~out_play), 1500)
        f["ball_out_after"] = np.minimum(frames_since(out_play[::-1])[::-1], 1500)
        for key in ("calm", "ball_static_run", "ball_out_run", "n_walk_all"):
            for half in (50, 125, 250):
                f["{}_c{}".format(key, half)] = rolling_mean(f[key], half)

    if VERSION >= "f9":
        # the ball reversing direction is a turnover cue: a pass goes one way, the next goes back
        vxb = np.full(T, np.nan, np.float32)
        vyb = np.full(T, np.nan, np.float32)
        vxb[4:-4] = (Bm[8:, 0] - Bm[:-8, 0]) * (FPS / 8.0)
        vyb[4:-4] = (Bm[8:, 1] - Bm[:-8, 1]) * (FPS / 8.0)
        fast = present & (np.hypot(vxb, vyb) > 2.0)
        sx = np.where(fast, np.sign(vxb), 0.0)
        sy = np.where(fast, np.sign(vyb), 0.0)
        rev_x = np.zeros(T, bool)
        rev_y = np.zeros(T, bool)
        k = 6
        prev_x = np.r_[np.zeros(k), sx[:-k]]
        prev_y = np.r_[np.zeros(k), sy[:-k]]
        rev_x[k:] = fast[k:] & (prev_x[k:] != 0) & (sx[k:] != 0) & (sx[k:] != prev_x[k:])
        rev_y[k:] = fast[k:] & (prev_y[k:] != 0) & (sy[k:] != 0) & (sy[k:] != prev_y[k:])
        rev = rev_x | rev_y
        f["rev_x"] = rev_x.astype(np.float32)
        f["rev_y"] = rev_y.astype(np.float32)
        f["since_rev"] = np.minimum(frames_since(rev), 1500)
        for half in (50, 125):
            f["rev_c%d" % half] = rolling_mean(rev.astype(np.float32), half) * half
        # how long the ball has held one x direction, and how sharply it is turning
        hold = np.zeros(T, np.int32)
        for t in range(1, T):
            hold[t] = hold[t - 1] + 1 if (sx[t] == sx[t - 1] and sx[t] != 0) else 0
        f["dir_hold"] = np.minimum(hold, 1500).astype(np.float32)
        dot = vxb * np.r_[np.full(25, np.nan), vxb[:-25]] + np.r_[
            np.full(25, np.nan), vyb[:-25]] * vyb
        dot = dot / np.maximum(np.hypot(vxb, vyb) * np.hypot(
            np.r_[np.full(25, np.nan), vxb[:-25]], np.r_[np.full(25, np.nan), vyb[:-25]]), 1e-6)
        f["ball_turn"] = np.clip(dot, -1.0, 1.0)
        for half in (12, 50):
            f["ball_turn_c%d" % half] = rolling_mean(f["ball_turn"], half)

    f["dmin_diff"] = f["a_dmin"] - f["h_dmin"]
    f["dmin_last_diff"] = f["a_dmin_last"] - f["h_dmin_last"]
    f["cx_diff"] = f["h_cx"] - f["a_cx"]
    f["spd_all"] = nanstat(np.nanmean, np.concatenate([home_spd, away_spd], 1), 1)
    f["spd_diff"] = f["h_spd"] - f["a_spd"]
    f["spread_diff"] = f["h_sx"] * f["h_sy"] - f["a_sx"] * f["a_sy"]

    # temporal context: centred and one-sided means of the strongest signals
    for key in ("spd_all", "h_spd", "a_spd", "dmin_diff", "dmin_last_diff", "cx_diff",
                "ball_v", "ball_present", "spread_diff", "h_n_fast", "a_n_fast",
                "h_advance", "a_advance", "h_n_still", "a_n_still", "ball_out") + (
                    ("ball_wander", "box_diff", "dlast_diff", "closing_diff")
                    if VERSION >= "f2" else ()):
        for half in (12, 50, 125, 250):
            f["{}_c{}".format(key, half)] = rolling_mean(f[key], half)
        for half in (50, 125):
            f["{}_p{}".format(key, half)] = rolling_mean(f[key], half, "past")
            f["{}_f{}".format(key, half)] = rolling_mean(f[key], half, "future")
    # how much the ball / players moved over 2 s (static restarts)
    names = list(f)
    X = np.stack([np.asarray(f[k], dtype=np.float32) for k in names], 1)
    X[np.isinf(X)] = np.nan
    return names, X
