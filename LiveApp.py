#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:04:53 2026

@author: antoinemaratray
"""
import io
import sys
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mplsoccer import Pitch
from pathlib import Path

import streamlit as st

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="HudlStatsbomb Live Match Analyser",
    page_icon="⚽",
    layout="wide",
)

# =========================
# xT grid (16x12)
# =========================
xT_GRID = np.array([
    [0.001,0.002,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.012,0.014,0.016,0.018,0.017,0.017],
    [0.002,0.003,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.013,0.016,0.018,0.021,0.020,0.021],
    [0.002,0.003,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.011,0.013,0.016,0.021,0.025,0.027,0.024],
    [0.002,0.003,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.011,0.014,0.018,0.024,0.029,0.039,0.031],
    [0.003,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.012,0.014,0.019,0.027,0.033,0.055,0.071],
    [0.004,0.004,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.012,0.014,0.019,0.033,0.077,0.142,0.332],
    [0.004,0.004,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.012,0.014,0.020,0.034,0.085,0.134,0.320],
    [0.004,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011,0.013,0.014,0.020,0.028,0.062,0.095,0.085],
    [0.003,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.012,0.013,0.018,0.025,0.035,0.042,0.033],
    [0.002,0.003,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011,0.016,0.021,0.026,0.026,0.022],
    [0.002,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.012,0.014,0.016,0.018,0.017,0.017],
    [0.001,0.002,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.012,0.014,0.016,0.018,0.017,0.017],
])

TOKEN_URL      = "https://live-api.statsbomb.com/v1/token"
LIVE_ENDPOINT  = "https://live-api.statsbomb.com/v1/graphql"
PLAYER_MAP_URL = "https://data.statsbomb.com/api/v1/player-mapping"

# =========================
# Tuning constants
# =========================
MIN_PASSES_PAIR          = 2
MIN_TOUCHES_NODE         = 3
NODE_SIZE_MIN            = 120
NODE_SIZE_MAX            = 1600
EDGE_WIDTH_MIN           = 0.6
EDGE_WIDTH_MAX           = 6.0
CARRY_MIN_DIST           = 5.0
CARRY_MIN_DX             = 2.0
CARRY_MAX_GAP_SECONDS    = 12
BACKWARD_DAMPING         = 0.25
SMALL_NEG_CLIP           = 0.002
PROGRESSIVE_DX           = 10.0
PROGRESSIVE_BONUS        = 1.35
NONPROG_POS_SCALE        = 1.00
SWING_WINDOW_MIN         = 3
SWING_MIN_XT_CREATED     = 0.08
SWING_MIN_GAP_MIN        = 8
SWING_MAX_LABELS_PER_TM  = 3
BINS                     = (16, 12)
SMOOTH_SIGMA             = 0.8
MIN_EVENTS_PER_CELL      = 2
MASK_QUANTILE            = 0.35
PERC_CLIP                = 98
GAMMA                    = 0.65
HEATMAP_ORIGIN           = "start"
DEF_EVENTS               = {"ball-recovery","block","clearance","dispossessed","duel","foul-committed","interception"}
DEF_BINS_X               = 6
DEF_BINS_Y               = 5
PITCH_X                  = 120.0
PITCH_Y                  = 80.0

# =========================
# GQL queries
# =========================
QUERY_COMP_SEASONS = """
query ExampleQuery {
  live_competition_season {
    season_name
    competition_name
    competition_id
    season_id
  }
}
"""

QUERY_MATCHES = """
query LiveMatches($competition_id: Int!, $season_id: Int!) {
  live_match(where: {
    competition_id: {_eq: $competition_id},
    season_id:      {_eq: $season_id}
  }) {
    match_id
    match_home_team_id
    match_home_team_name
    match_away_team_id
    match_away_team_name
  }
}
"""

QUERY_TEAMS = """
query LiveMatch($match_id: Int!) {
  live_match(where: {match_id: {_eq: $match_id}}) {
    match_home_team_id match_home_team_name
    match_away_team_id match_away_team_name
  }
}
"""

QUERY_EVENTS = """
query LiveMatchEvents($match_id: Int!, $m_start: smallint!, $m_end: smallint!) {
  live_match_event(
    where: {
      match_id: {_eq: $match_id},
      name: {_in: ["pass","ball-receipt","shot","ball-recovery","block","clearance",
                   "dispossessed","duel","foul-committed","interception"]},
      minute: {_gte: $m_start, _lte: $m_end}
    },
    order_by: {timestamp: asc},
    limit: 5000
  ) {
    id name team_id player_id recipient_id
    minute second period
    start_x start_y end_x end_y
    outcome xg goal_for goal_against timestamp
  }
}
"""

# =========================
# Session state init
# =========================
for _k, _v in {
    "live_token":    "",
    "match_data":    None,
    "cache_bust":    0,
    "last_fig":      None,
    # credential memory
    "saved_client_id":     "",
    "saved_client_secret": "",
    "saved_data_user":     "",
    "saved_data_pass":     "",
    # match-picker cache
    "comp_seasons":        None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# =========================
# Helpers
# =========================
def gql(query, variables, token):
    r = requests.post(
        LIVE_ENDPOINT,
        json={"query": query, "variables": variables},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data["data"]

def gql_no_vars(query, token):
    r = requests.post(
        LIVE_ENDPOINT,
        json={"query": query},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data["data"]

def _rot180_xy(x, y):
    return (PITCH_X - x, PITCH_Y - y)

def _to_display_xy(team_id, x, y, away_id):
    if team_id == away_id:
        return _rot180_xy(x, y)
    return x, y

def _last_name(name):
    if not isinstance(name, str) or not name:
        return "?"
    parts = name.split()
    return parts[-1] if parts else name

def normalize_outcome(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"complete","completed","success","successful"}:
        return "complete"
    if s in {"incomplete","incompleted","fail","failed","unsuccessful","unknown"}:
        return "incomplete"
    return s

def xt_lookup(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    cx = np.clip((xs * (16.0/120.0)).astype(int), 0, 15)
    cy = np.clip((ys * (12.0/80.0)).astype(int), 0, 11)
    return xT_GRID[cy, cx]

def shape_xt(raw_xt, dx):
    raw_xt = np.asarray(raw_xt, dtype=float)
    dx     = np.asarray(dx,     dtype=float)
    shaped = raw_xt.copy()
    shaped[(shaped < 0) & (np.abs(shaped) <= SMALL_NEG_CLIP)] = 0.0
    backward = dx < 0
    shaped[backward & (shaped < 0)] *= BACKWARD_DAMPING
    progressive = dx >= PROGRESSIVE_DX
    shaped[progressive  & (shaped > 0)] *= PROGRESSIVE_BONUS
    shaped[~progressive & (shaped > 0)] *= NONPROG_POS_SCALE
    return shaped

def _truthy(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    if isinstance(v, (int, np.integer, float, np.floating)):
        return v != 0
    return str(v).strip().lower() in {"1","true","t","yes","y"}

def _smooth_gaussian(A, sigma):
    if sigma <= 0:
        return A
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(A, sigma=sigma, mode="nearest")
    except Exception:
        rad = max(1, int(round(3*sigma)))
        x = np.arange(-rad, rad+1)
        k = np.exp(-(x**2)/(2*sigma**2)); k = (k/k.sum()).astype(float)
        A1 = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 0, A)
        return np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, A1)

# =========================
# Data fetching (cached)
# =========================
@st.cache_data(show_spinner=False)
def fetch_comp_seasons(token):
    data = gql_no_vars(QUERY_COMP_SEASONS, token)
    return data.get("live_competition_season", [])

@st.cache_data(show_spinner=False)
def fetch_matches(competition_id, season_id, token):
    data = gql(QUERY_MATCHES, {"competition_id": int(competition_id), "season_id": int(season_id)}, token)
    return data.get("live_match", [])

@st.cache_data(show_spinner=False)
def fetch_match_data(match_id, start_min, end_min, competition_id, season_id,
                     enable_carries, token, data_user, data_pass, _cache_bust=0):
    """Fetch and process all match data. Returns a dict of computed artefacts."""

    # --- Teams ---
    teams_res = gql(QUERY_TEAMS, {"match_id": int(match_id)}, token)
    lm = teams_res.get("live_match", [])
    if not lm:
        raise RuntimeError("Could not retrieve live_match info.")
    home_id   = lm[0]["match_home_team_id"];   home_name = lm[0]["match_home_team_name"]
    away_id   = lm[0]["match_away_team_id"];   away_name = lm[0]["match_away_team_name"]
    team_map  = {home_id: home_name, away_id: away_name}

    # --- Events ---
    events_res = gql(QUERY_EVENTS,
                     {"match_id": int(match_id),
                      "m_start": int(start_min),
                      "m_end":   int(end_min)},
                     token)
    events = events_res.get("live_match_event", [])
    if not events:
        raise RuntimeError(f"No events for match {match_id} in minutes {start_min}-{end_min}.")

    df = pd.DataFrame(events)
    req_cols = ["name","team_id","minute","period","start_x","start_y"]
    for c in req_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' from events.")

    df = df.dropna(subset=["start_x","start_y","minute","period","team_id"])
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce").fillna(0).astype(int)
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int)
    df["second"] = pd.to_numeric(df.get("second", 0), errors="coerce").fillna(0).astype(int)

    # --- Player mapping ---
    cache_path = Path(f"./player_map_comp{competition_id}_season{season_id}.csv")
    if cache_path.exists():
        mapping_df = pd.read_csv(cache_path)
    else:
        resp = requests.get(PLAYER_MAP_URL,
                            params={"competition-id": competition_id, "season-id": season_id},
                            auth=(data_user, data_pass), timeout=60)
        resp.raise_for_status()
        mapping_df = pd.DataFrame(resp.json())
        try:
            mapping_df.to_csv(cache_path, index=False)
        except Exception:
            pass

    if "live_player_id" not in mapping_df.columns:
        raise RuntimeError("Player Mapping response missing 'live_player_id'.")

    sort_cols = [c for c in ["most_recent_match_date","updated_at","created_at"] if c in mapping_df.columns]
    mapping_df = mapping_df.sort_values(["live_player_id"] + sort_cols, na_position="last")
    player_name_map = (mapping_df
                       .drop_duplicates("live_player_id", keep="last")
                       .set_index("live_player_id")["player_name"]
                       .to_dict())

    # --- Passes + xT ---
    df["outcome_norm"] = df.get("outcome", pd.Series([None]*len(df))).apply(normalize_outcome)
    passes = df[df["name"].eq("pass")].copy()
    passes["has_end"]   = passes["end_x"].notna() & passes["end_y"].notna()
    passes["completed"] = np.where(
        passes["outcome_norm"].isna(),
        passes["has_end"],
        passes["outcome_norm"].eq("complete")
    )
    passes["xT_start"] = xt_lookup(passes["start_x"].values, passes["start_y"].values)
    passes["xT_end"]   = 0.0
    mask_end = passes["completed"] & passes["end_x"].notna() & passes["end_y"].notna()
    passes.loc[mask_end, "xT_end"] = xt_lookup(passes.loc[mask_end,"end_x"].values,
                                                passes.loc[mask_end,"end_y"].values)
    passes["dx"]       = passes["end_x"].fillna(passes["start_x"]) - passes["start_x"]
    passes["xT_added"] = shape_xt((passes["xT_end"]-passes["xT_start"]).values, passes["dx"].values)

    # --- Carries ---
    carries = pd.DataFrame()
    if enable_carries:
        rec = df[df["name"].eq("ball-receipt")].copy().sort_values(["timestamp","minute","second"])
        rec["prev_player_id"] = rec["player_id"].shift(1)
        rec["prev_team_id"]   = rec["team_id"].shift(1)
        rec["prev_period"]    = rec["period"].shift(1)
        rec["prev_minute"]    = rec["minute"].shift(1)
        rec["prev_second"]    = rec["second"].shift(1)
        rec["prev_end_x"]     = rec["start_x"].shift(1)
        rec["prev_end_y"]     = rec["start_y"].shift(1)
        same_player = (rec["player_id"].eq(rec["prev_player_id"]) &
                       rec["team_id"].eq(rec["prev_team_id"]) &
                       rec["period"].eq(rec["prev_period"]))
        carries = rec.loc[same_player, ["team_id","player_id","minute","second","period",
                                        "prev_minute","prev_second","prev_end_x","prev_end_y",
                                        "start_x","start_y"]].rename(
            columns={"prev_end_x":"start_x","prev_end_y":"start_y","start_x":"end_x","start_y":"end_y"}
        ).copy()
        carries["gap_s"] = ((carries["minute"]*60 + carries["second"].fillna(0)) -
                            (carries["prev_minute"]*60 + carries["prev_second"].fillna(0)))
        carries = carries[(carries["gap_s"].notna()) &
                          (carries["gap_s"] >= 0) &
                          (carries["gap_s"] <= CARRY_MAX_GAP_SECONDS)].copy()
        carries["dx"]   = carries["end_x"] - carries["start_x"]
        carries["dist"] = np.hypot(carries["dx"], carries["end_y"]-carries["start_y"])
        carries = carries[carries["dist"] > CARRY_MIN_DIST].copy()
        if CARRY_MIN_DX > 0:
            carries = carries[carries["dx"] >= CARRY_MIN_DX].copy()
        if not carries.empty:
            carries["xT_start"] = xt_lookup(carries["start_x"].values, carries["start_y"].values)
            carries["xT_end"]   = xt_lookup(carries["end_x"].values,   carries["end_y"].values)
            carries["xT_added"] = shape_xt((carries["xT_end"]-carries["xT_start"]).values,
                                           carries["dx"].values)
            carries["name"] = "carry"; carries["completed"] = True
            carries["has_end"] = True; carries["recipient_id"] = pd.NA; carries["id"] = pd.NA

    # --- Build action table ---
    XT_COLS = ["id","recipient_id","name","team_id","player_id","minute","second","period",
               "start_x","start_y","end_x","end_y","xT_start","xT_end","xT_added","completed","has_end"]
    passes_xt  = passes.reindex(columns=XT_COLS)
    carries_xt = carries.reindex(columns=XT_COLS) if (enable_carries and not carries.empty) else pd.DataFrame(columns=XT_COLS)
    xt_actions = pd.concat([passes_xt, carries_xt], ignore_index=True)
    xt_actions["team"]        = xt_actions["team_id"].map(team_map).fillna(xt_actions["team_id"].astype(str))
    xt_actions["player_name"] = xt_actions["player_id"].apply(
        lambda v: player_name_map.get(int(v), f"ID {int(v)}") if pd.notna(v) else "Unknown"
    )
    xt_actions["xT_created"] = xt_actions["xT_added"].clip(lower=0)
    xt_actions["xT_lost"]    = (-xt_actions["xT_added"].clip(upper=0))

    # --- Aggregations ---
    team_minute = (xt_actions.groupby(["team","minute"], as_index=False)
                   .agg(xT_net=("xT_added","sum"), xT_created=("xT_created","sum"))
                   .sort_values(["team","minute"]))
    team_minute["cum_xT_net"]     = team_minute.groupby("team")["xT_net"].cumsum()
    team_minute["cum_xT_created"] = team_minute.groupby("team")["xT_created"].cumsum()

    player_totals = (xt_actions.groupby(["team_id","team","player_id","player_name"], as_index=False)
                     .agg(xT_created=("xT_created","sum"),
                          xT_lost=("xT_lost","sum"),
                          xT_net=("xT_added","sum"))
                     .sort_values(["xT_created","xT_net"], ascending=False))

    pass_inv    = df[df["name"].eq("pass")].groupby(["team_id","player_id"]).size().reset_index(name="pass_inv")
    receipt_inv = df[df["name"].eq("ball-receipt")].groupby(["team_id","player_id"]).size().reset_index(name="receipt_inv")
    carry_inv   = (carries_xt.groupby(["team_id","player_id"]).size().reset_index(name="carry_inv")
                   if not carries_xt.empty else pd.DataFrame(columns=["team_id","player_id","carry_inv"]))
    inv = (pass_inv.merge(receipt_inv, on=["team_id","player_id"], how="outer")
                   .merge(carry_inv,   on=["team_id","player_id"], how="outer"))
    for c in ["pass_inv","receipt_inv","carry_inv"]:
        inv[c] = inv[c].fillna(0).astype(int)
    inv["involvements"] = inv["pass_inv"] + inv["receipt_inv"] + inv["carry_inv"]
    player_totals = player_totals.merge(inv[["team_id","player_id","involvements"]],
                                        on=["team_id","player_id"], how="left")
    player_totals["involvements"] = player_totals["involvements"].fillna(0).astype(int)
    player_totals["xT_per_involvement"] = (player_totals["xT_created"] /
                                           player_totals["involvements"].clip(lower=1))

    # --- Shots + scoreline ---
    shots = df[df["name"] == "shot"].copy()
    shots["xg"] = pd.to_numeric(shots.get("xg", np.nan), errors="coerce").fillna(0.0)
    shots["team"] = shots["team_id"].map(team_map)
    shots["minute_f"] = shots["minute"].astype(float) + shots["second"].fillna(0)/60.0
    shots["is_goal"] = (shots.get("goal_for", 0).apply(_truthy) |
                        shots.get("outcome","").astype(str).str.lower().str.contains("goal", na=False))
    shots = shots.sort_values(["team_id","minute_f","timestamp"], na_position="last")
    shots["cum_xG"] = shots.groupby("team")["xg"].cumsum()

    xg_minute = (shots.groupby(["team","minute"], as_index=False)["xg"]
                 .sum().sort_values(["team","minute"]))
    xg_minute["cum_xG"] = xg_minute.groupby("team")["xg"].cumsum()

    goals_home = int(shots[(shots["team_id"] == home_id) & shots["is_goal"]].shape[0])
    goals_away = int(shots[(shots["team_id"] == away_id) & shots["is_goal"]].shape[0])

    return dict(
        df=df, passes=passes, carries_xt=carries_xt, xt_actions=xt_actions,
        team_minute=team_minute, player_totals=player_totals,
        shots=shots, xg_minute=xg_minute,
        goals_home=goals_home, goals_away=goals_away,
        home_id=home_id, home_name=home_name,
        away_id=away_id, away_name=away_name,
        player_name_map=player_name_map,
    )

# =========================
# Plotting helpers  (unchanged)
# =========================
def _hist_xt_and_counts(df_, team_id, away_id, xcol="end_x", ycol="end_y", bins=BINS):
    v = df_[df_[xcol].notna() & df_[ycol].notna()].copy()
    xedges = np.linspace(0, 120, bins[0]+1)
    yedges = np.linspace(0,  80, bins[1]+1)
    if v.empty:
        return np.zeros((bins[1], bins[0])), np.zeros((bins[1], bins[0])), xedges, yedges
    xs, ys = _to_display_xy(team_id, v[xcol].astype(float).values, v[ycol].astype(float).values, away_id)
    w = np.maximum(v["xT_added"].values, 0.0)
    H, xedges, yedges = np.histogram2d(xs, ys, bins=bins, range=[[0,120],[0,80]], weights=w)
    C, _, _ = np.histogram2d(xs, ys, bins=bins, range=[[0,120],[0,80]])
    return H.T, C.T, xedges, yedges

def prep_xt_heat(df_team, team_id, away_id):
    xcol, ycol = ("end_x","end_y") if HEATMAP_ORIGIN=="end" else ("start_x","start_y")
    H, C, xedges, yedges = _hist_xt_and_counts(df_team, team_id, away_id, xcol=xcol, ycol=ycol)
    Hs = _smooth_gaussian(H, SMOOTH_SIGMA)
    M  = np.ones_like(Hs, dtype=bool)
    if MIN_EVENTS_PER_CELL > 0:
        M &= (C >= MIN_EVENTS_PER_CELL)
    if MASK_QUANTILE and np.any(Hs > 0):
        M &= (Hs >= np.quantile(Hs[Hs > 0], MASK_QUANTILE))
    Hm = np.where(M, Hs, np.nan)
    vals = Hm[np.isfinite(Hm) & (Hm > 0)]
    vmax = np.percentile(vals, PERC_CLIP) if vals.size else 1.0
    norm = mcolors.PowerNorm(gamma=GAMMA, vmin=0, vmax=vmax)
    return Hm, xedges, yedges, norm

def build_pass_network(df, passes, player_name_map, team_id, away_id):
    team_pass_starts = df[(df["team_id"]==team_id) & (df["name"]=="pass")][["player_id","start_x","start_y"]]
    team_receipts    = df[(df["team_id"]==team_id) & (df["name"]=="ball-receipt")][["player_id","start_x","start_y"]]
    touches = pd.concat([team_pass_starts, team_receipts], ignore_index=True).dropna()
    if touches.empty:
        return pd.DataFrame(columns=["player_id","x","y","touches","player_name","label","size"]), \
               pd.DataFrame(columns=["player_id","recipient_id","passes","xT_sum","x_start","y_start","x_end","y_end","width","alpha"])
    nodes = (touches.groupby("player_id", as_index=False)
                    .agg(x=("start_x","mean"), y=("start_y","mean"), touches=("start_x","size")))
    nodes = nodes[nodes["touches"] >= MIN_TOUCHES_NODE].copy()
    nodes["player_name"] = nodes["player_id"].apply(lambda pid: player_name_map.get(int(pid), f"ID {int(pid)}"))
    nodes["label"] = nodes["player_name"].apply(_last_name)
    t = nodes["touches"].astype(float)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9)
    nodes["size"] = NODE_SIZE_MIN + t_norm * (NODE_SIZE_MAX - NODE_SIZE_MIN)

    p_team = passes[(passes["team_id"]==team_id) & passes["completed"] & passes["recipient_id"].notna()].copy()
    if p_team.empty:
        return nodes, pd.DataFrame(columns=["player_id","recipient_id","passes","xT_sum",
                                            "x_start","y_start","x_end","y_end","width","alpha"])
    edges = (p_team.groupby(["player_id","recipient_id"], as_index=False)
                   .agg(passes=("id","size"), xT_sum=("xT_added","sum")))
    edges = edges[edges["passes"] >= MIN_PASSES_PAIR].copy()
    pos = nodes.set_index("player_id")[["x","y"]]
    edges = (edges.join(pos, on="player_id").rename(columns={"x":"x_start","y":"y_start"})
                  .join(pos, on="recipient_id").rename(columns={"x":"x_end","y":"y_end"}))
    edges = edges.dropna(subset=["x_start","y_start","x_end","y_end"])
    if not edges.empty:
        w = edges["passes"].astype(float)
        edges["width"] = EDGE_WIDTH_MIN + (w-w.min())/(w.max()-w.min()+1e-9)*(EDGE_WIDTH_MAX-EDGE_WIDTH_MIN)
        a = edges["xT_sum"].astype(float)
        edges["alpha"] = 0.25 + 0.75*(a-a.min())/(a.max()-a.min()+1e-9)
    return nodes, edges

# =========================
# Draw functions  (unchanged)
# =========================
def draw_shot_map(ax, team_shots, team_name, team_id, away_id):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black", linewidth=1.25)
    pitch.draw(ax=ax)
    if not team_shots.empty:
        xs, ys = _to_display_xy(team_id,
                                team_shots["start_x"].astype(float).values,
                                team_shots["start_y"].astype(float).values, away_id)
        ng = ~team_shots["is_goal"].values; g = team_shots["is_goal"].values
        if np.any(ng):
            ax.scatter(xs[ng], ys[ng],
                       s=np.clip(team_shots.loc[ng,"xg"].values*1800, 25, 2200),
                       facecolors="none", edgecolors="black", linewidths=1.2, alpha=0.85, zorder=7)
        if np.any(g):
            ax.scatter(xs[g], ys[g],
                       s=np.clip(team_shots.loc[g,"xg"].values*2200, 60, 2600),
                       facecolors="#2ecc71", edgecolors="white", linewidths=1.2, alpha=0.95, zorder=8)
    ax.set_title(f"{team_name} | Shot Map")

def draw_pass_network_ax(ax, df, passes, player_name_map, team_id, team_name, away_id, start_min, end_min):
    nodes, edges = build_pass_network(df, passes, player_name_map, team_id, away_id)
    pitch = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black", linewidth=1.25)
    pitch.draw(ax=ax)
    if not edges.empty:
        xs1, ys1 = _to_display_xy(team_id, edges["x_start"].astype(float).values, edges["y_start"].astype(float).values, away_id)
        xs2, ys2 = _to_display_xy(team_id, edges["x_end"].astype(float).values,   edges["y_end"].astype(float).values,   away_id)
        for i, e in enumerate(edges.itertuples(index=False)):
            pitch.lines(xs1[i], ys1[i], xs2[i], ys2[i],
                        ax=ax, color="black", lw=float(e.width), alpha=float(e.alpha), zorder=6)
    if not nodes.empty:
        xn, yn = _to_display_xy(team_id, nodes["x"].astype(float).values, nodes["y"].astype(float).values, away_id)
        ax.scatter(xn, yn, s=nodes["size"].values*1.2,
                   facecolors="white", edgecolors="black", linewidths=1.2, zorder=7)
        for i, r in enumerate(nodes.itertuples(index=False)):
            ax.text(xn[i], yn[i], r.label, ha="center", va="center", fontsize=10, zorder=8)
    ax.set_title(f"{team_name} | Pass Network ({start_min}'–{end_min}')")

def draw_xt_map(ax, H, xedges, yedges, norm, title):
    pitch_bg    = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black", linewidth=1.25)
    pitch_lines = Pitch(pitch_type="statsbomb", pitch_color="none",  line_color="black", linewidth=1.25)
    cmap = plt.cm.Reds.copy(); cmap.set_bad(color="white", alpha=0.0)
    pitch_bg.draw(ax=ax)
    ax.pcolormesh(xedges, yedges, H, shading="auto", cmap=cmap, norm=norm, zorder=0)
    pitch_lines.draw(ax=ax)
    ax.set_title(title)

def draw_defensive_grid(ax, df, team_id, team_name, away_id):
    d = df[(df["team_id"]==team_id) & (df["name"].isin(DEF_EVENTS))].dropna(subset=["start_x","start_y"])
    pitch_bg    = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="black", linewidth=1.25)
    pitch_lines = Pitch(pitch_type="statsbomb", pitch_color="none",  line_color="black", linewidth=1.25)
    pitch_bg.draw(ax=ax)
    ax.set_title(f"{team_name} | Defensive activity")
    if d.empty:
        return
    x, y = _to_display_xy(team_id, d["start_x"].astype(float).clip(0, 119.9).values,
                           d["start_y"].astype(float).clip(0, 79.9).values, away_id)
    H_counts, xedges, yedges = np.histogram2d(x, y, bins=[DEF_BINS_X, DEF_BINS_Y], range=[[0,120],[0,80]])
    H_counts = H_counts.T
    mx = float(np.nanmax(H_counts)) if np.any(H_counts > 0) else 1.0
    Z  = H_counts / mx
    cmap = plt.cm.Reds.copy(); cmap.set_under("white")
    ax.pcolormesh(xedges, yedges, Z, shading="auto", cmap=cmap, vmin=0.0001, vmax=1.0, alpha=0.92, zorder=0)
    pitch_lines.draw(ax=ax)
    xc = (xedges[:-1]+xedges[1:])/2; yc = (yedges[:-1]+yedges[1:])/2
    for iy, ycc in enumerate(yc):
        for ix, xcc in enumerate(xc):
            c = int(round(H_counts[iy, ix]))
            if c > 0:
                ax.text(xcc, ycc, str(c), ha="center", va="center", fontsize=11, fontweight="bold",
                        color=("white" if Z[iy,ix]>=0.45 else "black"), zorder=50)

def draw_bar_top_xt_created(ax, df_team, title, n=8):
    top = df_team.sort_values("xT_created", ascending=False).head(n)
    if top.empty:
        ax.axis("off"); ax.set_title(title); return
    labels = [_last_name(nm) for nm in top["player_name"].values]
    ax.barh(range(len(top))[::-1], top["xT_created"].values[::-1], color="black", alpha=0.85)
    ax.set_yticks(range(len(top))[::-1], labels[::-1], fontsize=9)
    ax.set_xlabel("xT created"); ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlim(0, float(top["xT_created"].max())*1.10+1e-9)

def draw_bar_top_xt_per_inv(ax, df_team, title, n=8):
    d = df_team.sort_values("xT_per_involvement", ascending=False).head(n)
    if d.empty:
        ax.axis("off"); ax.set_title(title); return
    labels = [_last_name(nm) for nm in d["player_name"].values]
    vals   = d["xT_per_involvement"].values.astype(float)
    invs   = d["involvements"].values.astype(int)
    y_pos  = np.arange(len(d))[::-1]
    ax.barh(y_pos, vals[::-1], color="black", alpha=0.85)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("xT created / involvement"); ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    xmax = float(vals.max())*1.10+1e-9; ax.set_xlim(0, xmax)
    x0, x1 = ax.get_xlim(); span = (x1-x0) if (x1-x0)>0 else 1.0
    for yi, v, inv_ in zip(y_pos, vals[::-1], invs[::-1]):
        ax.text(x0+0.012*span, yi, f"{int(inv_)}", va="center", ha="left",
                fontsize=12, fontweight="bold",
                color=("white" if v>=(x0+0.012*span-x0) else "black"), zorder=50, clip_on=False)

def find_swings(team_minute, team_name):
    d = team_minute[team_minute["team"]==team_name][["minute","xT_created"]].copy().sort_values("minute")
    if d.empty: return []
    d["roll"] = d["xT_created"].rolling(SWING_WINDOW_MIN, min_periods=1).sum()
    cand = d[d["roll"]>=SWING_MIN_XT_CREATED].sort_values("roll", ascending=False)
    chosen = []
    for _, r in cand.iterrows():
        m = int(r["minute"]); roll = float(r["roll"])
        if all(abs(m-mm)>=SWING_MIN_GAP_MIN for mm, _ in chosen):
            chosen.append((m, roll))
        if len(chosen) >= SWING_MAX_LABELS_PER_TM:
            break
    return sorted(chosen, key=lambda x: x[0])

def swing_contributor(xt_actions, team_id, minute_center, start_min, end_min):
    m0 = max(start_min, minute_center-(SWING_WINDOW_MIN-1)); m1 = min(end_min, minute_center)
    w = xt_actions[(xt_actions["team_id"]==team_id) &
                   (xt_actions["minute"]>=m0) & (xt_actions["minute"]<=m1)]
    if w.empty: return "?"
    g = (w.groupby(["player_id","player_name"], as_index=False)["xT_created"]
          .sum().sort_values("xT_created", ascending=False))
    return _last_name(g.iloc[0]["player_name"]) if not g.empty else "?"

def annotate_swings(ax, team_minute, xt_actions, team_id, team_name, start_min, end_min):
    d = team_minute[team_minute["team"]==team_name][["minute","cum_xT_created"]].copy()
    if d.empty: return
    y_map  = dict(zip(d["minute"].astype(int), d["cum_xT_created"].astype(float)))
    swings = find_swings(team_minute, team_name)
    if not swings: return
    y0, y1 = ax.get_ylim(); ypad = 0.04*(y1-y0+1e-9)
    for m, roll in swings:
        who = swing_contributor(xt_actions, team_id, m, start_min, end_min)
        y   = y_map.get(int(m))
        if y is None: continue
        ax.axvline(m, color="black", alpha=0.10, linewidth=1.0, zorder=1)
        ax.text(m+0.6, y+ypad, f"{m}' {who} +{roll:.2f}", fontsize=8, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75), zorder=20)

# =========================
# Build full figure  (unchanged)
# =========================
def build_figure(d, start_min, end_min):
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"]   = "white"

    home_id = d["home_id"]; home_name = d["home_name"]
    away_id = d["away_id"]; away_name = d["away_name"]
    df = d["df"]; passes = d["passes"]; xt_actions = d["xt_actions"]
    team_minute = d["team_minute"]; player_totals = d["player_totals"]
    shots = d["shots"]; xg_minute = d["xg_minute"]
    goals_home = d["goals_home"]; goals_away = d["goals_away"]
    player_name_map = d["player_name_map"]

    shots_home = shots[shots["team_id"]==home_id]
    shots_away = shots[shots["team_id"]==away_id]
    goals_home_df = shots_home[shots_home["is_goal"]]
    goals_away_df = shots_away[shots_away["is_goal"]]

    home_actions = xt_actions[xt_actions["team_id"]==home_id]
    away_actions = xt_actions[xt_actions["team_id"]==away_id]
    H_home, xeh_h, yeh_h, norm_home = prep_xt_heat(home_actions, home_id, away_id)
    H_away, xeh_a, yeh_a, norm_away = prep_xt_heat(away_actions, away_id, away_id)
    pt_home = player_totals[player_totals["team"]==home_name].copy()
    pt_away = player_totals[player_totals["team"]==away_name].copy()

    fig = plt.figure(figsize=(18, 29))
    gs  = gridspec.GridSpec(6, 1, height_ratios=[1.2, 1.6, 1.55, 1.2, 1.0, 1.0], hspace=0.38)
    title_txt = f"{home_name} {goals_home} - {goals_away} {away_name}\n→                                                                ←"
    fig.suptitle(title_txt, fontsize=45, fontweight="bold", y=0.94)

    # Row 1 — Shot maps + xG race
    row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.20)
    ax_sm_home = fig.add_subplot(row1[0, 0])
    ax_xg      = fig.add_subplot(row1[0, 1])
    ax_sm_away = fig.add_subplot(row1[0, 2])
    draw_shot_map(ax_sm_home, shots_home, home_name, home_id, away_id)
    draw_shot_map(ax_sm_away, shots_away, away_name, away_id, away_id)
    for tm in [home_name, away_name]:
        d_ = xg_minute[xg_minute["team"]==tm]
        if not d_.empty:
            ax_xg.step(d_["minute"], d_["cum_xG"], where="post", label=tm)
    for gdf in [goals_home_df, goals_away_df]:
        if not gdf.empty:
            ax_xg.scatter(gdf["minute_f"], gdf["cum_xG"],
                          s=90, facecolors="#2ecc71", edgecolors="black", linewidths=1.0, zorder=10)
            for m in gdf["minute_f"]:
                ax_xg.axvline(m, color="#2ecc71", linestyle="-", alpha=0.25, linewidth=1.0, zorder=5)
    ax_xg.set_xlabel("Minute"); ax_xg.set_ylabel("Cumulative xG")
    ax_xg.set_title("xG Race"); ax_xg.grid(True, alpha=0.3); ax_xg.legend()

    # Row 2 — Pass networks
    row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.12)
    ax_net_home = fig.add_subplot(row2[0, 0]); ax_net_away = fig.add_subplot(row2[0, 1])
    draw_pass_network_ax(ax_net_home, df, passes, player_name_map, home_id, home_name, away_id, start_min, end_min)
    draw_pass_network_ax(ax_net_away, df, passes, player_name_map, away_id, away_name, away_id, start_min, end_min)

    # Row 3 — Defensive activity
    row3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.15)
    ax_def_home = fig.add_subplot(row3[0, 0]); ax_def_away = fig.add_subplot(row3[0, 1])
    draw_defensive_grid(ax_def_home, df, home_id, home_name, away_id)
    draw_defensive_grid(ax_def_away, df, away_id, away_name, away_id)

    # Row 4 — xT maps + race
    row4 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3], wspace=0.20)
    ax_xt_home = fig.add_subplot(row4[0, 0]); ax_xt_race = fig.add_subplot(row4[0, 1]); ax_xt_away = fig.add_subplot(row4[0, 2])
    draw_xt_map(ax_xt_home, H_home, xeh_h, yeh_h, norm_home, f"{home_name} | xT Map")
    draw_xt_map(ax_xt_away, H_away, xeh_a, yeh_a, norm_away, f"{away_name} | xT Map")
    for tm in [home_name, away_name]:
        d_ = team_minute[team_minute["team"]==tm]
        if not d_.empty:
            ax_xt_race.step(d_["minute"], d_["cum_xT_created"], where="post", label=tm)
    ax_xt_race.set_xlabel("Minute"); ax_xt_race.set_ylabel("Cumulative xT created")
    ax_xt_race.set_title(f"xT Race | Threshold > {SWING_MIN_XT_CREATED:.2f} over {SWING_WINDOW_MIN}m")
    ax_xt_race.grid(True, alpha=0.3); ax_xt_race.legend(loc="upper left")
    annotate_swings(ax_xt_race, team_minute, xt_actions, home_id, home_name, start_min, end_min)
    annotate_swings(ax_xt_race, team_minute, xt_actions, away_id, away_name, start_min, end_min)

    # Row 5 — Top xT created
    row5 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[4], wspace=0.25)
    ax_top_home = fig.add_subplot(row5[0, 0]); ax_top_away = fig.add_subplot(row5[0, 1])
    draw_bar_top_xt_created(ax_top_home, pt_home, f"{home_name} | Top players by xT created", n=10)
    draw_bar_top_xt_created(ax_top_away, pt_away, f"{away_name} | Top players by xT created", n=10)

    # Row 6 — xT per involvement
    row6 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[5], wspace=0.25)
    ax_eff_home = fig.add_subplot(row6[0, 0]); ax_eff_away = fig.add_subplot(row6[0, 1])
    draw_bar_top_xt_per_inv(ax_eff_home, pt_home, f"{home_name} | Player xT per involvement", n=10)
    draw_bar_top_xt_per_inv(ax_eff_away, pt_away, f"{away_name} | Player xT per involvement", n=10)

    fig.text(0.5, 0.08, "by Antoine Maratray | antoine.maratray@hudl.com",
             ha="center", fontsize=17, fontweight="bold")
    return fig

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.title("⚽ Match Analyser")

    # ── Authentication ──────────────────────────────────────────
    st.header("🔐 Authentication")

    remember_auth = st.checkbox(
        "Remember credentials this session",
        value=bool(st.session_state["saved_client_id"]),
        key="remember_auth",
    )

    client_id = st.text_input(
        "Client ID",
        value=st.session_state["saved_client_id"],
        type="password",
        key="input_client_id",
    )
    client_secret = st.text_input(
        "Client Secret",
        value=st.session_state["saved_client_secret"],
        type="password",
        key="input_client_secret",
    )

    fetch_token_btn = st.button("Fetch Token", use_container_width=True)
    if fetch_token_btn:
        if not client_id or not client_secret:
            st.error("Enter both Client ID and Client Secret.")
        else:
            with st.spinner("Fetching token…"):
                try:
                    r = requests.post(
                        TOKEN_URL,
                        json={"client_id": client_id, "client_secret": client_secret},
                        headers={"content-type": "application/json"},
                        timeout=15,
                    )
                    r.raise_for_status()
                    tok = r.json().get("access_token", "")
                    if tok:
                        st.session_state["live_token"] = tok
                        # Save credentials if checkbox is ticked
                        if remember_auth:
                            st.session_state["saved_client_id"]     = client_id
                            st.session_state["saved_client_secret"] = client_secret
                        else:
                            st.session_state["saved_client_id"]     = ""
                            st.session_state["saved_client_secret"] = ""
                        st.success("✅ Token acquired (valid 24 h)")
                        # Pre-fetch competition/season list right away
                        st.session_state["comp_seasons"] = None
                    else:
                        st.error("No access_token in response.")
                except Exception as e:
                    st.error(f"Token fetch failed: {e}")

    token_status = "✅ Token loaded" if st.session_state.get("live_token") else "⚠️ No token"
    st.caption(token_status)

    st.divider()

    # ── Data API credentials ─────────────────────────────────────
    st.header("📊 Data API Credentials")

    remember_data = st.checkbox(
        "Remember credentials this session",
        value=bool(st.session_state["saved_data_user"]),
        key="remember_data",
    )

    data_user = st.text_input(
        "Username",
        value=st.session_state["saved_data_user"],
        key="data_user",
    )
    data_pass = st.text_input(
        "Password",
        value=st.session_state["saved_data_pass"],
        type="password",
        key="data_pass",
    )

    # Persist data creds when either field changes
    if remember_data:
        st.session_state["saved_data_user"] = data_user
        st.session_state["saved_data_pass"] = data_pass
    else:
        # Only wipe if the user explicitly unchecked
        if not st.session_state.get("remember_data"):
            st.session_state["saved_data_user"] = ""
            st.session_state["saved_data_pass"] = ""

    st.divider()

    # ── Match picker ─────────────────────────────────────────────
    st.header("🏆 Match Picker")

    token = st.session_state.get("live_token", "")
    match_id         = None
    competition_id   = None
    season_id        = None

    if not token:
        st.info("Fetch a token above to enable the match picker.")
    else:
        # Load competition/season catalogue once per token
        if st.session_state["comp_seasons"] is None:
            with st.spinner("Loading competitions…"):
                try:
                    st.session_state["comp_seasons"] = fetch_comp_seasons(token)
                except Exception as e:
                    st.error(f"Failed to load competitions: {e}")

        cs = st.session_state.get("comp_seasons") or []

        if cs:
            cs_df = pd.DataFrame(cs)

            # 1️⃣  Season
            seasons_sorted = sorted(cs_df["season_name"].unique(), reverse=True)
            sel_season = st.selectbox("Season", seasons_sorted)

            # 2️⃣  Competition (filtered by season)
            # Build unique labels — append competition_id in brackets if name clashes
            season_rows = cs_df[cs_df["season_name"] == sel_season].copy()
            name_counts = season_rows["competition_name"].value_counts()
            def _comp_label(r):
                if name_counts[r["competition_name"]] > 1:
                    return f"{r['competition_name']} (ID {r['competition_id']})"
                return r["competition_name"]
            season_rows["comp_label"] = season_rows.apply(_comp_label, axis=1)
            comp_labels = sorted(season_rows["comp_label"].unique())
            sel_comp_label = st.selectbox("Competition", comp_labels)

            # Resolve IDs from the unique label
            row = season_rows[season_rows["comp_label"] == sel_comp_label].iloc[0]
            competition_id = int(row["competition_id"])
            season_id      = int(row["season_id"])

            # 3️⃣  Load matches for this comp/season
            try:
                matches_raw = fetch_matches(competition_id, season_id, token)
            except Exception as e:
                st.error(f"Failed to load matches: {e}")
                matches_raw = []

            if matches_raw:
                m_df = pd.DataFrame(matches_raw)

                # 4️⃣  Home team
                home_teams = sorted(m_df["match_home_team_name"].unique())
                sel_home = st.selectbox("Home team", home_teams)

                # 5️⃣  Away team (only those who have played home vs sel_home)
                away_teams = sorted(
                    m_df[m_df["match_home_team_name"] == sel_home]["match_away_team_name"].unique()
                )
                if away_teams:
                    sel_away = st.selectbox("Away team", away_teams)
                    # Resolve match_id
                    matched = m_df[
                        (m_df["match_home_team_name"] == sel_home) &
                        (m_df["match_away_team_name"] == sel_away)
                    ]
                    if not matched.empty:
                        match_id = int(matched.iloc[0]["match_id"])
                        st.caption(f"Match ID: **{match_id}**")
                else:
                    st.warning("No matches found for this home team.")
            else:
                st.warning("No matches found for this competition/season.")

    st.divider()

    # ── Match settings ───────────────────────────────────────────
    st.header("⚙️ Match Settings")
    col1, col2 = st.columns(2)
    start_min      = col1.number_input("Start min", value=0,   min_value=0, max_value=200)
    end_min        = col2.number_input("End min",   value=94,  min_value=0, max_value=200)
    enable_carries = st.checkbox("Enable Carries", value=True)

    st.divider()
    run_btn     = st.button("▶ Run Analysis",  use_container_width=True, type="primary")
    refresh_btn = st.button("🔄 Refresh Data", use_container_width=True)

# =========================
# Main panel
# =========================
st.title("⚽ HudlStatsbomb Live Match Analyser")

if run_btn or refresh_btn:
    token = st.session_state.get("live_token", "")
    if not token:
        st.error("No bearer token — use 'Fetch Token' in the sidebar first.")
    elif match_id is None:
        st.error("Please select a match using the Match Picker in the sidebar.")
    elif not data_user or not data_pass:
        st.error("Enter your StatsBomb Data API credentials in the sidebar.")
    else:
        if refresh_btn:
            st.session_state["cache_bust"] += 1

        with st.spinner("Fetching match data…"):
            try:
                result = fetch_match_data(
                    match_id=match_id,
                    start_min=start_min,
                    end_min=end_min,
                    competition_id=competition_id,
                    season_id=season_id,
                    enable_carries=enable_carries,
                    token=token,
                    data_user=data_user,
                    data_pass=data_pass,
                    _cache_bust=st.session_state["cache_bust"],
                )
                st.session_state["match_data"] = result
            except Exception as e:
                st.error(f"Data fetch error: {e}")
                st.stop()

        with st.spinner("Building dashboard…"):
            fig = build_figure(st.session_state["match_data"], start_min, end_min)
            st.session_state["last_fig"] = fig

if st.session_state.get("last_fig") is not None:
    d = st.session_state["match_data"]
    st.subheader(f"{d['home_name']} {d['goals_home']} – {d['goals_away']} {d['away_name']}")
    st.pyplot(st.session_state["last_fig"], use_container_width=True)

    buf = io.BytesIO()
    st.session_state["last_fig"].savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="📥 Download dashboard as PNG",
        data=buf,
        file_name=f"match_{match_id}_{start_min}-{end_min}min.png",
        mime="image/png",
    )
else:
    st.info("Configure settings in the sidebar and click **▶ Run Analysis** to generate the dashboard.")
