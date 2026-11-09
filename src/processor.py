"""Data processing and analytical calculations for players, clubs, and gameweek opponents."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def build_teams_lookup(bootstrap: Dict[str, Any]) -> pd.DataFrame:
    """Extract base club metadata and official FPL strength ratings."""
    teams = pd.DataFrame(bootstrap.get("teams", []))
    if teams.empty:
        return pd.DataFrame()
    cols = [
        "id",
        "code",
        "name",
        "short_name",
        "strength",
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ]
    existing_cols = [c for c in cols if c in teams.columns]
    out = teams[existing_cols].copy()
    if "code" in out.columns:
        out["crest_url"] = (
            "https://resources.premierleague.com/premierleague/badges/70/t"
            + out["code"].astype(str)
            + ".png"
        )
    else:
        out["crest_url"] = ""
    return out


def build_players_df(bootstrap: Dict[str, Any]) -> pd.DataFrame:
    """Build enriched master player DataFrame with per-90 and value metrics."""
    elements = pd.DataFrame(bootstrap.get("elements", []))
    teams = build_teams_lookup(bootstrap)
    if elements.empty or teams.empty:
        return pd.DataFrame()

    team_name_map = dict(zip(teams["id"], teams["name"]))
    team_short_map = dict(zip(teams["id"], teams["short_name"]))
    team_crest_map = dict(zip(teams["id"], teams.get("crest_url", "")))

    df = elements.copy()
    df["team_name"] = df["team"].map(team_name_map)
    df["team_short"] = df["team"].map(team_short_map)
    df["crest_url"] = df["team"].map(team_crest_map)
    df["position"] = df["element_type"].map(POSITION_MAP)
    df["price"] = df["now_cost"] / 10.0
    if "code" in df.columns:
        df["photo_url"] = (
            "https://resources.premierleague.com/premierleague/photos/players/110x140/p"
            + df["code"].astype(str)
            + ".png"
        )
    else:
        df["photo_url"] = ""

    float_cols = [
        "selected_by_percent",
        "form",
        "points_per_game",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "ict_index",
        "influence",
        "creativity",
        "threat",
        "value_form",
        "value_season",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    int_cols = [
        "minutes",
        "total_points",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "bonus",
        "bps",
        "saves",
        "starts",
        "yellow_cards",
        "red_cards",
        "transfers_in_event",
        "transfers_out_event",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df.rename(
        columns={
            "expected_goals": "xG",
            "expected_assists": "xA",
            "expected_goal_involvements": "xGI",
            "expected_goals_conceded": "xGC",
        },
        inplace=True,
    )

    nineties = (df["minutes"] / 90.0).replace(0, np.nan)
    df["nineties"] = (df["minutes"] / 90.0).round(2)
    df["xG_per_90"] = (df["xG"] / nineties).fillna(0.0).round(2)
    df["xA_per_90"] = (df["xA"] / nineties).fillna(0.0).round(2)
    df["xGI_per_90"] = (df["xGI"] / nineties).fillna(0.0).round(2)
    df["xGC_per_90"] = (df["xGC"] / nineties).fillna(0.0).round(2)
    df["pts_per_90"] = (df["total_points"] / nineties).fillna(0.0).round(2)
    df["bps_per_90"] = (df["bps"] / nineties).fillna(0.0).round(1)

    df["goal_involvements"] = df["goals_scored"] + df["assists"]
    df["xGI_delta"] = (df["goal_involvements"] - df["xGI"]).round(2)
    df["G_minus_xG"] = (df["goals_scored"] - df["xG"]).round(2)
    df["points_per_million"] = (df["total_points"] / df["price"].replace(0, np.nan)).fillna(0.0).round(2)
    df["net_transfers_gw"] = df["transfers_in_event"] - df["transfers_out_event"]
    df["transfer_velocity"] = (
        df["net_transfers_gw"] / df["selected_by_percent"].clip(lower=1.0)
    ).round(0)
    df["price_trend"] = np.where(
        (df["net_transfers_gw"] >= 45000)
        | ((df["net_transfers_gw"] >= 25000) & (df["transfer_velocity"] >= 4000)),
        "📈 Likely Rise (+£0.1m)",
        np.where(
            df["net_transfers_gw"] >= 15000,
            "↗️ Trending Up",
            np.where(
                (df["net_transfers_gw"] <= -35000)
                | ((df["net_transfers_gw"] <= -15000) & (df["transfer_velocity"] <= -3500)),
                "📉 Likely Fall (-£0.1m)",
                np.where(df["net_transfers_gw"] <= -10000, "↘️ Trending Down", "➖ Stable"),
            ),
        ),
    )

    # Set-piece and penalty taker orders & weighting
    pen_ord = pd.to_numeric(df.get("penalties_order"), errors="coerce").fillna(0).astype(int)
    crn_ord = pd.to_numeric(df.get("corners_and_indirect_freekicks_order"), errors="coerce").fillna(0).astype(int)
    fk_ord = pd.to_numeric(df.get("direct_freekicks_order"), errors="coerce").fillna(0).astype(int)

    df["penalties_order"] = pen_ord
    df["corners_order"] = crn_ord
    df["freekicks_order"] = fk_ord

    sp_badges = []
    for p_o, c_o, f_o in zip(pen_ord, crn_ord, fk_ord):
        tags = []
        if p_o in (1, 2):
            tags.append(f"PEN #{p_o}")
        if c_o in (1, 2):
            tags.append(f"CRN #{c_o}")
        if f_o in (1, 2):
            tags.append(f"FK #{f_o}")
        sp_badges.append(", ".join(tags) if tags else "-")
    df["set_piece_duty"] = sp_badges

    df["sp_multiplier"] = (
        1.0
        + np.where(pen_ord == 1, 0.08, np.where(pen_ord == 2, 0.03, 0.0))
        + np.where(crn_ord == 1, 0.04, np.where(crn_ord == 2, 0.015, 0.0))
        + np.where(fk_ord == 1, 0.02, 0.0)
    ).round(3)

    return df


def build_club_and_opponent_stats(
    bootstrap: Dict[str, Any],
    fixtures: List[Dict[str, Any]],
    players_df: pd.DataFrame,
    rolling_window: int = 5,
    elo_map: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Aggregate club performance, home/away splits, rolling form, Elo, and opponent vulnerability."""
    teams = build_teams_lookup(bootstrap)
    if teams.empty:
        return pd.DataFrame()

    elo_lookup = elo_map or {}
    fix_df = pd.DataFrame(fixtures)
    finished = fix_df[fix_df["finished"] == True].copy() if not fix_df.empty else pd.DataFrame()

    # Estimate team xG from player xG sum, and team xGC from minutes-normalized Goalkeeper xGC
    team_xg = players_df.groupby("team")["xG"].sum().to_dict() if not players_df.empty else {}
    if not players_df.empty:
        gkps = players_df[players_df["position"] == "GKP"]
        gkp_xgc_sum = gkps.groupby("team")["xGC"].sum().to_dict()
        gkp_mins_sum = gkps.groupby("team")["minutes"].sum().to_dict()
    else:
        gkp_xgc_sum, gkp_mins_sum = {}, {}

    records = []
    for _, team in teams.iterrows():
        tid = int(team["id"])
        t_name = team["name"]
        t_short = team["short_name"]

        if not finished.empty:
            home_m = finished[finished["team_h"] == tid]
            away_m = finished[finished["team_a"] == tid]

            mp_h, mp_a = len(home_m), len(away_m)
            mp = mp_h + mp_a

            gf_h = int(home_m["team_h_score"].sum()) if mp_h else 0
            ga_h = int(home_m["team_a_score"].sum()) if mp_h else 0
            gf_a = int(away_m["team_a_score"].sum()) if mp_a else 0
            ga_a = int(away_m["team_h_score"].sum()) if mp_a else 0

            gf = gf_h + gf_a
            ga = ga_h + ga_a

            cs_h = int((home_m["team_a_score"] == 0).sum()) if mp_h else 0
            cs_a = int((away_m["team_h_score"] == 0).sum()) if mp_a else 0
            cs = cs_h + cs_a

            failed_to_score = (
                int((home_m["team_h_score"] == 0).sum()) + int((away_m["team_a_score"] == 0).sum())
            )

            # Build chronological match log for rolling window
            h_log = pd.DataFrame(
                {
                    "event": home_m["event"],
                    "kickoff_time": home_m["kickoff_time"],
                    "gf": home_m["team_h_score"],
                    "ga": home_m["team_a_score"],
                }
            )
            a_log = pd.DataFrame(
                {
                    "event": away_m["event"],
                    "kickoff_time": away_m["kickoff_time"],
                    "gf": away_m["team_a_score"],
                    "ga": away_m["team_h_score"],
                }
            )
            match_log = pd.concat([h_log, a_log], ignore_index=True).sort_values("kickoff_time")
            match_log["pts"] = np.where(
                match_log["gf"] > match_log["ga"],
                3,
                np.where(match_log["gf"] == match_log["ga"], 1, 0),
            )
            match_log["result_char"] = np.where(
                match_log["gf"] > match_log["ga"],
                "W",
                np.where(match_log["gf"] == match_log["ga"], "D", "L"),
            )

            pts = int(match_log["pts"].sum())
            wins = int((match_log["pts"] == 3).sum())
            draws = int((match_log["pts"] == 1).sum())
            losses = int((match_log["pts"] == 0).sum())

            recent = match_log.tail(rolling_window)
            r_mp = len(recent)
            r_gf = float(recent["gf"].sum()) if r_mp else 0.0
            r_ga = float(recent["ga"].sum()) if r_mp else 0.0
            r_cs = int((recent["ga"] == 0).sum()) if r_mp else 0
            r_pts = int(recent["pts"].sum()) if r_mp else 0
            form_str = "".join(recent["result_char"].tolist())
        else:
            mp = mp_h = mp_a = gf = ga = gf_h = ga_h = gf_a = ga_a = cs = cs_h = cs_a = 0
            failed_to_score = pts = wins = draws = losses = r_mp = r_cs = r_pts = 0
            r_gf = r_ga = 0.0
            form_str = "-"

        xg_tot = round(float(team_xg.get(tid, 0.0)), 2)
        raw_gkp_xgc = float(gkp_xgc_sum.get(tid, 0.0))
        raw_gkp_mins = float(gkp_mins_sum.get(tid, 0.0))
        denom = max(mp, 1)
        r_denom = max(r_mp, 1)
        if raw_gkp_mins >= 90.0 and mp > 0:
            xgc_per_m = round((raw_gkp_xgc / (raw_gkp_mins / 90.0)), 2)
            xgc_tot = round(xgc_per_m * mp, 2)
        else:
            xgc_tot = round(raw_gkp_xgc if raw_gkp_xgc > 0 else float(ga), 2)
            xgc_per_m = round(xgc_tot / denom, 2)

        # ClubElo rating (with fallback derived from official FPL overall strength)
        fpl_str_avg = 0.5 * (
            float(team.get("strength_overall_home", 1150)) + float(team.get("strength_overall_away", 1150))
        )
        fallback_elo = round(1700.0 + (fpl_str_avg - 1150.0) * 1.15, 1)
        club_elo = float(elo_lookup.get(t_short, fallback_elo))

        records.append(
            {
                "team_id": tid,
                "crest_url": team.get("crest_url", ""),
                "team_name": t_name,
                "team_short": t_short,
                "Elo": club_elo,
                "MP": mp,
                "W": wins,
                "D": draws,
                "L": losses,
                "Pts": pts,
                "GF": gf,
                "GA": ga,
                "GD": gf - ga,
                "CS": cs,
                "FTS": failed_to_score,
                "GF_per_match": round(gf / denom, 2),
                "GA_per_match": round(ga / denom, 2),
                "xG": xg_tot,
                "xGC": xgc_tot,
                "xG_per_match": round(xg_tot / denom, 2),
                "xGC_per_match": round(xgc_tot / denom, 2),
                "GF_home_pm": round(gf_h / max(mp_h, 1), 2),
                "GA_home_pm": round(ga_h / max(mp_h, 1), 2),
                "GF_away_pm": round(gf_a / max(mp_a, 1), 2),
                "GA_away_pm": round(ga_a / max(mp_a, 1), 2),
                "last_n_form": form_str,
                "last_n_pts": r_pts,
                "last_n_GF_pm": round(r_gf / r_denom, 2),
                "last_n_GA_pm": round(r_ga / r_denom, 2),
                "last_n_CS": r_cs,
                "strength_attack_home": team.get("strength_attack_home", 1000),
                "strength_attack_away": team.get("strength_attack_away", 1000),
                "strength_defence_home": team.get("strength_defence_home", 1000),
                "strength_defence_away": team.get("strength_defence_away", 1000),
            }
        )

    club_df = pd.DataFrame(records)
    if not club_df.empty:
        # Target scores: higher = easier opponent to attack / easier opponent to keep a CS against
        club_df["attack_vulnerability"] = (
            0.5 * club_df["GA_per_match"] + 0.3 * club_df["xGC_per_match"] + 0.2 * club_df["last_n_GA_pm"]
        ).round(2)
        club_df["cs_opportunity_vs"] = (
            3.5 - (0.5 * club_df["GF_per_match"] + 0.3 * club_df["xG_per_match"] + 0.2 * club_df["last_n_GF_pm"])
        ).clip(lower=0.1).round(2)

        # Continuous Split FDR ratings (1.0 = Easiest opponent, 5.0 = Hardest opponent)
        # att_fdr_vs: Difficulty of SCORING against this club (lower attack_vulnerability => higher difficulty)
        vuln_min, vuln_max = float(club_df["attack_vulnerability"].min()), float(club_df["attack_vulnerability"].max())
        vuln_span = max(0.2, vuln_max - vuln_min)
        club_df["att_fdr_vs"] = (1.0 + 4.0 * (vuln_max - club_df["attack_vulnerability"]) / vuln_span).clip(1.0, 5.0).round(2)

        # def_fdr_vs: Difficulty of KEEPING A CLEAN SHEET against this club (lower cs_opportunity_vs => higher difficulty)
        cs_min, cs_max = float(club_df["cs_opportunity_vs"].min()), float(club_df["cs_opportunity_vs"].max())
        cs_span = max(0.2, cs_max - cs_min)
        club_df["def_fdr_vs"] = (1.0 + 4.0 * (cs_max - club_df["cs_opportunity_vs"]) / cs_span).clip(1.0, 5.0).round(2)

    return club_df


def build_fixture_ticker(
    bootstrap: Dict[str, Any],
    fixtures: List[Dict[str, Any]],
    start_gw: int,
    num_gws: int = 6,
    club_df: Optional[pd.DataFrame] = None,
    fdr_mode: str = "overall",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create upcoming fixture grid supporting Overall, Attacking, and Defensive (CS) FDR modes."""
    teams = build_teams_lookup(bootstrap)
    if teams.empty:
        return pd.DataFrame(), pd.DataFrame()

    short_map = dict(zip(teams["id"], teams["short_name"]))
    gw_range = list(range(start_gw, min(39, start_gw + num_gws)))

    att_fdr_map: Dict[int, float] = {}
    def_fdr_map: Dict[int, float] = {}
    elo_fdr_map: Dict[int, float] = {}
    if club_df is not None and not club_df.empty:
        att_fdr_map = dict(zip(club_df["team_id"].astype(int), club_df["att_fdr_vs"]))
        def_fdr_map = dict(zip(club_df["team_id"].astype(int), club_df["def_fdr_vs"]))
        elo_min, elo_max = float(club_df["Elo"].min()), float(club_df["Elo"].max())
        elo_span = max(50.0, elo_max - elo_min)
        for _, crow in club_df.iterrows():
            elo_fdr_map[int(crow["team_id"])] = 1.0 + 4.0 * (float(crow["Elo"]) - elo_min) / elo_span

    fix_df = pd.DataFrame(fixtures)
    if not fix_df.empty:
        fix_df = fix_df[fix_df["event"].isin(gw_range)]

    text_rows = []
    fdr_rows = []

    for _, team in teams.sort_values("name").iterrows():
        tid = int(team["id"])
        t_short = team["short_name"]
        t_name = team["name"]

        row_text: Dict[str, Any] = {"Club": t_name, "Short": t_short}
        row_fdr: Dict[str, Any] = {"Club": t_name, "Short": t_short}
        fdr_values = []

        for gw in gw_range:
            col_label = f"GW{gw}"
            if fix_df.empty:
                row_text[col_label] = "-"
                row_fdr[col_label] = np.nan
                continue

            gw_matches = fix_df[(fix_df["event"] == gw) & ((fix_df["team_h"] == tid) | (fix_df["team_a"] == tid))]
            if gw_matches.empty:
                row_text[col_label] = "BGW"
                row_fdr[col_label] = np.nan
            else:
                labels = []
                fdrs = []
                for _, m in gw_matches.iterrows():
                    is_home = int(m["team_h"]) == tid
                    opp_id = int(m["team_a"]) if is_home else int(m["team_h"])
                    opp = short_map.get(opp_id, "?")
                    base_fdr = float(m["team_h_difficulty"] if is_home else m["team_a_difficulty"])
                    venue_adj = -0.25 if is_home else 0.25

                    if fdr_mode == "attack" and opp_id in att_fdr_map:
                        fdr_val = float(np.clip(att_fdr_map[opp_id] + venue_adj, 1.0, 5.0))
                    elif fdr_mode == "defense" and opp_id in def_fdr_map:
                        fdr_val = float(np.clip(def_fdr_map[opp_id] + venue_adj, 1.0, 5.0))
                    elif opp_id in elo_fdr_map:
                        fdr_val = float(np.clip(0.5 * base_fdr + 0.5 * (elo_fdr_map[opp_id] + venue_adj), 1.0, 5.0))
                    else:
                        fdr_val = base_fdr

                    labels.append(f"{opp} ({'H' if is_home else 'A'})")
                    fdrs.append(round(fdr_val, 2))
                    fdr_values.append(fdr_val)

                row_text[col_label] = " / ".join(labels)
                row_fdr[col_label] = round(float(np.mean(fdrs)), 2)

        avg_fdr = round(float(np.mean(fdr_values)), 2) if fdr_values else 9.99
        row_text["Avg FDR"] = avg_fdr
        row_fdr["Avg FDR"] = avg_fdr
        text_rows.append(row_text)
        fdr_rows.append(row_fdr)

    text_df = pd.DataFrame(text_rows).sort_values("Avg FDR").reset_index(drop=True)
    fdr_df = pd.DataFrame(fdr_rows).loc[text_df.index].reset_index(drop=True)
    return text_df, fdr_df


def build_gameweek_matchups(
    fixtures: List[Dict[str, Any]],
    club_df: pd.DataFrame,
    players_df: pd.DataFrame,
    event_id: int,
) -> pd.DataFrame:
    """Build Poisson match projections (Expected Goals, CS%, and key players) for a gameweek."""
    fix_df = pd.DataFrame(fixtures)
    if fix_df.empty or club_df.empty:
        return pd.DataFrame()

    gw_fix = fix_df[fix_df["event"] == event_id].sort_values("kickoff_time")
    if gw_fix.empty:
        return pd.DataFrame()

    club_map = club_df.set_index("team_id").to_dict(orient="index")

    # League-wide baselines for multiplicative Poisson model
    avg_att = max(0.8, float((0.6 * club_df["xG_per_match"] + 0.4 * club_df["GF_per_match"]).mean()))
    avg_def = max(0.8, float((0.6 * club_df["xGC_per_match"] + 0.4 * club_df["GA_per_match"]).mean()))
    league_home_gf = max(1.0, float(club_df["GF_home_pm"].mean()) if "GF_home_pm" in club_df else 1.48)
    league_away_gf = max(0.8, float(club_df["GF_away_pm"].mean()) if "GF_away_pm" in club_df else 1.22)

    rows = []
    for _, m in gw_fix.iterrows():
        h_id = int(m["team_h"])
        a_id = int(m["team_a"])
        h_info = club_map.get(h_id, {})
        a_info = club_map.get(a_id, {})

        h_short = h_info.get("team_short", "HOME")
        a_short = a_info.get("team_short", "AWAY")

        # Multiplicative Poisson goal expectancy (lambda_Home, lambda_Away) + Elo adjustment
        h_att_rate = (0.6 * h_info.get("xG_per_match", avg_att) + 0.4 * h_info.get("GF_per_match", avg_att)) / avg_att
        a_def_rate = (0.6 * a_info.get("xGC_per_match", avg_def) + 0.4 * a_info.get("GA_per_match", avg_def)) / avg_def
        a_att_rate = (0.6 * a_info.get("xG_per_match", avg_att) + 0.4 * a_info.get("GF_per_match", avg_att)) / avg_att
        h_def_rate = (0.6 * h_info.get("xGC_per_match", avg_def) + 0.4 * h_info.get("GA_per_match", avg_def)) / avg_def

        elo_diff = float(h_info.get("Elo", 1750.0)) - float(a_info.get("Elo", 1750.0))
        elo_factor_h = float(np.clip(1.0 + (elo_diff / 1200.0), 0.78, 1.28))
        elo_factor_a = float(np.clip(1.0 - (elo_diff / 1200.0), 0.78, 1.28))

        h_proj_goals = round(float(np.clip(h_att_rate * a_def_rate * league_home_gf * elo_factor_h, 0.30, 3.80)), 2)
        a_proj_goals = round(float(np.clip(a_att_rate * h_def_rate * league_away_gf * elo_factor_a, 0.25, 3.50)), 2)

        # Exact Poisson Clean Sheet probabilities: P(CS_Home) = exp(-lambda_Away), P(CS_Away) = exp(-lambda_Home)
        h_cs_pct = round(100.0 * float(np.exp(-a_proj_goals)), 1)
        a_cs_pct = round(100.0 * float(np.exp(-h_proj_goals)), 1)

        # Top form/points players from each club
        if not players_df.empty:
            h_top = (
                players_df[players_df["team"] == h_id]
                .sort_values(["form", "total_points"], ascending=False)
                .head(2)["web_name"]
                .tolist()
            )
            a_top = (
                players_df[players_df["team"] == a_id]
                .sort_values(["form", "total_points"], ascending=False)
                .head(2)["web_name"]
                .tolist()
            )
        else:
            h_top, a_top = [], []

        score_str = (
            f"{int(m['team_h_score'])} - {int(m['team_a_score'])}"
            if pd.notna(m.get("team_h_score")) and pd.notna(m.get("team_a_score"))
            else "vs"
        )

        rows.append(
            {
                "fixture_id": int(m["id"]),
                "Matchup": f"{h_short} {score_str} {a_short}",
                "Score": score_str,
                "Home": h_info.get("team_name", h_short),
                "Home Short": h_short,
                "Home Crest": h_info.get("crest_url", ""),
                "Away": a_info.get("team_name", a_short),
                "Away Short": a_short,
                "Away Crest": a_info.get("crest_url", ""),
                "Home FDR": int(m.get("team_h_difficulty", 3)),
                "Away FDR": int(m.get("team_a_difficulty", 3)),
                "Home Form": h_info.get("last_n_form", "-"),
                "Away Form": a_info.get("last_n_form", "-"),
                "Home xG/m": h_info.get("xG_per_match", 0.0),
                "Home xGC/m": h_info.get("xGC_per_match", 0.0),
                "Away xG/m": a_info.get("xG_per_match", 0.0),
                "Away xGC/m": a_info.get("xGC_per_match", 0.0),
                "Home Proj G": h_proj_goals,
                "Away Proj G": a_proj_goals,
                "Home CS %": h_cs_pct,
                "Away CS %": a_cs_pct,
                "Best Attacking Side": h_short if h_proj_goals >= a_proj_goals else a_short,
                "Key Home Players": ", ".join(h_top),
                "Key Away Players": ", ".join(a_top),
                "Finished": bool(m.get("finished", False)),
            }
        )

    return pd.DataFrame(rows)


def build_gw_live_player_table(
    players_df: pd.DataFrame,
    live_data: Dict[str, Any],
    fixtures: List[Dict[str, Any]],
    teams_df: pd.DataFrame,
    event_id: int,
) -> pd.DataFrame:
    """Combine gameweek live/historical stats with player metadata and opponent context."""
    elements = live_data.get("elements", [])
    if not elements or players_df.empty:
        return pd.DataFrame()

    short_map = dict(zip(teams_df["id"], teams_df["short_name"])) if not teams_df.empty else {}
    fix_df = pd.DataFrame(fixtures)
    gw_fix = fix_df[fix_df["event"] == event_id] if not fix_df.empty else pd.DataFrame()

    # Map team_id -> opponent string for this GW
    opp_map: Dict[int, str] = {}
    if not gw_fix.empty:
        for _, m in gw_fix.iterrows():
            th = int(m["team_h"])
            ta = int(m["team_a"])
            h_opp = f"{short_map.get(ta, '?')} (H)"
            a_opp = f"{short_map.get(th, '?')} (A)"
            opp_map[th] = f"{opp_map[th]} / {h_opp}" if th in opp_map else h_opp
            opp_map[ta] = f"{opp_map[ta]} / {a_opp}" if ta in opp_map else a_opp

    records = []
    for item in elements:
        pid = int(item["id"])
        stats = item.get("stats", {})
        records.append(
            {
                "id": pid,
                "gw_minutes": int(stats.get("minutes", 0)),
                "gw_points": int(stats.get("total_points", 0)),
                "gw_goals": int(stats.get("goals_scored", 0)),
                "gw_assists": int(stats.get("assists", 0)),
                "gw_cs": int(stats.get("clean_sheets", 0)),
                "gw_gc": int(stats.get("goals_conceded", 0)),
                "gw_saves": int(stats.get("saves", 0)),
                "gw_bonus": int(stats.get("bonus", 0)),
                "gw_bps": int(stats.get("bps", 0)),
                "gw_xG": float(stats.get("expected_goals", 0.0)),
                "gw_xA": float(stats.get("expected_assists", 0.0)),
                "gw_xGI": float(stats.get("expected_goal_involvements", 0.0)),
                "gw_xGC": float(stats.get("expected_goals_conceded", 0.0)),
                "gw_ict": float(stats.get("ict_index", 0.0)),
            }
        )

    gw_df = pd.DataFrame(records)
    merged = players_df[
        ["id", "web_name", "team", "team_short", "position", "price", "selected_by_percent"]
    ].merge(gw_df, on="id", how="inner")
    merged["opponent"] = merged["team"].map(lambda t: opp_map.get(int(t), "BGW"))
    return merged.sort_values(["gw_points", "gw_bps", "gw_xGI"], ascending=False).reset_index(drop=True)


def format_player_element_summary(
    summary: Dict[str, Any], teams_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Format a single player's past gameweek history and upcoming fixtures."""
    short_map = dict(zip(teams_df["id"], teams_df["short_name"])) if not teams_df.empty else {}

    history = pd.DataFrame(summary.get("history", []))
    if not history.empty:
        history["opponent"] = history["opponent_team"].map(short_map)
        history["venue"] = np.where(history["was_home"], "H", "A")
        history["match"] = history["opponent"] + " (" + history["venue"] + ")"
        for col in ["expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded"]:
            if col in history.columns:
                history[col] = pd.to_numeric(history[col], errors="coerce").fillna(0.0)
        history.rename(
            columns={
                "round": "GW",
                "total_points": "Points",
                "minutes": "Mins",
                "goals_scored": "G",
                "assists": "A",
                "clean_sheets": "CS",
                "goals_conceded": "GC",
                "expected_goals": "xG",
                "expected_assists": "xA",
                "expected_goal_involvements": "xGI",
                "expected_goals_conceded": "xGC",
                "bps": "BPS",
                "bonus": "Bonus",
            },
            inplace=True,
        )
        history["Price"] = (history["value"] / 10.0).round(1)

    upcoming = pd.DataFrame(summary.get("fixtures", []))
    if not upcoming.empty:
        upcoming["opp_id"] = np.where(upcoming["is_home"], upcoming["team_a"], upcoming["team_h"])
        upcoming["Opponent"] = upcoming["opp_id"].map(short_map)
        upcoming["Venue"] = np.where(upcoming["is_home"], "H", "A")
        upcoming["Fixture"] = upcoming["Opponent"] + " (" + upcoming["Venue"] + ")"
        upcoming.rename(columns={"event": "GW", "difficulty": "FDR"}, inplace=True)

    return history, upcoming


def compute_player_xp(
    players_df: pd.DataFrame,
    club_df: pd.DataFrame,
    fixtures: List[Dict[str, Any]],
    current_gw: int,
    horizon: int = 3,
) -> pd.DataFrame:
    """Compute vectorized Expected Points (xP) with position-specific scoring and Bayesian minutes shrinkage."""
    if players_df.empty or club_df.empty:
        return players_df.copy()

    next_gw = min(38, current_gw + 1)
    gw_list = list(range(next_gw, min(39, next_gw + horizon)))

    df = players_df.copy()

    # Availability probability & status label
    status_prob_map = {"a": 1.0, "d": 0.75, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}
    chance_col = pd.to_numeric(df.get("chance_of_playing_next_round"), errors="coerce")
    fallback_prob = df["status"].map(status_prob_map).fillna(1.0)
    df["play_prob"] = np.where(chance_col.notna(), chance_col / 100.0, fallback_prob).clip(0.0, 1.0)

    df["Availability"] = np.where(
        df["play_prob"] >= 1.0,
        "100% Fit",
        np.where(df["play_prob"] > 0.0, f"{(df['play_prob'] * 100).astype(int)}% Doubt", "0% Out"),
    )

    # Minutes security factor
    total_avail_mins = max(1, current_gw) * 90.0
    mins_ratio = (df["minutes"] / total_avail_mins).clip(0.0, 1.0)
    mins_factor = np.where(df["minutes"] == 0, 0.0, np.clip(0.35 + 0.85 * mins_ratio, 0.0, 1.0))

    # Empirical Bayes shrinkage (prior = 3.0 nineties / 270 mins toward minutes-weighted positional mean)
    prior_90s = 3.0
    nineties = df["minutes"] / 90.0
    pos_90s = df.groupby("position")["minutes"].transform("sum") / 90.0
    pos_90s_safe = pos_90s.replace(0, np.nan)

    pos_xg_90 = (df.groupby("position")["xG"].transform("sum") / pos_90s_safe).fillna(0.15)
    pos_xa_90 = (df.groupby("position")["xA"].transform("sum") / pos_90s_safe).fillna(0.10)
    pos_bps_90 = (df.groupby("position")["bps"].transform("sum") / pos_90s_safe).fillna(12.0)
    pos_saves_90 = (df.groupby("position")["saves"].transform("sum") / pos_90s_safe).fillna(0.0)

    shrunk_xg_90 = (df["xG"] + prior_90s * pos_xg_90) / (nineties + prior_90s)
    shrunk_xa_90 = (df["xA"] + prior_90s * pos_xa_90) / (nineties + prior_90s)
    shrunk_bps_90 = (df["bps"] + prior_90s * pos_bps_90) / (nineties + prior_90s)
    shrunk_saves_90 = (df["saves"] + prior_90s * pos_saves_90) / (nineties + prior_90s)

    # Position-specific FPL scoring rules
    goal_pts_map = {"GKP": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
    cs_pts_map = {"GKP": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}
    goal_weight = df["position"].map(goal_pts_map).fillna(4.0)
    cs_weight = df["position"].map(cs_pts_map).fillna(0.0)
    is_def_unit = df["position"].isin(["GKP", "DEF"])

    # Attacking + appearance + bonus + save base component per 90 (weighted by Set-Piece multiplier)
    sp_mult = df["sp_multiplier"] if "sp_multiplier" in df.columns else 1.0
    att_bonus_saves = (
        2.0
        + (shrunk_xg_90 * goal_weight + shrunk_xa_90 * 3.0) * sp_mult
        + (shrunk_bps_90 / 18.0)
        + np.where(df["position"] == "GKP", shrunk_saves_90 / 3.0, 0.0)
    )

    # Club lookup maps for opponent & Poisson clean-sheet adjustments
    short_map = dict(zip(club_df["team_id"], club_df["team_short"]))
    att_vuln_map = dict(zip(club_df["team_id"], club_df["attack_vulnerability"]))
    team_xg_pm = dict(zip(club_df["team_id"], club_df["xG_per_match"]))
    team_gf_pm = dict(zip(club_df["team_id"], club_df["GF_per_match"]))
    team_xgc_pm = dict(zip(club_df["team_id"], club_df["xGC_per_match"]))
    team_ga_pm = dict(zip(club_df["team_id"], club_df["GA_per_match"]))
    avg_att_vuln = max(0.5, float(club_df["attack_vulnerability"].mean()))

    fix_df = pd.DataFrame(fixtures)
    fix_df = fix_df[fix_df["event"].isin(gw_list)] if not fix_df.empty else pd.DataFrame()

    team_gw_fixtures: Dict[int, Dict[int, List[Tuple[int, bool, int]]]] = {
        int(tid): {gw: [] for gw in gw_list} for tid in club_df["team_id"]
    }
    if not fix_df.empty:
        for _, m in fix_df.iterrows():
            ev = int(m["event"])
            th = int(m["team_h"])
            ta = int(m["team_a"])
            fdr_h = int(m.get("team_h_difficulty", 3))
            fdr_a = int(m.get("team_a_difficulty", 3))
            if th in team_gw_fixtures and ev in team_gw_fixtures[th]:
                team_gw_fixtures[th][ev].append((ta, True, fdr_h))
            if ta in team_gw_fixtures and ev in team_gw_fixtures[ta]:
                team_gw_fixtures[ta][ev].append((th, False, fdr_a))

    # Precompute per-team fixture strings and multipliers across the 20 clubs
    team_next_opp: Dict[int, str] = {}
    team_next_fdr: Dict[int, float] = {}
    team_hor_opps: Dict[int, str] = {}
    # Per-(team, gw_idx): list of (att_mult, cs_prob, exp_gc)
    team_gw_metrics: Dict[int, List[List[Tuple[float, float, float]]]] = {}

    for tid in club_df["team_id"].astype(int):
        next_opps_str = []
        next_fdrs = []
        all_opps_str = []
        gw_metrics_list: List[List[Tuple[float, float, float]]] = []

        own_def = 0.5 * team_xgc_pm.get(tid, 1.3) + 0.5 * team_ga_pm.get(tid, 1.3)

        for gw in gw_list:
            matches = team_gw_fixtures.get(tid, {}).get(gw, [])
            if not matches:
                gw_metrics_list.append([])
                if gw == next_gw:
                    next_opps_str.append("BGW")
                    next_fdrs.append(5)
                all_opps_str.append("BGW")
                continue

            gw_labels = []
            m_tuples: List[Tuple[float, float, float]] = []
            for opp_id, is_home, fdr in matches:
                opp_short = short_map.get(opp_id, "?")
                venue_char = "H" if is_home else "A"
                gw_labels.append(f"{opp_short}({venue_char})")
                if gw == next_gw:
                    next_opps_str.append(f"{opp_short} ({venue_char})")
                    next_fdrs.append(fdr)

                venue_att_mult = 1.07 if is_home else 0.94
                opp_vuln = att_vuln_map.get(opp_id, avg_att_vuln)
                att_mult = venue_att_mult * (0.65 + 0.35 * (opp_vuln / avg_att_vuln))

                # Expected goals conceded (lambda_opp) & Poisson Clean Sheet probability
                opp_att = 0.5 * team_xg_pm.get(opp_id, 1.3) + 0.5 * team_gf_pm.get(opp_id, 1.3)
                venue_def_mult = 0.90 if is_home else 1.12
                lambda_opp = max(0.25, 0.55 * opp_att + 0.45 * own_def) * venue_def_mult
                cs_prob = float(np.exp(-lambda_opp))
                m_tuples.append((att_mult, cs_prob, lambda_opp))

            gw_metrics_list.append(m_tuples)
            all_opps_str.append("/".join(gw_labels))

        team_next_opp[tid] = " / ".join(next_opps_str) if next_opps_str else "BGW"
        team_next_fdr[tid] = round(float(np.mean(next_fdrs)), 1) if next_fdrs else 5.0
        team_hor_opps[tid] = ", ".join(all_opps_str)
        team_gw_metrics[tid] = gw_metrics_list

    # Vectorized player xP across gw_list
    teams_arr = df["team"].astype(int).values
    play_prob_arr = df["play_prob"].values
    form_arr = df["form"].values
    ppg_arr = df["points_per_game"].values
    att_base_arr = att_bonus_saves.values
    cs_w_arr = cs_weight.values
    is_def_arr = is_def_unit.values

    xp_next_arr = np.zeros(len(df), dtype=float)
    xp_hor_arr = np.zeros(len(df), dtype=float)

    for gw_idx in range(len(gw_list)):
        eff_prob = (
            play_prob_arr
            if gw_idx == 0
            else np.where(play_prob_arr > 0, np.minimum(1.0, play_prob_arr + 0.15), 0.0)
        )
        gw_xp = np.zeros(len(df), dtype=float)

        for tid, gw_m_list in team_gw_metrics.items():
            if gw_idx >= len(gw_m_list) or not gw_m_list[gw_idx]:
                continue
            mask = teams_arr == tid
            if not np.any(mask):
                continue

            team_match_xp = np.zeros(int(mask.sum()), dtype=float)
            for att_mult, cs_prob, lambda_opp in gw_m_list[gw_idx]:
                # Position-specific underlying expected points for this fixture
                gc_penalty = np.where(is_def_arr[mask], 0.5 * lambda_opp, 0.0)
                fixture_underlying = (
                    att_base_arr[mask] * att_mult
                    + cs_w_arr[mask] * cs_prob
                    - gc_penalty
                )
                blended_rate = (
                    0.40 * form_arr[mask] * att_mult
                    + 0.25 * ppg_arr[mask] * att_mult
                    + 0.35 * np.maximum(0.5, fixture_underlying)
                )
                team_match_xp += blended_rate * eff_prob[mask] * mins_factor[mask]

            gw_xp[mask] = team_match_xp

        if gw_idx == 0:
            xp_next_arr = gw_xp.copy()
        xp_hor_arr += gw_xp

    df["xP_next"] = np.round(xp_next_arr, 2)
    df["xP_horizon"] = np.round(xp_hor_arr, 2)
    df["next_opp"] = df["team"].astype(int).map(team_next_opp).fillna("BGW")
    df["next_fdr"] = df["team"].astype(int).map(team_next_fdr).fillna(5.0)
    df["horizon_fixtures"] = df["team"].astype(int).map(team_hor_opps).fillna("BGW")
    return df


def optimize_starting_xi(squad_df: pd.DataFrame) -> Dict[str, Any]:
    """Find the optimal valid FPL starting XI, Captain, Vice-Captain, and Bench order."""
    if squad_df.empty or len(squad_df) < 15:
        return {}

    sq = squad_df.copy().reset_index(drop=True)

    # Current saved lineup expected points (including current captain multiplier)
    curr_starters = sq[sq["squad_slot"] <= 11]
    curr_cap_xp = float(sq[sq["is_captain"]]["xP_next"].sum())
    current_xi_xp = round(float(curr_starters["xP_next"].sum()) + curr_cap_xp, 2)

    # Separate by position sorted by xP_next descending, tie-breaking with xP_horizon and form
    gkps = sq[sq["position"] == "GKP"].sort_values(["xP_next", "xP_horizon", "form"], ascending=False)
    defs = sq[sq["position"] == "DEF"].sort_values(["xP_next", "xP_horizon", "form"], ascending=False)
    mids = sq[sq["position"] == "MID"].sort_values(["xP_next", "xP_horizon", "form"], ascending=False)
    fwds = sq[sq["position"] == "FWD"].sort_values(["xP_next", "xP_horizon", "form"], ascending=False)

    valid_formations = [
        (3, 4, 3),
        (3, 5, 2),
        (4, 4, 2),
        (4, 3, 3),
        (4, 5, 1),
        (5, 3, 2),
        (5, 4, 1),
        (5, 2, 3),
    ]

    best_score = -1.0
    best_starter_ids: List[int] = []
    best_formation = "3-4-3"

    start_gkp = gkps.head(1)
    for n_def, n_mid, n_fwd in valid_formations:
        if len(defs) >= n_def and len(mids) >= n_mid and len(fwds) >= n_fwd:
            chosen = pd.concat(
                [start_gkp, defs.head(n_def), mids.head(n_mid), fwds.head(n_fwd)],
                ignore_index=True,
            )
            score = float(chosen["xP_next"].sum())
            if score > best_score:
                best_score = score
                best_starter_ids = chosen["id"].astype(int).tolist()
                best_formation = f"{n_def}-{n_mid}-{n_fwd}"

    sq["is_opt_starter"] = sq["id"].astype(int).isin(best_starter_ids)
    starters = sq[sq["is_opt_starter"]].sort_values(["xP_next", "xP_horizon"], ascending=False)
    cap_id = int(starters.iloc[0]["id"]) if not starters.empty else -1
    vc_id = int(starters.iloc[1]["id"]) if len(starters) > 1 else -1

    # Order starters by position (GKP -> DEF -> MID -> FWD) and bench (Bench GKP -> Bench 1, 2, 3 by xP_next)
    pos_order = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
    sq["pos_rank"] = sq["position"].map(pos_order)

    opt_starters = sq[sq["is_opt_starter"]].sort_values(["pos_rank", "xP_next"], ascending=[True, False]).copy()
    bench_gkp = sq[(~sq["is_opt_starter"]) & (sq["position"] == "GKP")].copy()
    bench_outfield = (
        sq[(~sq["is_opt_starter"]) & (sq["position"] != "GKP")]
        .sort_values(["xP_next", "xP_horizon"], ascending=False)
        .copy()
    )

    opt_starters["Recommended_Role"] = np.where(
        opt_starters["id"] == cap_id,
        "© Captain (Starter)",
        np.where(opt_starters["id"] == vc_id, "VC (Starter)", "Starter"),
    )
    bench_gkp["Recommended_Role"] = "Bench GKP"
    bench_outfield["Recommended_Role"] = [f"Bench {i + 1}" for i in range(len(bench_outfield))]

    ordered_sq = pd.concat([opt_starters, bench_gkp, bench_outfield], ignore_index=True)
    ordered_sq["Opt_Mult"] = np.where(
        ordered_sq["id"] == cap_id, 2, np.where(ordered_sq["is_opt_starter"], 1, 0)
    )
    ordered_sq["Effective_xP"] = (ordered_sq["xP_next"] * ordered_sq["Opt_Mult"]).round(2)

    opt_xi_xp = round(float(ordered_sq["Effective_xP"].sum()), 2)
    bench_xp = round(float(ordered_sq[~ordered_sq["is_opt_starter"]]["xP_next"].sum()), 2)
    cap_row = ordered_sq[ordered_sq["id"] == cap_id].iloc[0] if cap_id != -1 else None
    vc_row = ordered_sq[ordered_sq["id"] == vc_id].iloc[0] if vc_id != -1 else None

    return {
        "squad_table": ordered_sq,
        "formation": best_formation,
        "opt_xi_xp": opt_xi_xp,
        "current_xi_xp": current_xi_xp,
        "lineup_gain_xp": round(opt_xi_xp - current_xi_xp, 2),
        "bench_xp": bench_xp,
        "captain_name": f"{cap_row['web_name']} ({cap_row['next_opp']})" if cap_row is not None else "-",
        "captain_xp": float(cap_row["xP_next"]) if cap_row is not None else 0.0,
        "vc_name": f"{vc_row['web_name']} ({vc_row['next_opp']})" if vc_row is not None else "-",
    }


def evaluate_chip_strategy(
    mgr_hist: Dict[str, Any],
    lineup_res: Dict[str, Any],
    next_gw: int,
) -> pd.DataFrame:
    """Evaluate chip availability and recommend whether to play or hold chips for the next GW."""
    chips_raw = mgr_hist.get("chips", [])
    used_chips: Dict[str, int] = {}
    for c in chips_raw:
        c_name = c.get("name")
        c_ev = int(c.get("event", 0))
        if c_name == "wildcard":
            # FPL grants two Wildcards: WC1 (GW 1-19) and WC2 (GW 20-38)
            if (next_gw <= 19 and c_ev <= 19) or (next_gw >= 20 and c_ev >= 20):
                used_chips["wildcard"] = c_ev
        elif c_name:
            used_chips[c_name] = c_ev

    sq = lineup_res.get("squad_table", pd.DataFrame())
    bench_xp = float(lineup_res.get("bench_xp", 0.0))
    cap_xp = float(lineup_res.get("captain_xp", 0.0))
    opt_xi_xp = float(lineup_res.get("opt_xi_xp", 0.0))
    flagged_count = int((sq["play_prob"] < 0.75).sum()) if not sq.empty else 0

    wc_label = "Wildcard 1 (GW1–19)" if next_gw <= 19 else "Wildcard 2 (GW20–38)"
    chip_Meta = [
        ("wildcard", wc_label),
        ("freehit", "Free Hit (FH)"),
        ("bboost", "Bench Boost (BB)"),
        ("3xc", "Triple Captain (TC)"),
    ]

    rows = []
    for key, label in chip_Meta:
        is_used = key in used_chips
        status_str = f"Used (GW {used_chips[key]})" if is_used else "✅ Available"

        if is_used:
            rec = "Already Used"
            rationale = f"Played in GW {used_chips[key]}."
        elif key == "bboost":
            if bench_xp >= 15.0:
                rec = f"🔥 Strong Play (+{bench_xp:.1f} xP)"
                rationale = f"Your 4 bench players project for {bench_xp:.1f} xP in GW {next_gw}."
            else:
                rec = f"⏳ Hold ({bench_xp:.1f} Bench xP)"
                rationale = f"Bench projects for {bench_xp:.1f} xP (target >= 15.0 xP or a Double GW)."
        elif key == "3xc":
            if cap_xp >= 7.2:
                rec = f"🔥 Strong Play (+{cap_xp:.1f} extra xP)"
                rationale = f"Top captain projects for {cap_xp:.2f} base xP ({cap_xp * 3:.1f} pts with 3x)."
            else:
                rec = f"⏳ Hold (+{cap_xp:.1f} extra xP)"
                rationale = f"Top captain projects for {cap_xp:.2f} base xP (target >= 7.2 xP or Double GW)."
        elif key == "freehit":
            if opt_xi_xp < 38.0 or flagged_count >= 5:
                rec = "⚠️ Consider Free Hit"
                rationale = f"Squad has {flagged_count} flagged players and low GW {next_gw} projection ({opt_xi_xp:.1f} xP)."
            else:
                rec = "⏳ Save for Blank/Double GW"
                rationale = f"Optimal XI projects for a healthy {opt_xi_xp:.1f} xP with {flagged_count} flagged players."
        else:  # wildcard
            if flagged_count >= 4 or opt_xi_xp < 40.0:
                rec = "⚠️ Consider Wildcard"
                rationale = f"{flagged_count} squad issues detected and multi-GW structure can be upgraded."
            else:
                rec = "⏳ Hold Wildcard"
                rationale = f"Core squad is healthy ({opt_xi_xp:.1f} GW {next_gw} xP); use Free Transfers."

        rows.append(
            {
                "Chip": label,
                "Status": status_str,
                f"GW {next_gw} Recommendation": rec,
                "Analytical Rationale": rationale,
            }
        )

    return pd.DataFrame(rows)


def build_initial_squad(
    all_players_xp: pd.DataFrame,
    budget_m: float = 100.0,
    bench_weight: float = 0.1,
    min_play_prob: float = 0.75,
) -> pd.DataFrame:
    """Recommend a starting 15-man squad that maximises full-horizon xP.

    Solved as a small integer program: 2 GKP / 5 DEF / 5 MID / 3 FWD, max 3 players per
    club, total price within budget, and a valid starting XI (1 GKP, 3-5 DEF, 2-5 MID,
    1-3 FWD). Starters count fully and bench players count at ``bench_weight``.

    Returns a picks-style DataFrame (columns: element, position, is_captain,
    is_vice_captain, multiplier) so it can replace a manager's real picks. Returns an
    empty DataFrame if no feasible squad exists.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    pool = all_players_xp[
        (all_players_xp["play_prob"] >= min_play_prob) & (all_players_xp["price"] > 0)
    ].copy()
    # Prune to the strongest candidates per position (by raw xP and by xP per £m) to keep the solve fast.
    pool["_xp_per_m"] = pool["xP_horizon"].fillna(0.0) / pool["price"]
    keep_ids = set()
    for _, grp in pool.groupby("position"):
        keep_ids |= set(grp.nlargest(25, "xP_horizon")["id"])
        keep_ids |= set(grp.nlargest(25, "_xp_per_m")["id"])
        keep_ids |= set(grp.nsmallest(4, "price")["id"])  # cheap fillers keep the budget feasible
    pool = pool[pool["id"].isin(keep_ids)].reset_index(drop=True)
    n = len(pool)
    if n == 0:
        return pd.DataFrame()

    # Variables: x_i (in squad) for i in [0, n), s_i (in starting XI) for i in [n, 2n).
    xp = pool["xP_horizon"].fillna(0.0).to_numpy(dtype=float)
    cost = pool["price"].to_numpy(dtype=float)
    pos = pool["position"].to_numpy()
    team = pool["team"].astype(int).to_numpy()

    c = -np.concatenate([bench_weight * xp, (1.0 - bench_weight) * xp])
    rows, lo, hi = [], [], []

    def add(coefs: np.ndarray, low: float, high: float) -> None:
        rows.append(coefs)
        lo.append(low)
        hi.append(high)

    zeros, eye = np.zeros(n), np.eye(n)
    add(np.concatenate([cost, zeros]), 0, budget_m)
    add(np.concatenate([np.ones(n), zeros]), 15, 15)
    add(np.concatenate([zeros, np.ones(n)]), 11, 11)
    for p, (squad_n, (xi_lo, xi_hi)) in {
        "GKP": (2, (1, 1)),
        "DEF": (5, (3, 5)),
        "MID": (5, (2, 5)),
        "FWD": (3, (1, 3)),
    }.items():
        mask = (pos == p).astype(float)
        add(np.concatenate([mask, zeros]), squad_n, squad_n)
        add(np.concatenate([zeros, mask]), xi_lo, xi_hi)
    for t in np.unique(team):
        add(np.concatenate([(team == t).astype(float), zeros]), 0, 3)
    for i in range(n):  # starter must be in squad
        add(np.concatenate([-eye[i], eye[i]]), -np.inf, 0)

    res = milp(
        c,
        constraints=LinearConstraint(np.vstack(rows), lo, hi),
        integrality=np.ones(2 * n),
        bounds=Bounds(0, 1),
        options={"time_limit": 20},
    )
    if res.x is None:  # a time-limited solve may still return a good feasible solution
        return pd.DataFrame()

    in_squad = res.x[:n] > 0.5
    starting = res.x[n:] > 0.5
    chosen = pool.loc[in_squad].copy()
    chosen["is_starter"] = starting[in_squad]

    # Slots 1-11 = starters (best xP_next first), 12-15 = bench ordered by xP_next.
    chosen = chosen.sort_values(
        ["is_starter", "xP_next"], ascending=[False, False]
    ).reset_index(drop=True)
    chosen["slot"] = np.arange(1, len(chosen) + 1)
    starters = chosen[chosen["is_starter"]]
    cap_id = int(starters.iloc[0]["id"])
    vc_id = int(starters.iloc[1]["id"]) if len(starters) > 1 else cap_id

    return pd.DataFrame(
        {
            "element": chosen["id"].astype(int),
            "position": chosen["slot"].astype(int),
            "is_captain": chosen["id"].astype(int) == cap_id,
            "is_vice_captain": chosen["id"].astype(int) == vc_id,
            "multiplier": np.where(chosen["id"].astype(int) == cap_id, 2, np.where(chosen["is_starter"], 1, 0)),
        }
    )


def estimate_free_transfers(history: Dict[str, Any], max_ft: int = 5) -> int:
    """Estimate the free transfers a manager has for the next gameweek.

    The FPL API exposes no free-transfer count, so it is replayed from history:
    GW1 transfers are unlimited, then each GW uses its transfers from the bank
    and adds 1 (capped at ``max_ft``). A Wildcard resets the next GW to 1 free
    transfer. A Free Hit uses none and keeps the bank. Estimate only: transfers
    already made for the upcoming GW are not yet in history.
    """
    chips = {int(c["event"]): c.get("name") for c in (history.get("chips", []) or []) if c.get("event")}
    ft = 1
    for row in sorted(history.get("current", []) or [], key=lambda r: int(r["event"])):
        gw = int(row["event"])
        if gw == 1:
            continue
        chip = chips.get(gw)
        if chip == "wildcard":
            ft = 1
        elif chip == "freehit":
            ft = min(ft + 1, max_ft)
        else:
            ft = min(max(ft - int(row.get("event_transfers") or 0), 0) + 1, max_ft)
    return max(1, ft)


def recommend_transfers(
    squad_df: pd.DataFrame,
    all_players_xp: pd.DataFrame,
    bank_m: float,
    free_transfers: int = 1,
    horizon: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate 1-FT recommendations, 2-Transfer combinations, and Position Target Watchlist."""
    if squad_df.empty or all_players_xp.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    squad_ids = set(squad_df["id"].astype(int).tolist())
    club_counts = squad_df["team"].astype(int).value_counts().to_dict()
    base_xi_xp = float(optimize_starting_xi(squad_df).get("opt_xi_xp", 0.0))

    # Pool of healthy external candidates who have played or are fit
    candidates = all_players_xp[
        (~all_players_xp["id"].astype(int).isin(squad_ids))
        & (all_players_xp["play_prob"] >= 0.75)
        & (all_players_xp["xP_horizon"] > 0)
    ].copy()

    # 1. Single Transfer Recommendations (evaluate best upgrades for each player in squad)
    single_moves = []
    for _, out_p in squad_df.iterrows():
        out_id = int(out_p["id"])
        out_team = int(out_p["team"])
        out_pos = out_p["position"]
        out_price = (
            round(float(out_p["selling_price"]) / 10.0, 1)
            if "selling_price" in out_p and pd.notna(out_p["selling_price"])
            else float(out_p["price"])
        )
        max_budget = round(out_price + bank_m, 1)

        pos_cands = candidates[
            (candidates["position"] == out_pos) & (candidates["price"] <= max_budget)
        ].copy()

        # Enforce max 3 players per club after removing out_team
        def _club_valid(cand_team: int) -> bool:
            curr = club_counts.get(int(cand_team), 0)
            if int(cand_team) == out_team:
                curr -= 1
            return curr < 3

        pos_cands = pos_cands[pos_cands["team"].apply(_club_valid)]
        if pos_cands.empty:
            continue

        pos_cands["xP_next_gain"] = (pos_cands["xP_next"] - float(out_p["xP_next"])).round(2)
        pos_cands["xP_horizon_gain"] = (pos_cands["xP_horizon"] - float(out_p["xP_horizon"])).round(2)

        best_for_out = (
            pos_cands[pos_cands["xP_horizon_gain"] > 0.5]
            .sort_values(["xP_horizon_gain", "xP_next_gain"], ascending=False)
            .head(3)
        )

        for _, in_p in best_for_out.iterrows():
            hit_cost = 0.0 if free_transfers >= 1 else 4.0
            net_hor_gain = round(float(in_p["xP_horizon_gain"]) - hit_cost, 2)

            # Evaluate true Starting XI impact (including captaincy re-optimization)
            trial_sq = squad_df.copy()
            replace_mask = trial_sq["id"].astype(int) == out_id
            for col in ["id", "web_name", "team", "team_short", "position", "price", "form", "xP_next", "xP_horizon", "next_opp"]:
                if col in trial_sq.columns and col in in_p:
                    trial_sq.loc[replace_mask, col] = in_p[col]
            new_xi_xp = float(optimize_starting_xi(trial_sq).get("opt_xi_xp", base_xi_xp))
            xi_next_gain = round(new_xi_xp - base_xi_xp, 2)

            single_moves.append(
                {
                    "Pos": out_pos,
                    "Transfer Out": f"{out_p['web_name']} ({out_p['team_short']} £{out_price:.1f}m)",
                    "Out Availability": out_p["Availability"],
                    "Out Next Opp": out_p["next_opp"],
                    "Out Next xP": float(out_p["xP_next"]),
                    f"Out {horizon}-GW xP": float(out_p["xP_horizon"]),
                    "Transfer In": f"{in_p['web_name']} ({in_p['team_short']} £{in_p['price']:.1f}m)",
                    "In Next Opp": in_p["next_opp"],
                    "In Next xP": float(in_p["xP_next"]),
                    f"In {horizon}-GW xP": float(in_p["xP_horizon"]),
                    "Next GW Gain": float(in_p["xP_next_gain"]),
                    "XI Next GW Gain": xi_next_gain,
                    f"Net {horizon}-GW xP Gain": net_hor_gain,
                    "New ITB (£m)": round(max_budget - float(in_p["price"]), 1),
                    "_out_id": out_id,
                    "_in_id": int(in_p["id"]),
                    "_out_team": out_team,
                    "_in_team": int(in_p["team"]),
                    "_out_price": out_price,
                    "_in_price": float(in_p["price"]),
                }
            )

    single_df = pd.DataFrame(single_moves)
    if not single_df.empty:
        single_df = single_df.sort_values(
            [f"Net {horizon}-GW xP Gain", "XI Next GW Gain", "Next GW Gain"], ascending=False
        ).reset_index(drop=True)

    # 2. Double Transfer (2-Player Package) Recommendations
    # Pair weakest squad players with top candidates within combined budget (out1 + out2 + bank)
    double_moves = []
    weak_squad = squad_df.sort_values(["play_prob", "xP_horizon", "xP_next"], ascending=[True, True, True]).head(6)
    top_cands_by_pos = {
        pos: candidates[candidates["position"] == pos]
        .sort_values("xP_horizon", ascending=False)
        .head(8)
        for pos in ["GKP", "DEF", "MID", "FWD"]
    }

    weak_list = list(weak_squad.iterrows())
    hit_2ft = 0.0 if free_transfers >= 2 else (4.0 if free_transfers == 1 else 8.0)

    for i in range(len(weak_list)):
        for j in range(i + 1, len(weak_list)):
            _, out1 = weak_list[i]
            _, out2 = weak_list[j]
            pos1, pos2 = out1["position"], out2["position"]
            comb_budget = round(float(out1["price"]) + float(out2["price"]) + bank_m, 1)
            base_hor_xp = float(out1["xP_horizon"]) + float(out2["xP_horizon"])
            base_next_xp = float(out1["xP_next"]) + float(out2["xP_next"])

            cands1 = top_cands_by_pos.get(pos1, pd.DataFrame())
            cands2 = top_cands_by_pos.get(pos2, pd.DataFrame())
            if cands1.empty or cands2.empty:
                continue

            for _, in1 in cands1.iterrows():
                for _, in2 in cands2.iterrows():
                    if int(in1["id"]) == int(in2["id"]):
                        continue
                    cost = round(float(in1["price"]) + float(in2["price"]), 1)
                    if cost > comb_budget:
                        continue

                    # Check 3-per-club constraint
                    temp_counts = club_counts.copy()
                    temp_counts[int(out1["team"])] = temp_counts.get(int(out1["team"]), 1) - 1
                    temp_counts[int(out2["team"])] = temp_counts.get(int(out2["team"]), 1) - 1
                    t1, t2 = int(in1["team"]), int(in2["team"])
                    temp_counts[t1] = temp_counts.get(t1, 0) + 1
                    temp_counts[t2] = temp_counts.get(t2, 0) + 1
                    if temp_counts[t1] > 3 or temp_counts[t2] > 3:
                        continue

                    new_hor_xp = float(in1["xP_horizon"]) + float(in2["xP_horizon"])
                    new_next_xp = float(in1["xP_next"]) + float(in2["xP_next"])
                    net_gain = round(new_hor_xp - base_hor_xp - hit_2ft, 2)
                    if net_gain <= 1.0:
                        continue

                    double_moves.append(
                        {
                            "Positions": f"{pos1} + {pos2}",
                            "Package Out": f"{out1['web_name']} (£{out1['price']:.1f}m) + {out2['web_name']} (£{out2['price']:.1f}m)",
                            "Package In": f"{in1['web_name']} (£{in1['price']:.1f}m) + {in2['web_name']} (£{in2['price']:.1f}m)",
                            "Hit Cost": f"-{int(hit_2ft)} pts" if hit_2ft > 0 else "Free (2 FT)",
                            "Next GW Gain": round(new_next_xp - base_next_xp, 2),
                            f"Net {horizon}-GW xP Gain": net_gain,
                            "Remaining ITB (£m)": round(comb_budget - cost, 1),
                        }
                    )

    double_df = pd.DataFrame(double_moves)
    if not double_df.empty:
        double_df = (
            double_df.sort_values([f"Net {horizon}-GW xP Gain", "Next GW Gain"], ascending=False)
            .drop_duplicates(subset=["Package Out"])
            .head(10)
            .reset_index(drop=True)
        )

    # 3. Targeted Watchlist by Position
    watchlist_rows = []
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        top_pos = (
            candidates[candidates["position"] == pos]
            .sort_values(["xP_horizon", "xP_next"], ascending=False)
            .head(5)
        )
        for _, p in top_pos.iterrows():
            watchlist_rows.append(
                {
                    "Position": pos,
                    "Player": p["web_name"],
                    "Club": p["team_short"],
                    "Price (£m)": float(p["price"]),
                    "Set-Piece Duty": p.get("set_piece_duty", "-"),
                    "Selected %": float(p["selected_by_percent"]),
                    "Form": float(p["form"]),
                    "xGI / 90": float(p["xGI_per_90"]),
                    "Next Opponent": p["next_opp"],
                    "Next GW xP": float(p["xP_next"]),
                    f"{horizon}-GW xP": float(p["xP_horizon"]),
                    "Upcoming Schedule": p["horizon_fixtures"],
                }
            )
    watchlist_df = pd.DataFrame(watchlist_rows)

    return single_df.head(15), double_df, watchlist_df


def build_player_radar_df(
    players_df: pd.DataFrame,
    fixtures: List[Dict[str, Any]],
    current_gw: int,
    player_ids: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute 6-axis percentile radar scores (0-100) and raw comparison table for selected players."""
    if players_df.empty or not player_ids:
        return pd.DataFrame(), pd.DataFrame()

    df = players_df.copy()

    # Compute upcoming 5-GW average FDR per club
    fx_df = pd.DataFrame(fixtures)
    start_gw = min(38, current_gw + 1)
    end_gw = min(38, start_gw + 4)
    team_avg_fdr: Dict[int, float] = {}

    if not fx_df.empty and "event" in fx_df.columns:
        upc = fx_df[
            fx_df["event"].notna()
            & (fx_df["event"] >= start_gw)
            & (fx_df["event"] <= end_gw)
        ]
        for tid in df["team"].dropna().unique():
            tid_int = int(tid)
            h_fdr = upc[upc["team_h"] == tid_int]["team_h_difficulty"].tolist()
            a_fdr = upc[upc["team_a"] == tid_int]["team_a_difficulty"].tolist()
            all_fdrs = h_fdr + a_fdr
            team_avg_fdr[tid_int] = round(float(np.mean(all_fdrs)), 2) if all_fdrs else 3.5

    df["avg_5gw_fdr"] = df["team"].astype(int).map(team_avg_fdr).fillna(3.5)
    df["fixture_ease"] = (5.0 - df["avg_5gw_fdr"]).clip(lower=0.5, upper=4.5)

    # Reference pool of active players (at least 180 mins, or >0 mins fallback)
    active_mask = df["minutes"] >= 180
    if active_mask.sum() < 30:
        active_mask = df["minutes"] > 0

    metrics = [
        ("xG_per_90", "Goal Threat (xG/90)"),
        ("xA_per_90", "Creativity (xA/90)"),
        ("bps_per_90", "Bonus Magnet (BPS/90)"),
        ("form", "Recent Form"),
        ("points_per_million", "Value (Pts / £m)"),
        ("fixture_ease", "Fixture Ease (Next 5 GW)"),
    ]

    for col, _ in metrics:
        ref_vals = df.loc[active_mask, col].values
        if len(ref_vals) > 0:
            # Percentile rank (0 to 100) relative to active player pool
            df[f"{col}_pct"] = (
                df[col].apply(lambda v: float(np.mean(ref_vals <= v) * 100.0)).round(1)
            )
        else:
            df[f"{col}_pct"] = 50.0

    sel_df = df[df["id"].astype(int).isin([int(pid) for pid in player_ids])].copy()
    if sel_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    radar_rows = []
    for _, row in sel_df.iterrows():
        p_label = f"{row['web_name']} ({row['team_short']} £{row['price']:.1f}m)"
        for col, axis_label in metrics:
            radar_rows.append(
                {
                    "Player": p_label,
                    "Axis": axis_label,
                    "Percentile": float(row[f"{col}_pct"]),
                    "Raw_Value": float(row["avg_5gw_fdr"]) if col == "fixture_ease" else float(row[col]),
                }
            )

    raw_table = sel_df[
        [
            "web_name",
            "team_short",
            "position",
            "price",
            "selected_by_percent",
            "set_piece_duty",
            "price_trend",
            "net_transfers_gw",
            "minutes",
            "total_points",
            "form",
            "xG_per_90",
            "xA_per_90",
            "xGI_per_90",
            "bps_per_90",
            "points_per_million",
            "avg_5gw_fdr",
        ]
    ].rename(
        columns={
            "web_name": "Player",
            "team_short": "Club",
            "position": "Pos",
            "price": "Price (£m)",
            "selected_by_percent": "TSB %",
            "set_piece_duty": "Set Pieces",
            "price_trend": "Price Trend",
            "net_transfers_gw": "Net Transfers (GW)",
            "minutes": "Minutes",
            "total_points": "Total Pts",
            "form": "Form",
            "xG_per_90": "xG / 90",
            "xA_per_90": "xA / 90",
            "xGI_per_90": "xGI / 90",
            "bps_per_90": "BPS / 90",
            "points_per_million": "Pts / £m",
            "avg_5gw_fdr": "Next 5 GW Avg FDR",
        }
    )

    return pd.DataFrame(radar_rows), raw_table.reset_index(drop=True)


