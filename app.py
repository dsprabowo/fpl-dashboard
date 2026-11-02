"""Streamlit FPL Analytics Dashboard — Dark Sports Broadcast & Glassmorphism UI."""

from datetime import datetime, timedelta, timezone
import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.api import (
    get_bootstrap_static,
    get_classic_league_standings,
    get_club_elo,
    get_current_or_last_event,
    get_event_live,
    get_fixtures,
    get_manager_history,
    get_manager_info,
    get_manager_picks,
    get_player_summary,
)
from src.processor import (
    build_club_and_opponent_stats,
    build_fixture_ticker,
    build_gameweek_matchups,
    build_gw_live_player_table,
    build_player_radar_df,
    build_players_df,
    build_teams_lookup,
    compute_player_xp,
    evaluate_chip_strategy,
    format_player_element_summary,
    optimize_starting_xi,
    recommend_transfers,
)

st.set_page_config(
    page_title="FPL Pro Analytics Studio",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Official Premier League Club Primary Kit Colors
CLUB_KIT_COLORS = {
    "ARS": "#EF0107",
    "AVL": "#670E36",
    "BOU": "#DA291C",
    "BRE": "#E30613",
    "BHA": "#0057B8",
    "CHE": "#034694",
    "CRY": "#1B458F",
    "EVE": "#003399",
    "FUL": "#475569",
    "IPS": "#3A64A3",
    "LEI": "#003090",
    "LIV": "#C8102E",
    "MCI": "#6CABDD",
    "MUN": "#DA291C",
    "NEW": "#334155",
    "NFO": "#DD0000",
    "SOU": "#D71920",
    "SUN": "#EB172B",
    "TOT": "#3b82f6",
    "WHU": "#7A263A",
    "WOL": "#FDB913",
    "BUR": "#6C1D45",
    "LEE": "#FFCD00",
    "SHU": "#EE2737",
    "LUT": "#F78F1E",
}

FDR_HEX_MAP = {
    1: "#16a34a",
    2: "#22c55e",
    3: "#64748b",
    4: "#f97316",
    5: "#dc2626",
}


def inject_custom_css() -> None:
    """Inject Dark Sports Broadcast & Glassmorphism UI stylesheet."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        /* Hide default Streamlit chrome */
        #MainMenu, footer {
            visibility: hidden;
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        .stApp {
            background: radial-gradient(circle at 12% 8%, rgba(55, 0, 60, 0.42) 0%, rgba(11, 15, 25, 0) 42%),
                        radial-gradient(circle at 88% 12%, rgba(0, 255, 135, 0.08) 0%, rgba(11, 15, 25, 0) 38%),
                        #0b0f19 !important;
            color: #f8fafc !important;
        }
        .block-container {
            padding-top: 1.1rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1440px !important;
        }
        /* Sidebar Dark Glass */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #0b0f19 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }
        /* Broadcast Hero Header */
        .fpl-hero-banner {
            background: linear-gradient(130deg, rgba(55, 0, 60, 0.92) 0%, rgba(19, 27, 46, 0.95) 55%, rgba(15, 23, 42, 0.96) 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-bottom: 3px solid #00ff87;
            border-radius: 16px;
            padding: 1.2rem 1.6rem;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.45);
            margin-bottom: 1.25rem;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
        }
        .fpl-hero-title {
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 0.55rem;
        }
        .fpl-hero-sub {
            font-size: 0.84rem;
            color: #94a3b8;
            margin-top: 0.22rem;
            font-weight: 500;
        }
        .fpl-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            align-items: center;
        }
        .fpl-pill {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.16);
            color: #e2e8f0;
            padding: 0.36rem 0.82rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            backdrop-filter: blur(8px);
        }
        .fpl-pill-accent {
            background: linear-gradient(90deg, #00ff87 0%, #04f5ff 100%);
            color: #0b0f19;
            padding: 0.36rem 0.9rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            box-shadow: 0 0 16px rgba(0, 255, 135, 0.35);
        }

        /* Custom HTML Player Spotlight Cards Grid */
        .spotlight-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(235px, 1fr));
            gap: 0.95rem;
            margin: 0.65rem 0 1.2rem 0;
        }
        .spotlight-card {
            background: linear-gradient(155deg, rgba(24, 34, 56, 0.92) 0%, rgba(15, 23, 42, 0.96) 100%);
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-radius: 14px;
            padding: 0.9rem 1.05rem;
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.32);
            transition: transform 0.16s ease, border-color 0.16s ease;
        }
        .spotlight-card:hover {
            transform: translateY(-2px);
            border-color: rgba(0, 255, 135, 0.45);
        }
        .spotlight-accent-bar {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
        }
        .spotlight-info {
            flex: 1;
            min-width: 0;
        }
        .spotlight-label {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #94a3b8;
            margin-bottom: 0.2rem;
        }
        .spotlight-name {
            font-size: 1.05rem;
            font-weight: 800;
            color: #ffffff;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .spotlight-stat-row {
            display: flex;
            align-items: baseline;
            gap: 0.45rem;
            margin-top: 0.3rem;
        }
        .spotlight-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.32rem;
            font-weight: 800;
            color: #00ff87;
        }
        .spotlight-sub {
            font-size: 0.75rem;
            font-weight: 600;
            color: #cbd5e1;
            background: rgba(255, 255, 255, 0.07);
            padding: 0.12rem 0.48rem;
            border-radius: 6px;
        }
        .spotlight-avatar-wrap {
            position: relative;
            width: 58px;
            height: 68px;
            flex-shrink: 0;
        }
        .spotlight-photo {
            width: 56px;
            height: 68px;
            object-fit: cover;
            object-position: top;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.04);
        }
        .spotlight-crest {
            position: absolute;
            bottom: -2px;
            right: -4px;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: #0f172a;
            padding: 2px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        /* Pure HTML/CSS 2D Football Pitch with Real Player Photos */
        .fpl-pitch-stage {
            background: repeating-linear-gradient(
                180deg,
                #14532d 0px,
                #14532d 62px,
                #166534 62px,
                #166534 124px
            );
            border: 2px solid rgba(255, 255, 255, 0.72);
            border-radius: 16px 16px 0 0;
            padding: 1.15rem 0.6rem 1.25rem 0.6rem;
            position: relative;
            box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.35), 0 10px 28px rgba(0, 0, 0, 0.4);
            overflow: hidden;
        }
        /* Halfway line and center circle overlay */
        .fpl-pitch-halfline {
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 1.5px;
            background: rgba(255, 255, 255, 0.42);
            z-index: 0;
        }
        .fpl-pitch-circle {
            position: absolute;
            top: 50%;
            left: 50%;
            width: 96px;
            height: 96px;
            transform: translate(-50%, -50%);
            border: 1.5px solid rgba(255, 255, 255, 0.42);
            border-radius: 50%;
            z-index: 0;
        }
        .fpl-pitch-box-top {
            position: absolute;
            top: 0;
            left: 26%;
            width: 48%;
            height: 68px;
            border: 1.5px solid rgba(255, 255, 255, 0.42);
            border-top: none;
            z-index: 0;
        }
        .fpl-pitch-box-bottom {
            position: absolute;
            bottom: 0;
            left: 26%;
            width: 48%;
            height: 68px;
            border: 1.5px solid rgba(255, 255, 255, 0.42);
            border-bottom: none;
            z-index: 0;
        }
        .fpl-pitch-row {
            display: flex;
            justify-content: space-evenly;
            align-items: center;
            margin: 0.45rem 0;
            position: relative;
            z-index: 2;
        }
        .fpl-dugout-bar {
            background: linear-gradient(180deg, #0f172a 0%, #131b2e 100%);
            border: 2px solid #334155;
            border-top: none;
            border-radius: 0 0 16px 16px;
            padding: 0.85rem 0.6rem 0.9rem 0.6rem;
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.45);
            margin-bottom: 0.6rem;
        }
        .fpl-dugout-title {
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: #94a3b8;
            text-align: center;
            margin-bottom: 0.45rem;
        }
        .fpl-pcard {
            width: 96px;
            background: rgba(15, 23, 42, 0.90);
            border: 1.5px solid rgba(255, 255, 255, 0.22);
            border-radius: 10px;
            text-align: center;
            position: relative;
            box-shadow: 0 6px 14px rgba(0, 0, 0, 0.45);
            backdrop-filter: blur(6px);
            overflow: hidden;
            transition: transform 0.14s ease;
        }
        .fpl-pcard:hover {
            transform: scale(1.06);
            z-index: 5;
        }
        .fpl-pcard-cap {
            border: 2px solid #fbbf24 !important;
            box-shadow: 0 0 14px rgba(251, 191, 36, 0.45) !important;
        }
        .fpl-pcard-vc {
            border: 2px solid #38bdf8 !important;
        }
        .fpl-pcard-new {
            border: 2px solid #00ff87 !important;
            box-shadow: 0 0 14px rgba(0, 255, 135, 0.42) !important;
        }
        .fpl-pcard-badge {
            position: absolute;
            top: 3px;
            left: 3px;
            font-size: 0.62rem;
            font-weight: 800;
            padding: 1px 5px;
            border-radius: 4px;
            z-index: 3;
        }
        .fpl-pcard-crest {
            position: absolute;
            top: 3px;
            right: 3px;
            width: 17px;
            height: 17px;
            z-index: 3;
        }
        .fpl-pcard-img-wrap {
            height: 48px;
            display: flex;
            justify-content: center;
            align-items: flex-end;
            padding-top: 4px;
        }
        .fpl-pcard-img {
            height: 45px;
            width: auto;
            object-fit: contain;
        }
        .fpl-pcard-name {
            background: #f8fafc;
            color: #0f172a;
            font-size: 0.72rem;
            font-weight: 800;
            padding: 2px 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .fpl-pcard-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 3px 5px;
            font-size: 0.67rem;
            font-weight: 700;
            background: #0f172a;
            color: #e2e8f0;
        }
        .fpl-pcard-xp {
            font-family: 'JetBrains Mono', monospace;
            color: #00ff87;
            font-weight: 800;
        }

        /* Sofascore-Style Matchup Cards in Tab 3 */
        .matchup-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
            gap: 0.8rem;
            margin: 0.7rem 0 1.2rem 0;
        }
        .matchup-card {
            background: linear-gradient(145deg, rgba(24, 34, 56, 0.90) 0%, rgba(15, 23, 42, 0.96) 100%);
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-radius: 12px;
            padding: 0.75rem 0.95rem;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.28);
        }
        .matchup-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
        }
        .matchup-team {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-weight: 700;
            font-size: 0.9rem;
            color: #f8fafc;
            width: 38%;
        }
        .matchup-team-right {
            justify-content: flex-end;
            text-align: right;
        }
        .matchup-crest {
            width: 26px;
            height: 26px;
            object-fit: contain;
        }
        .matchup-center {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            font-weight: 800;
            background: rgba(0, 255, 135, 0.12);
            border: 1px solid rgba(0, 255, 135, 0.35);
            color: #00ff87;
            padding: 0.22rem 0.55rem;
            border-radius: 8px;
            text-align: center;
        }
        .matchup-bar-bg {
            height: 6px;
            background: #1e293b;
            border-radius: 999px;
            overflow: hidden;
            display: flex;
            margin: 0.5rem 0 0.35rem 0;
        }
        .matchup-subrow {
            display: flex;
            justify-content: space-between;
            font-size: 0.71rem;
            color: #94a3b8;
            font-weight: 600;
        }

        /* Metric Card Fallback Styling */
        div[data-testid="stMetric"] {
            background: linear-gradient(155deg, rgba(24, 34, 56, 0.88) 0%, rgba(15, 23, 42, 0.94) 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.09) !important;
            border-left: 4px solid #00ff87 !important;
            border-radius: 12px !important;
            padding: 0.75rem 1rem !important;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.28) !important;
        }
        div[data-testid="stMetricLabel"] > div {
            font-weight: 700 !important;
            color: #94a3b8 !important;
            font-size: 0.76rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        div[data-testid="stMetricValue"] > div {
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 800 !important;
            color: #ffffff !important;
            font-size: 1.3rem !important;
        }

        /* Modern Broadcast Pill Tabs */
        div[data-baseweb="tab-list"] {
            gap: 0.45rem !important;
            background: rgba(19, 27, 46, 0.75) !important;
            padding: 0.38rem !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.07) !important;
        }
        button[data-baseweb="tab"] {
            font-weight: 700 !important;
            font-size: 0.88rem !important;
            padding: 0.5rem 1.05rem !important;
            border-radius: 8px !important;
            color: #94a3b8 !important;
            border: none !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #0b0f19 !important;
            background: linear-gradient(90deg, #00ff87 0%, #04f5ff 100%) !important;
            font-weight: 800 !important;
            box-shadow: 0 4px 12px rgba(0, 255, 135, 0.28) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_broadcast_plotly_theme(fig: go.Figure, height: int = 440) -> go.Figure:
    """Apply unified Dark Sports Broadcast theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(19, 27, 46, 0.65)",
        plot_bgcolor="rgba(11, 15, 25, 0.45)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#e2e8f0", size=12),
        title_font=dict(family="Plus Jakarta Sans, sans-serif", color="#ffffff", size=15),
        height=height,
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="#00ff87",
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#f8fafc", size=12),
        ),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.10)", zerolinecolor="rgba(148, 163, 184, 0.15)"),
        yaxis=dict(gridcolor="rgba(148, 163, 184, 0.10)", zerolinecolor="rgba(148, 163, 184, 0.15)"),
    )
    return fig


def render_spotlight_cards(cards: list) -> None:
    """Render a responsive grid of HTML Player Spotlight Cards with headshots, club crests, and club glow."""
    fallback_avatar = "https://resources.premierleague.com/premierleague/photos/players/110x140/Photo-Missing.png"
    html_parts = ['<div class="spotlight-grid">']
    for c in cards:
        title = html.escape(str(c.get("title", "")))
        name = html.escape(str(c.get("name", "-")))
        stat_val = html.escape(str(c.get("stat", "")))
        sub_val = html.escape(str(c.get("sub", "")))
        team_short = str(c.get("team_short", ""))
        accent = CLUB_KIT_COLORS.get(team_short, "#00ff87")
        photo_url = str(c.get("photo_url", "") or fallback_avatar)
        crest_url = str(c.get("crest_url", ""))
        crest_img = (
            f'<img src="{crest_url}" class="spotlight-crest" onerror="this.style.display=\'none\'"/>'
            if crest_url
            else ""
        )
        html_parts.append(
            f'<div class="spotlight-card">'
            f'<div class="spotlight-accent-bar" style="background: linear-gradient(90deg, {accent}, #00ff87);"></div>'
            f'<div class="spotlight-info">'
            f'<div class="spotlight-label">{title}</div>'
            f'<div class="spotlight-name">{name}</div>'
            f'<div class="spotlight-stat-row">'
            f'<span class="spotlight-val">{stat_val}</span>'
            f'<span class="spotlight-sub">{sub_val}</span>'
            f"</div></div>"
            f'<div class="spotlight-avatar-wrap">'
            f'<img src="{photo_url}" class="spotlight-photo" onerror="this.src=\'{fallback_avatar}\'"/>'
            f"{crest_img}"
            f"</div></div>"
        )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_html_pitch(opt_sq: pd.DataFrame, formation: str, next_gw: int) -> None:
    """Render a pure HTML/CSS 2D Football Pitch with real Premier League player headshots and club crests."""
    fallback_avatar = "https://resources.premierleague.com/premierleague/photos/players/110x140/Photo-Missing.png"
    if "is_opt_starter" in opt_sq.columns:
        starter_mask = opt_sq["is_opt_starter"].astype(bool)
    elif "Is_Starter" in opt_sq.columns:
        starter_mask = opt_sq["Is_Starter"].astype(bool)
    else:
        starter_mask = ~opt_sq["Recommended_Role"].astype(str).str.startswith("Bench")

    starters = opt_sq[starter_mask].copy()
    bench = opt_sq[~starter_mask].copy()

    def _player_card_html(row: pd.Series, is_bench: bool = False) -> str:
        role = str(row.get("Recommended_Role", "Starter"))
        curr_role = str(row.get("Current_Role", ""))
        is_new_in = "🆕" in curr_role
        is_cap = ("Captain" in role) and ("Vice" not in role)
        is_vc = ("VC" in role) or ("Vice" in role)

        card_cls = "fpl-pcard"
        badge_html = ""
        if is_cap:
            card_cls += " fpl-pcard-cap"
            badge_html = '<span class="fpl-pcard-badge" style="background:#fbbf24;color:#0f172a;">© CAP</span>'
        elif is_vc:
            card_cls += " fpl-pcard-vc"
            badge_html = '<span class="fpl-pcard-badge" style="background:#38bdf8;color:#0f172a;">VC</span>'
        elif is_new_in:
            card_cls += " fpl-pcard-new"
            badge_html = '<span class="fpl-pcard-badge" style="background:#00ff87;color:#0f172a;">🆕 IN</span>'
        elif is_bench:
            b_short = role.replace("Bench ", "B")
            badge_html = f'<span class="fpl-pcard-badge" style="background:#334155;color:#cbd5e1;">{html.escape(b_short)}</span>'

        photo = str(row.get("photo_url", "") or fallback_avatar)
        crest = str(row.get("crest_url", ""))
        crest_html = (
            f'<img src="{crest}" class="fpl-pcard-crest" onerror="this.style.display=\'none\'"/>'
            if crest
            else ""
        )
        name = html.escape(str(row.get("web_name", "-")))
        opp = html.escape(str(row.get("next_opp", "-")))
        fdr_val = int(round(float(row.get("next_fdr", 3))))
        fdr_bg = FDR_HEX_MAP.get(fdr_val, "#475569")
        eff_xp = float(row.get("Effective_xP", row.get("xP_next", 0.0)))
        base_xp = float(row.get("xP_next", 0.0))
        shown_xp = base_xp if is_bench else eff_xp

        return (
            f'<div class="{card_cls}" title="{name} ({row.get("team_short", "")}) • £{float(row.get("price", 0)):.1f}m • Base xP: {base_xp:.2f}">'
            f"{badge_html}{crest_html}"
            f'<div class="fpl-pcard-img-wrap">'
            f'<img src="{photo}" class="fpl-pcard-img" onerror="this.src=\'{fallback_avatar}\'"/>'
            f"</div>"
            f'<div class="fpl-pcard-name">{name}</div>'
            f'<div class="fpl-pcard-meta">'
            f'<span style="background:{fdr_bg};color:#fff;padding:1px 4px;border-radius:3px;font-size:0.62rem;">{opp}</span>'
            f'<span class="fpl-pcard-xp">{shown_xp:.1f}p</span>'
            f"</div></div>"
        )

    rows_html = []
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        pos_group = starters[starters["position"] == pos]
        if pos_group.empty:
            continue
        cards_str = "".join(_player_card_html(r, is_bench=False) for _, r in pos_group.iterrows())
        rows_html.append(f'<div class="fpl-pitch-row">{cards_str}</div>')

    bench_cards_str = "".join(_player_card_html(r, is_bench=True) for _, r in bench.iterrows())

    full_html = (
        f'<div class="fpl-pitch-stage">'
        f'<div class="fpl-pitch-box-top"></div>'
        f'<div class="fpl-pitch-halfline"></div>'
        f'<div class="fpl-pitch-circle"></div>'
        f'<div class="fpl-pitch-box-bottom"></div>'
        f'{"".join(rows_html)}'
        f"</div>"
        f'<div class="fpl-dugout-bar">'
        f'<div class="fpl-dugout-title">🪑 Substitute Bench (GW {next_gw} Projections)</div>'
        f'<div class="fpl-pitch-row">{bench_cards_str}</div>'
        f"</div>"
    )
    st.markdown(full_html, unsafe_allow_html=True)


def render_matchup_cards_html(matchups_df: pd.DataFrame) -> None:
    """Render Sofascore/FotMob-style Head-to-Head Fixture Projection Cards."""
    parts = ['<div class="matchup-grid">']
    for _, m in matchups_df.iterrows():
        h_short = html.escape(str(m.get("Home Short", m.get("Home", "HOME"))))
        a_short = html.escape(str(m.get("Away Short", m.get("Away", "AWAY"))))
        h_crest = str(m.get("Home Crest", ""))
        a_crest = str(m.get("Away Crest", ""))
        h_xg = float(m.get("Home Proj G", 1.2))
        a_xg = float(m.get("Away Proj G", 1.0))
        tot_xg = max(0.1, h_xg + a_xg)
        h_pct = round(100.0 * h_xg / tot_xg, 1)
        a_pct = round(100.0 - h_pct, 1)
        h_cs = float(m.get("Home CS %", 25.0))
        a_cs = float(m.get("Away CS %", 20.0))
        score_str = str(m.get("Score", "vs"))
        center_badge = score_str if score_str != "vs" else f"{h_xg:.2f} – {a_xg:.2f} λ"

        h_img = f'<img src="{h_crest}" class="matchup-crest" onerror="this.style.display=\'none\'"/>' if h_crest else ""
        a_img = f'<img src="{a_crest}" class="matchup-crest" onerror="this.style.display=\'none\'"/>' if a_crest else ""

        parts.append(
            f'<div class="matchup-card">'
            f'<div class="matchup-row">'
            f'<div class="matchup-team">{h_img}<span>{h_short}</span></div>'
            f'<div class="matchup-center">{center_badge}</div>'
            f'<div class="matchup-team matchup-team-right"><span>{a_short}</span>{a_img}</div>'
            f"</div>"
            f'<div class="matchup-bar-bg">'
            f'<div style="width:{h_pct}%;background:#00ff87;"></div>'
            f'<div style="width:{a_pct}%;background:#38bdf8;"></div>'
            f"</div>"
            f'<div class="matchup-subrow">'
            f'<span>🛡️ CS: <b style="color:#00ff87">{h_cs:.0f}%</b> • Form: {html.escape(str(m.get("Home Form", "-")))}</span>'
            f'<span>Form: {html.escape(str(m.get("Away Form", "-")))} • CS: <b style="color:#38bdf8">{a_cs:.0f}%</b> 🛡️</span>'
            f"</div></div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def get_next_deadline_info(bootstrap: dict, next_gw: int) -> str:
    """Format the upcoming gameweek deadline in WIB (UTC+7) and remaining time."""
    events = bootstrap.get("events", [])
    target_ev = next((e for e in events if int(e.get("id", 0)) == int(next_gw)), None)
    if not target_ev or not target_ev.get("deadline_time"):
        return f"GW {next_gw}"
    try:
        dt_utc = datetime.fromisoformat(str(target_ev["deadline_time"]).replace("Z", "+00:00"))
        wib_tz = timezone(timedelta(hours=7))
        dt_wib = dt_utc.astimezone(wib_tz)
        now_utc = datetime.now(timezone.utc)
        delta = dt_utc - now_utc
        if delta.total_seconds() > 0:
            days = delta.days
            hours = delta.seconds // 3600
            mins = (delta.seconds % 3600) // 60
            countdown = f"in {days}d {hours}h" if days > 0 else f"in {hours}h {mins}m"
            return f"GW {next_gw}: {dt_wib.strftime('%d %b %H:%M')} WIB ({countdown})"
        return f"GW {next_gw}: {dt_wib.strftime('%d %b %H:%M')} WIB"
    except Exception:
        return f"GW {next_gw}"


inject_custom_css()

# --- Load Master Data ---
with st.spinner("Loading FPL Broadcast Studio..."):
    try:
        bootstrap = get_bootstrap_static()
        fixtures = get_fixtures()
        elo_map = get_club_elo()
    except Exception as exc:
        st.error(f"Failed to connect to the official FPL API: {exc}")
        st.stop()

current_gw = get_current_or_last_event(bootstrap)
next_gw = min(38, current_gw + 1)
deadline_badge = get_next_deadline_info(bootstrap, next_gw)
teams_df = build_teams_lookup(bootstrap)
players_df = build_players_df(bootstrap)

# --- Sidebar Controls (Global & Persistent Manager Context) ---
with st.sidebar:
    st.markdown("### 🏆 Manager Studio")
    manager_id_input = st.number_input(
        "FPL Manager ID",
        min_value=0,
        value=0,
        step=1,
        help="Enter your FPL Manager ID from fantasy.premierleague.com to load your 15-man squad globally across all tabs.",
    )
    sb_col1, sb_col2 = st.columns(2)
    with sb_col1:
        free_transfers_input = st.selectbox("Free Transfers", options=[1, 2, 3, 4, 5], index=0)
    with sb_col2:
        plan_horizon = st.selectbox("Horizon (GWs)", options=[3, 4, 5, 6], index=0)
    league_id_input = st.number_input("Mini-League ID (Optional)", min_value=0, value=0, step=1)

    st.divider()
    st.markdown("### ⚙️ Model Parameters")
    rolling_n = st.slider("Club Rolling Form Window", min_value=3, max_value=10, value=5)
    if st.button("🔄 Refresh Live API Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

club_df = build_club_and_opponent_stats(
    bootstrap, fixtures, players_df, rolling_window=rolling_n, elo_map=elo_map
)
all_players_xp = compute_player_xp(
    players_df, club_df, fixtures, current_gw=int(current_gw), horizon=int(plan_horizon)
)

# --- Load Manager Squad Globally so Owned Players are Highlighted Everywhere ---
mgr_info, mgr_hist, picks_data = {}, {}, {}
squad_df_raw = pd.DataFrame()
owned_player_ids = set()
bank_m = 0.0
squad_val_m = 100.0
mgr_load_error = None

if manager_id_input > 0:
    try:
        mgr_info = get_manager_info(int(manager_id_input))
        mgr_hist = get_manager_history(int(manager_id_input))
        picks_data = get_manager_picks(int(manager_id_input), int(current_gw))
        entry_hist = picks_data.get("entry_history", {})
        bank_m = round(float(entry_hist.get("bank", mgr_info.get("last_deadline_bank", 0))) / 10.0, 1)
        squad_val_m = round(float(entry_hist.get("value", mgr_info.get("last_deadline_value", 1000))) / 10.0, 1)
        picks_list = pd.DataFrame(picks_data.get("picks", []))
        if not picks_list.empty:
            owned_player_ids = set(picks_list["element"].astype(int).tolist())
            squad_df_raw = picks_list.copy()
    except Exception as exc:
        mgr_load_error = str(exc)

if st.session_state.get("sandbox_mgr_id") != int(manager_id_input):
    st.session_state["sandbox_mgr_id"] = int(manager_id_input)
    st.session_state["sandbox_transfers"] = []

sandbox_active_ids = set(owned_player_ids)
for mv in st.session_state.get("sandbox_transfers", []):
    sandbox_active_ids.discard(int(mv["out_id"]))
    sandbox_active_ids.add(int(mv["in_id"]))

players_df["squad_status"] = np.where(
    players_df["id"].astype(int).isin(sandbox_active_ids),
    np.where(players_df["id"].astype(int).isin(owned_player_ids), "⭐ Owned", "🆕 Sandbox In"),
    "",
)
if not all_players_xp.empty:
    all_players_xp["squad_status"] = np.where(
        all_players_xp["id"].astype(int).isin(sandbox_active_ids),
        np.where(all_players_xp["id"].astype(int).isin(owned_player_ids), "⭐ Owned", "🆕 Sandbox In"),
        "",
    )

with st.sidebar:
    if mgr_info:
        st.divider()
        st.markdown(
            f"**👤 {mgr_info.get('name', 'Squad Loaded')}**\n"
            f"- **Manager**: {mgr_info.get('player_first_name', '')} {mgr_info.get('player_last_name', '')}\n"
            f"- **Overall Rank**: `#{mgr_info.get('summary_overall_rank', 0):,}`\n"
            f"- **Squad / Bank**: `£{squad_val_m:.1f}m` / `£{bank_m:.1f}m ITB`\n"
            f"- **Sandbox Moves**: `{len(st.session_state.get('sandbox_transfers', []))}` active"
        )

# --- Top Broadcast Hero Header ---
mgr_badge_html = (
    f"<span class='fpl-pill'>👤 {html.escape(str(mgr_info.get('name', '')))} • Rank #{mgr_info.get('summary_overall_rank', 0):,}</span>"
    if mgr_info
    else ""
)
st.markdown(
    f"""
    <div class="fpl-hero-banner">
        <div>
            <div class="fpl-hero-title">⚽ FPL Pro Analytics Studio</div>
            <div class="fpl-hero-sub">Empirical Bayes xP Engine • Multiplicative Poisson & ClubElo • Live What-If Squad Sandbox</div>
        </div>
        <div class="fpl-badge-row">
            <span class="fpl-pill">Live: GW {current_gw}</span>
            <span class="fpl-pill-accent">⏰ {html.escape(deadline_badge)}</span>
            {mgr_badge_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_manager, tab_players, tab_clubs, tab_gw = st.tabs(
    [
        f"🏆 Squad Pitch & Sandbox ({len(st.session_state.get('sandbox_transfers', []))} Moves)"
        if st.session_state.get("sandbox_transfers")
        else "🏆 Squad Pitch & Transfer Sandbox",
        "🏃 Player Scouting & Radar",
        "🛡️ Club & Opponent Vulnerability",
        "📅 Fixture Ticker & Matchups",
    ]
)

# =====================================================================
# TAB 1: PLAYER ANALYTICS & DEEP DIVE
# =====================================================================
with tab_players:
    # Compact Toolbar instead of 5 bulky widget rows
    bar_c1, bar_c2, bar_c3, bar_c4 = st.columns([1.45, 1.35, 0.9, 0.9])
    with bar_c1:
        pos_filter = st.multiselect(
            "Position",
            options=["GKP", "DEF", "MID", "FWD"],
            default=["GKP", "DEF", "MID", "FWD"],
            label_visibility="collapsed",
        )
    with bar_c2:
        name_search = st.text_input(
            "Search Player",
            placeholder="🔍 Search player name (e.g. Salah, Palmer, Haaland)...",
            label_visibility="collapsed",
        )
    with bar_c3:
        with st.popover("🎛️ Filters & Columns", use_container_width=True):
            club_options = sorted(players_df["team_short"].dropna().unique().tolist())
            club_filter = st.multiselect("Filter by Club", options=club_options, default=[])
            max_price_val = float(players_df["price"].max()) if not players_df.empty else 15.0
            min_price_val = float(players_df["price"].min()) if not players_df.empty else 3.8
            price_limit = st.slider(
                "Max Price (£m)",
                min_value=min_price_val,
                max_value=max_price_val,
                value=max_price_val,
                step=0.1,
            )
            max_mins = int(players_df["minutes"].max()) if not players_df.empty else 3420
            default_min_mins = min(270, max(0, int(max_mins * 0.2)))
            min_minutes = st.slider(
                "Min Minutes Played",
                min_value=0,
                max_value=max(90, max_mins),
                value=default_min_mins,
                step=45,
            )
            table_preset = st.radio(
                "Table Column View",
                options=["⚡ Compact Essential (14 cols)", "🔬 Full Analytical (26 cols)"],
                index=0,
            )
    with bar_c4:
        only_my_squad = st.checkbox(
            "⭐ My Squad Only",
            value=False,
            disabled=(len(sandbox_active_ids) == 0),
        )

    filtered_p = players_df[
        (players_df["position"].isin(pos_filter))
        & (players_df["price"] <= price_limit)
        & (players_df["minutes"] >= min_minutes)
    ].copy()
    if club_filter:
        filtered_p = filtered_p[filtered_p["team_short"].isin(club_filter)]
    if name_search.strip():
        filtered_p = filtered_p[
            filtered_p["web_name"].str.contains(name_search.strip(), case=False, na=False)
        ]
    if only_my_squad:
        filtered_p = filtered_p[filtered_p["squad_status"] != ""]

    # Bespoke HTML Spotlight Cards with Player Headshots & Club Crests
    if not filtered_p.empty:
        top_pts = filtered_p.sort_values("total_points", ascending=False).iloc[0]
        top_xgi = filtered_p.sort_values("xGI_per_90", ascending=False).iloc[0]
        top_val = filtered_p.sort_values("points_per_million", ascending=False).iloc[0]
        top_form = filtered_p.sort_values("form", ascending=False).iloc[0]

        render_spotlight_cards(
            [
                {
                    "title": "🏆 Top Points Scorer",
                    "name": top_pts["web_name"],
                    "stat": f"{int(top_pts['total_points'])} pts",
                    "sub": f"£{top_pts['price']:.1f}m • {top_pts['team_short']}",
                    "team_short": top_pts["team_short"],
                    "photo_url": top_pts.get("photo_url", ""),
                    "crest_url": top_pts.get("crest_url", ""),
                },
                {
                    "title": "⚡ Highest xGI / 90",
                    "name": top_xgi["web_name"],
                    "stat": f"{top_xgi['xGI_per_90']:.2f}",
                    "sub": f"{int(top_xgi['minutes'])} mins • {top_xgi['team_short']}",
                    "team_short": top_xgi["team_short"],
                    "photo_url": top_xgi.get("photo_url", ""),
                    "crest_url": top_xgi.get("crest_url", ""),
                },
                {
                    "title": "💎 Best Value (Pts/£m)",
                    "name": top_val["web_name"],
                    "stat": f"{top_val['points_per_million']:.1f}",
                    "sub": f"£{top_val['price']:.1f}m • {top_val['position']}",
                    "team_short": top_val["team_short"],
                    "photo_url": top_val.get("photo_url", ""),
                    "crest_url": top_val.get("crest_url", ""),
                },
                {
                    "title": "🔥 Hottest Form (30d)",
                    "name": top_form["web_name"],
                    "stat": f"{top_form['form']:.1f}",
                    "sub": f"£{top_form['price']:.1f}m • {top_form['team_short']}",
                    "team_short": top_form["team_short"],
                    "photo_url": top_form.get("photo_url", ""),
                    "crest_url": top_form.get("crest_url", ""),
                },
            ]
        )

    table_df = filtered_p.sort_values("total_points", ascending=False).reset_index(drop=True)

    compact_cols = [
        "photo_url",
        "crest_url",
        "web_name",
        "squad_status",
        "team_short",
        "position",
        "price",
        "selected_by_percent",
        "price_trend",
        "set_piece_duty",
        "total_points",
        "form",
        "xGI_per_90",
        "points_per_million",
    ]
    full_cols = compact_cols + [
        "net_transfers_gw",
        "minutes",
        "pts_per_90",
        "goals_scored",
        "assists",
        "xG",
        "xA",
        "xGI",
        "xG_per_90",
        "xA_per_90",
        "xGI_delta",
        "clean_sheets",
        "xGC_per_90",
        "bps_per_90",
        "bonus",
    ]
    active_cols = [
        c
        for c in (compact_cols if "Compact" in table_preset else full_cols)
        if c in table_df.columns
    ]

    st.caption(
        f"📊 Showing **{len(table_df)}** players • 🖱️ **Click any row** to inspect that player in the Deep Dive & Radar below."
    )
    table_event = st.dataframe(
        table_df[active_cols],
        column_config={
            "photo_url": st.column_config.ImageColumn("Photo", width="small"),
            "crest_url": st.column_config.ImageColumn("Club", width="small"),
            "web_name": st.column_config.TextColumn("Player"),
            "squad_status": st.column_config.TextColumn("Squad"),
            "team_short": st.column_config.TextColumn("Team"),
            "position": st.column_config.TextColumn("Pos"),
            "price": st.column_config.NumberColumn("Price", format="£%.1fm"),
            "selected_by_percent": st.column_config.ProgressColumn(
                "TSB %", format="%.1f%%", min_value=0.0, max_value=100.0
            ),
            "price_trend": st.column_config.TextColumn("Price Trend"),
            "net_transfers_gw": st.column_config.NumberColumn("GW Net Transfers", format="%+d"),
            "set_piece_duty": st.column_config.TextColumn("Set Pieces"),
            "points_per_million": st.column_config.NumberColumn("Pts / £m", format="%.2f"),
            "xGI_per_90": st.column_config.ProgressColumn(
                "xGI / 90", format="%.2f", min_value=0.0, max_value=1.5
            ),
        },
        use_container_width=True,
        hide_index=True,
        height=400,
        on_select="rerun",
        selection_mode="single-row",
        key="master_player_table",
    )

    clicked_player_id = None
    if table_event and hasattr(table_event, "selection") and table_event.selection.rows:
        row_idx = int(table_event.selection.rows[0])
        if 0 <= row_idx < len(table_df):
            clicked_player_id = int(table_df.iloc[row_idx]["id"])

    p_sub1, p_sub2, p_sub3 = st.tabs(
        [
            "🔍 Single-Player Deep Dive & Match Log",
            "🕸️ Head-to-Head Percentile Radar (2–4 Players)",
            "📈 Custom 2D Metric Scatter Explorer",
        ]
    )

    player_options = players_df.sort_values("total_points", ascending=False)
    player_label_map = {
        f"{row['web_name']} ({row['team_short']} - {row['position']} - £{row['price']:.1f}m)"
        + (f" [{row['squad_status']}]" if row.get("squad_status") else ""): int(row["id"])
        for _, row in player_options.iterrows()
    }
    id_to_label_map = {v: k for k, v in player_label_map.items()}

    with p_sub1:
        default_deep_idx = 0
        if clicked_player_id and clicked_player_id in id_to_label_map:
            target_lbl = id_to_label_map[clicked_player_id]
            default_deep_idx = list(player_label_map.keys()).index(target_lbl)

        selected_label = st.selectbox(
            "Select Player to Inspect (or click any row in the Master Table above)",
            options=list(player_label_map.keys()),
            index=default_deep_idx,
        )
        if selected_label:
            sel_pid = player_label_map[selected_label]
            summary_data = get_player_summary(sel_pid)
            hist_df, upc_df = format_player_element_summary(summary_data, teams_df)

            p_col1, p_col2 = st.columns([2, 1])
            with p_col1:
                if not hist_df.empty:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(
                        go.Bar(
                            x=hist_df["GW"].astype(str) + " " + hist_df["match"],
                            y=hist_df["Points"],
                            name="FPL Points",
                            marker_color="#00ff87",
                        )
                    )
                    fig_hist.add_trace(
                        go.Scatter(
                            x=hist_df["GW"].astype(str) + " " + hist_df["match"],
                            y=hist_df["xGI"],
                            name="xGI (Expected Goal Inv.)",
                            yaxis="y2",
                            mode="lines+markers",
                            line=dict(color="#04f5ff", width=3),
                        )
                    )
                    fig_hist.update_layout(
                        title=f"{selected_label} — Gameweek Points vs. xGI",
                        yaxis=dict(title="FPL Points"),
                        yaxis2=dict(title="xGI", overlaying="y", side="right"),
                        legend=dict(orientation="h", y=1.12),
                    )
                    apply_broadcast_plotly_theme(fig_hist, height=350)
                    st.plotly_chart(fig_hist, use_container_width=True)

                    hist_cols = [
                        "GW",
                        "match",
                        "Mins",
                        "Points",
                        "G",
                        "A",
                        "xG",
                        "xA",
                        "xGI",
                        "CS",
                        "GC",
                        "xGC",
                        "BPS",
                        "Bonus",
                        "Price",
                    ]
                    st.dataframe(
                        hist_df.sort_values("GW", ascending=False)[[c for c in hist_cols if c in hist_df.columns]],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No past gameweek appearances recorded yet for this player.")

            with p_col2:
                st.markdown("**Upcoming Fixtures & Difficulty (FDR)**")
                if not upc_df.empty:
                    st.dataframe(
                        upc_df[["GW", "Fixture", "FDR", "kickoff_time"]].head(8),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No upcoming fixtures available.")

    with p_sub2:
        default_radar_labels = list(player_label_map.keys())[:3]
        if clicked_player_id and clicked_player_id in id_to_label_map:
            clicked_lbl = id_to_label_map[clicked_player_id]
            if clicked_lbl not in default_radar_labels:
                default_radar_labels = [clicked_lbl] + default_radar_labels[:2]

        radar_selected_labels = st.multiselect(
            "Select 2–4 Players to Compare (Percentiles vs. Active Premier League Pool)",
            options=list(player_label_map.keys()),
            default=default_radar_labels,
            max_selections=4,
        )
        if radar_selected_labels:
            radar_pids = [player_label_map[lbl] for lbl in radar_selected_labels]
            radar_df, radar_raw_df = build_player_radar_df(
                players_df, fixtures, current_gw=int(current_gw), player_ids=radar_pids
            )
            if not radar_df.empty:
                r_col1, r_col2 = st.columns([1.15, 1.35])
                with r_col1:
                    fig_radar = go.Figure()
                    palette = ["#00ff87", "#04f5ff", "#f59e0b", "#ec4899"]
                    for idx, p_name in enumerate(radar_df["Player"].unique()):
                        sub = radar_df[radar_df["Player"] == p_name]
                        theta = sub["Axis"].tolist()
                        r_vals = sub["Percentile"].tolist()
                        raw_vals = sub["Raw_Value"].tolist()
                        theta.append(theta[0])
                        r_vals.append(r_vals[0])
                        raw_vals.append(raw_vals[0])
                        fig_radar.add_trace(
                            go.Scatterpolar(
                                r=r_vals,
                                theta=theta,
                                fill="toself",
                                name=p_name,
                                line=dict(color=palette[idx % len(palette)], width=2.5),
                                customdata=raw_vals,
                                hovertemplate="<b>%{theta}</b><br>Percentile: %{r:.1f}%<br>Raw Stat: %{customdata:.2f}<extra>%{fullData.name}</extra>",
                            )
                        )
                    fig_radar.update_layout(
                        polar=dict(
                            bgcolor="rgba(11, 15, 25, 0.55)",
                            radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%", gridcolor="rgba(148,163,184,0.18)"),
                            angularaxis=dict(gridcolor="rgba(148,163,184,0.18)"),
                        ),
                        title="6-Axis Percentile Radar (0–100%)",
                        margin=dict(l=45, r=45, t=55, b=35),
                        legend=dict(orientation="h", y=-0.12),
                    )
                    apply_broadcast_plotly_theme(fig_radar, height=420)
                    st.plotly_chart(fig_radar, use_container_width=True)
                with r_col2:
                    st.markdown("**Side-by-Side Metric & Price Velocity Comparison**")
                    st.dataframe(
                        radar_raw_df,
                        column_config={
                            "Price (£m)": st.column_config.NumberColumn("Price", format="£%.1fm"),
                            "TSB %": st.column_config.ProgressColumn(
                                "TSB %", format="%.1f%%", min_value=0.0, max_value=100.0
                            ),
                            "Net Transfers (GW)": st.column_config.NumberColumn(
                                "GW Net Transfers", format="%+d"
                            ),
                            "xGI / 90": st.column_config.ProgressColumn(
                                "xGI / 90", format="%.2f", min_value=0.0, max_value=1.5
                            ),
                        },
                        use_container_width=True,
                        hide_index=True,
                    )

    with p_sub3:
        sc_col1, sc_col2 = st.columns([1, 3])
        metric_choices = [
            "price",
            "total_points",
            "form",
            "points_per_million",
            "xG",
            "xA",
            "xGI",
            "xG_per_90",
            "xA_per_90",
            "xGI_per_90",
            "pts_per_90",
            "bps_per_90",
            "ict_index",
            "selected_by_percent",
        ]
        with sc_col1:
            x_axis = st.selectbox("X-Axis Metric", options=metric_choices, index=metric_choices.index("xGI_per_90"))
            y_axis = st.selectbox("Y-Axis Metric", options=metric_choices, index=metric_choices.index("pts_per_90"))
            size_axis = st.selectbox("Bubble Size", options=["price", "total_points", "minutes", "selected_by_percent"], index=0)

        with sc_col2:
            if not filtered_p.empty:
                fig_scatter = px.scatter(
                    filtered_p,
                    x=x_axis,
                    y=y_axis,
                    color="position",
                    size=size_axis,
                    hover_name="web_name",
                    hover_data=["team_short", "squad_status", "price", "minutes", "total_points", "xGI_per_90"],
                    text="web_name" if len(filtered_p) <= 35 else None,
                    title=f"Player Comparison: {y_axis} vs. {x_axis} (Filtered: {len(filtered_p)} players)",
                    color_discrete_sequence=["#00ff87", "#04f5ff", "#f59e0b", "#ec4899"],
                )
                if len(filtered_p) <= 35:
                    fig_scatter.update_traces(textposition="top center")
                apply_broadcast_plotly_theme(fig_scatter, height=440)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("No players match the current filter criteria.")

# =====================================================================
# TAB 2: CLUB & OPPONENT ANALYSIS
# =====================================================================
with tab_clubs:
    st.subheader(f"Club Performance & Opponent Vulnerability (Rolling Window: Last {rolling_n} Matches)")

    if not club_df.empty:
        c_col1, c_col2 = st.columns([2.8, 2.2])
        with c_col1:
            avg_xg = club_df["xG_per_match"].mean()
            avg_xgc = club_df["xGC_per_match"].mean()
            fig_quad = px.scatter(
                club_df,
                x="xG_per_match",
                y="xGC_per_match",
                text="team_short",
                size="GF_per_match",
                color="Elo",
                color_continuous_scale="Turbo",
                hover_name="team_name",
                hover_data=["Elo", "MP", "GF", "GA", "CS", "last_n_form", "last_n_GF_pm", "last_n_GA_pm"],
                title="Club Attack (xG/Match) vs. Defense Conceded (xGC/Match) — Lower Y is Stronger Defense",
            )
            fig_quad.update_traces(textposition="top center")
            fig_quad.add_vline(x=avg_xg, line_dash="dash", line_color="#94a3b8", annotation_text="Avg xG/m")
            fig_quad.add_hline(y=avg_xgc, line_dash="dash", line_color="#94a3b8", annotation_text="Avg xGC/m")
            fig_quad.update_yaxes(autorange="reversed")
            apply_broadcast_plotly_theme(fig_quad, height=500)
            st.plotly_chart(fig_quad, use_container_width=True)

        with c_col2:
            st.markdown("🎯 **Top 8 Weakest Defenses (Target with Your Attackers)**")
            weak_def = club_df.sort_values(
                ["attack_vulnerability", "GA_per_match", "xGC_per_match"], ascending=False
            ).head(8)
            st.dataframe(
                weak_def[
                    [
                        "crest_url",
                        "team_short",
                        "Elo",
                        "GA_per_match",
                        "xGC_per_match",
                        "last_n_GA_pm",
                        "CS",
                        "att_fdr_vs",
                        "attack_vulnerability",
                    ]
                ].rename(
                    columns={
                        "team_short": "Opponent",
                        "GA_per_match": "GA/m",
                        "xGC_per_match": "xGC/m",
                        "last_n_GA_pm": f"Last {rolling_n} GA/m",
                        "att_fdr_vs": "Att FDR",
                        "attack_vulnerability": "Target Index",
                    }
                ),
                column_config={
                    "crest_url": st.column_config.ImageColumn("Club", width="small"),
                },
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("🛡️ **Top 8 Weakest Attacks (Target for Clean Sheets)**")
            weak_att = club_df.sort_values(
                ["cs_opportunity_vs", "FTS"], ascending=False
            ).head(8)
            st.dataframe(
                weak_att[
                    [
                        "crest_url",
                        "team_short",
                        "Elo",
                        "GF_per_match",
                        "xG_per_match",
                        "last_n_GF_pm",
                        "FTS",
                        "def_fdr_vs",
                        "cs_opportunity_vs",
                    ]
                ].rename(
                    columns={
                        "team_short": "Opponent",
                        "GF_per_match": "GF/m",
                        "xG_per_match": "xG/m",
                        "last_n_GF_pm": f"Last {rolling_n} GF/m",
                        "def_fdr_vs": "Def FDR",
                        "cs_opportunity_vs": "CS Target Index",
                    }
                ),
                column_config={
                    "crest_url": st.column_config.ImageColumn("Club", width="small"),
                },
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Full Club Table & Home/Away Splits")
        club_cols = [
            "crest_url",
            "team_name",
            "team_short",
            "Elo",
            "MP",
            "W",
            "D",
            "L",
            "Pts",
            "GF",
            "GA",
            "GD",
            "CS",
            "FTS",
            "GF_per_match",
            "GA_per_match",
            "xG_per_match",
            "xGC_per_match",
            "att_fdr_vs",
            "def_fdr_vs",
            "GF_home_pm",
            "GA_home_pm",
            "GF_away_pm",
            "GA_away_pm",
            "last_n_form",
            "last_n_GF_pm",
            "last_n_GA_pm",
        ]
        st.dataframe(
            club_df.sort_values(["Pts", "GD", "GF"], ascending=False)[club_cols],
            column_config={
                "crest_url": st.column_config.ImageColumn("Crest", width="small"),
                "team_name": st.column_config.TextColumn("Club"),
                "team_short": st.column_config.TextColumn("Short"),
            },
            use_container_width=True,
            hide_index=True,
            height=450,
        )

# =====================================================================
# TAB 3: GAMEWEEK MATCHUPS & FIXTURE TICKER
# =====================================================================
with tab_gw:
    st.subheader("📅 Multi-Gameweek Fixture Difficulty Ticker (FDR)")
    t_col1, t_col2, t_col3 = st.columns([1, 1, 1.4])
    with t_col1:
        ticker_start = st.number_input("Start Gameweek", min_value=1, max_value=38, value=next_gw)
    with t_col2:
        ticker_span = st.slider("Number of Gameweeks to Display", min_value=3, max_value=10, value=6)
    with t_col3:
        fdr_mode_label = st.selectbox(
            "FDR Rating Mode",
            options=[
                "Overall FDR (Official FPL)",
                "Attacking FDR (Target Weak Defenses)",
                "Defensive FDR (Target Clean Sheets)",
            ],
            index=0,
        )
        fdr_mode_map = {
            "Overall FDR (Official FPL)": "overall",
            "Attacking FDR (Target Weak Defenses)": "attack",
            "Defensive FDR (Target Clean Sheets)": "defense",
        }
        fdr_mode_key = fdr_mode_map.get(fdr_mode_label, "overall")

    text_ticker, fdr_ticker = build_fixture_ticker(
        bootstrap,
        fixtures,
        start_gw=int(ticker_start),
        num_gws=int(ticker_span),
        club_df=club_df,
        fdr_mode=fdr_mode_key,
    )
    if not text_ticker.empty:
        gw_cols = [c for c in text_ticker.columns if c.startswith("GW")]
        z_vals = fdr_ticker[gw_cols].values
        text_vals = text_ticker[gw_cols].values
        y_labels = (
            text_ticker["Club"] + " (Avg: " + text_ticker["Avg FDR"].astype(str) + ")"
        ).tolist()

        fig_fdr = go.Figure(
            data=go.Heatmap(
                z=z_vals,
                x=gw_cols,
                y=y_labels,
                text=text_vals,
                texttemplate="%{text}",
                textfont={"size": 11, "color": "#0f172a"},
                colorscale=[
                    [0.0, "#00ff87"],
                    [0.25, "#4ade80"],
                    [0.5, "#cbd5e1"],
                    [0.75, "#fb923c"],
                    [1.0, "#ef4444"],
                ],
                zmin=1,
                zmax=5,
                colorbar=dict(title="FDR"),
            )
        )
        fig_fdr.update_layout(
            title=f"Upcoming Fixture Ticker (GW{ticker_start} – GW{min(38, int(ticker_start) + int(ticker_span) - 1)}) — {fdr_mode_label} (Sorted Easiest to Hardest)",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=180, r=20, t=50, b=30),
        )
        apply_broadcast_plotly_theme(fig_fdr, height=580)
        st.plotly_chart(fig_fdr, use_container_width=True)

    st.divider()
    st.subheader("⚔️ Single-Gameweek Matchup Broadcast Cards & Projections")
    selected_gw = st.slider("Select Gameweek to Analyze", min_value=1, max_value=38, value=current_gw)

    matchups_df = build_gameweek_matchups(fixtures, club_df, players_df, event_id=int(selected_gw))
    if not matchups_df.empty:
        # Render Sofascore-style Matchup Cards with Club Crests & Poisson Goal Bars
        render_matchup_cards_html(matchups_df)

        with st.expander(f"📊 View Detailed GW {selected_gw} Poisson Table & Key Form Players", expanded=False):
            st.dataframe(
                matchups_df[
                    [
                        "Matchup",
                        "Home Form",
                        "Away Form",
                        "Home xG/m",
                        "Away xGC/m",
                        "Home Proj G",
                        "Away Proj G",
                        "Home CS %",
                        "Away CS %",
                        "Best Attacking Side",
                        "Key Home Players",
                        "Key Away Players",
                    ]
                ],
                column_config={
                    "Home CS %": st.column_config.ProgressColumn(
                        "Home CS %", format="%.1f%%", min_value=0.0, max_value=70.0
                    ),
                    "Away CS %": st.column_config.ProgressColumn(
                        "Away CS %", format="%.1f%%", min_value=0.0, max_value=70.0
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

    st.markdown(f"### 📊 GW {selected_gw} Player Performance vs. Opponent")
    try:
        live_gw_data = get_event_live(int(selected_gw))
        gw_players_df = build_gw_live_player_table(
            players_df, live_gw_data, fixtures, teams_df, event_id=int(selected_gw)
        )
        if not gw_players_df.empty and (gw_players_df["gw_minutes"] > 0).any():
            gw_players_df["squad_status"] = np.where(
                gw_players_df["id"].astype(int).isin(sandbox_active_ids),
                np.where(gw_players_df["id"].astype(int).isin(owned_player_ids), "⭐ Owned", "🆕 Sandbox In"),
                "",
            )
            gw_f1, gw_f2 = st.columns([2, 1])
            with gw_f1:
                gw_club_filter = st.multiselect(
                    f"Filter GW {selected_gw} Players by Club",
                    options=sorted(gw_players_df["team_short"].unique().tolist()),
                    default=[],
                )
            with gw_f2:
                gw_only_owned = st.checkbox(
                    f"⭐ Show Only My Squad in GW {selected_gw}",
                    value=False,
                    disabled=(len(sandbox_active_ids) == 0),
                )
            gw_view = gw_players_df[gw_players_df["gw_minutes"] > 0].copy()
            if gw_club_filter:
                gw_view = gw_view[gw_view["team_short"].isin(gw_club_filter)]
            if gw_only_owned:
                gw_view = gw_view[gw_view["squad_status"] != ""]
            st.dataframe(
                gw_view[
                    [
                        "web_name",
                        "squad_status",
                        "team_short",
                        "position",
                        "opponent",
                        "gw_minutes",
                        "gw_points",
                        "gw_goals",
                        "gw_assists",
                        "gw_xG",
                        "gw_xA",
                        "gw_xGI",
                        "gw_cs",
                        "gw_gc",
                        "gw_xGC",
                        "gw_bps",
                        "gw_bonus",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                height=380,
            )
        else:
            st.info(f"Gameweek {selected_gw} has not kicked off yet or has no recorded player minutes.")
    except Exception as exc:
        st.warning(f"Could not load live player stats for GW {selected_gw}: {exc}")

# =====================================================================
# TAB 4: MANAGER SQUAD, HTML 2D PITCH & LIVE TRANSFER SANDBOX
# =====================================================================
with tab_manager:
    if mgr_load_error:
        st.error(f"Could not load manager data for ID {manager_id_input}: {mgr_load_error}")
    elif squad_df_raw.empty or all_players_xp.empty:
        st.info("Enter a valid FPL Manager ID in the left sidebar to load your 15-man squad.")
    else:
        picks_list = squad_df_raw.copy()
        picks_list.rename(columns={"position": "squad_slot"}, inplace=True)
        merge_cols = [
            "id",
            "photo_url",
            "crest_url",
            "web_name",
            "team",
            "team_short",
            "position",
            "price",
            "price_trend",
            "set_piece_duty",
            "form",
            "total_points",
            "xGI_per_90",
            "pts_per_90",
            "Availability",
            "play_prob",
            "next_opp",
            "next_fdr",
            "xP_next",
            "xP_horizon",
            "horizon_fixtures",
        ]
        avail_merge_cols = [c for c in merge_cols if c in all_players_xp.columns]
        squad = picks_list.merge(
            all_players_xp[avail_merge_cols],
            left_on="element",
            right_on="id",
            how="left",
        )
        squad["Current_Role"] = np.where(
            squad["is_captain"],
            "© Captain",
            np.where(
                squad["is_vice_captain"],
                "Vice Captain",
                np.where(squad["squad_slot"] <= 11, "Starter", f"Bench ({squad['squad_slot'] - 11})"),
            ),
        )

        base_lineup_res = optimize_starting_xi(squad)
        base_opt_xp = float(base_lineup_res.get("opt_xi_xp", 0.0))
        base_curr_xp = float(base_lineup_res.get("current_xi_xp", 0.0))
        base_hor_xp = float(squad["xP_horizon"].sum())

        working_squad = squad.copy()
        working_bank_m = float(bank_m)
        applied_moves_log = []

        xp_lookup = all_players_xp.set_index("id")
        for mv in st.session_state.get("sandbox_transfers", []):
            o_id = int(mv["out_id"])
            i_id = int(mv["in_id"])
            if o_id in working_squad["id"].astype(int).values and i_id in xp_lookup.index:
                mask = working_squad["id"].astype(int) == o_id
                out_row = working_squad.loc[mask].iloc[0]
                in_row = xp_lookup.loc[i_id]
                o_price = float(out_row["price"])
                i_price = float(in_row["price"])
                working_bank_m = round(working_bank_m + o_price - i_price, 1)

                for col in avail_merge_cols:
                    if col != "id" and col in working_squad.columns and col in in_row:
                        working_squad.loc[mask, col] = in_row[col]
                working_squad.loc[mask, "id"] = i_id
                working_squad.loc[mask, "element"] = i_id
                working_squad.loc[mask, "Current_Role"] = f"🆕 In (for {out_row['web_name']})"
                applied_moves_log.append(
                    f"{out_row['web_name']} (£{o_price:.1f}m) ➔ {in_row['web_name']} (£{i_price:.1f}m)"
                )

        num_sb_transfers = len(applied_moves_log)
        remaining_ft = max(0, int(free_transfers_input) - num_sb_transfers)
        sb_hit_cost = max(0, num_sb_transfers - int(free_transfers_input)) * 4.0

        lineup_res = optimize_starting_xi(working_squad)
        opt_sq = lineup_res.get("squad_table", working_squad)
        opt_xp = float(lineup_res.get("opt_xi_xp", 0.0))
        curr_xp = base_curr_xp
        gain_xp = round(opt_xp - curr_xp, 2)
        sb_xi_delta = round(opt_xp - base_opt_xp, 2)
        sb_hor_net_delta = round(float(working_squad["xP_horizon"].sum()) - base_hor_xp - sb_hit_cost, 2)

        single_moves_df, double_moves_df, watchlist_df = recommend_transfers(
            squad_df=opt_sq,
            all_players_xp=all_players_xp,
            bank_m=working_bank_m,
            free_transfers=int(remaining_ft),
            horizon=int(plan_horizon),
        )

        # Find Captain & Vice-Captain rows for Spotlight Cards
        cap_name = lineup_res.get("captain_name", "-")
        vc_name = lineup_res.get("vc_name", "-")
        cap_rows = opt_sq[opt_sq["web_name"] == cap_name]
        cap_row = cap_rows.iloc[0] if not cap_rows.empty else opt_sq.iloc[0]
        mvp_hor_row = opt_sq.sort_values("xP_horizon", ascending=False).iloc[0]

        render_spotlight_cards(
            [
                {
                    "title": f"© GW {next_gw} Captain Pick",
                    "name": cap_row["web_name"],
                    "stat": f"{float(cap_row.get('Effective_xP', cap_row['xP_next'] * 2)):.1f} xP",
                    "sub": f"VC: {vc_name} • vs {cap_row['next_opp']}",
                    "team_short": cap_row["team_short"],
                    "photo_url": cap_row.get("photo_url", ""),
                    "crest_url": cap_row.get("crest_url", ""),
                },
                {
                    "title": f"🚀 {plan_horizon}-GW Horizon MVP",
                    "name": mvp_hor_row["web_name"],
                    "stat": f"{mvp_hor_row['xP_horizon']:.1f} xP",
                    "sub": f"{mvp_hor_row['team_short']} • £{mvp_hor_row['price']:.1f}m",
                    "team_short": mvp_hor_row["team_short"],
                    "photo_url": mvp_hor_row.get("photo_url", ""),
                    "crest_url": mvp_hor_row.get("crest_url", ""),
                },
                {
                    "title": f"⚡ GW {next_gw} Optimal XI",
                    "name": f"Formation {lineup_res.get('formation', '3-4-3')}",
                    "stat": f"{opt_xp:.1f} pts",
                    "sub": (
                        f"{sb_xi_delta:+.1f} vs Base XI"
                        if num_sb_transfers > 0
                        else f"{gain_xp:+.1f} vs Current XI"
                    ),
                    "team_short": cap_row["team_short"],
                    "photo_url": cap_row.get("photo_url", ""),
                    "crest_url": "",
                },
                {
                    "title": "💰 Squad Value & Bank",
                    "name": mgr_info.get("name", "My Squad"),
                    "stat": f"£{working_bank_m:.1f}m ITB",
                    "sub": f"Squad £{squad_val_m:.1f}m • {remaining_ft} FT Left",
                    "team_short": mvp_hor_row["team_short"],
                    "photo_url": mvp_hor_row.get("photo_url", ""),
                    "crest_url": "",
                },
            ]
        )

        if num_sb_transfers > 0:
            st.info(
                f"🧪 **Sandbox Mode Active ({num_sb_transfers} Transfer{'s' if num_sb_transfers > 1 else ''} Applied)**: "
                + " | ".join(applied_moves_log)
                + f"  •  **Hit Cost**: `-{int(sb_hit_cost)} pts`  •  **GW {next_gw} XI Delta**: `{sb_xi_delta:+.2f} xP`  •  **Net {plan_horizon}-GW Delta (after hit)**: `{sb_hor_net_delta:+.2f} xP`"
            )

        # --- SIDE-BY-SIDE WORKSPACE: HTML 2D PITCH (LEFT) + LIVE SANDBOX (RIGHT) ---
        ws_left, ws_right = st.columns([1.3, 1.0])

        with ws_left:
            view_mode = st.radio(
                "Squad View Mode",
                options=["🏟️ Broadcast 2D Pitch (Player Photos)", "📊 Expected Points (xP) Bar Chart"],
                horizontal=True,
                label_visibility="collapsed",
            )
            if "Broadcast 2D Pitch" in view_mode:
                render_html_pitch(opt_sq, lineup_res.get("formation", "3-4-3"), int(next_gw))
            else:
                fig_sq = px.bar(
                    opt_sq.sort_values("xP_next", ascending=True),
                    x="xP_next",
                    y="web_name",
                    color="Recommended_Role",
                    orientation="h",
                    hover_data=["team_short", "next_opp", "xP_horizon", "Availability"],
                    title=f"Squad GW {next_gw} Base Expected Points (xP) Ranking",
                    color_discrete_sequence=["#00ff87", "#04f5ff", "#f59e0b", "#64748b"],
                )
                fig_sq.update_layout(
                    xaxis_title=f"GW {next_gw} Base Expected Points (xP)",
                    yaxis_title="",
                    legend=dict(orientation="h", y=1.12),
                )
                apply_broadcast_plotly_theme(fig_sq, height=560)
                st.plotly_chart(fig_sq, use_container_width=True)

        with ws_right:
            st.markdown(
                f"#### 🧪 Live Transfer Sandbox (`£{working_bank_m:.1f}m ITB` • `{remaining_ft} FT Left`)"
            )
            st.caption(
                "Apply a recommended upgrade or test a custom swap—your **Broadcast Pitch**, **Player Photos**, and **Optimal XI** on the left update immediately."
            )

            with st.container(border=True):
                st.markdown("**⚡ Quick-Apply Top Recommended Move**")
                if not single_moves_df.empty and "_out_id" in single_moves_df.columns:
                    rec_options = {
                        f"{row['Transfer Out']} ➔ {row['Transfer In']} (XI: {row['XI Next GW Gain']:+.2f}, {plan_horizon}GW: {row[f'Net {plan_horizon}-GW xP Gain']:+.2f})": (
                            int(row["_out_id"]),
                            int(row["_in_id"]),
                        )
                        for _, row in single_moves_df.iterrows()
                    }
                    sel_rec_label = st.selectbox(
                        "Best 1-FT Upgrades",
                        options=list(rec_options.keys()),
                        key="sb_rec_select",
                        label_visibility="collapsed",
                    )
                    if st.button("➕ Apply Recommended Transfer", use_container_width=True, type="primary"):
                        o_id, i_id = rec_options[sel_rec_label]
                        st.session_state["sandbox_transfers"].append({"out_id": o_id, "in_id": i_id})
                        st.rerun()
                else:
                    st.caption("No single recommended move exceeds +0.5 xP.")

            with st.container(border=True):
                st.markdown("**🛠️ Custom Player Swap (Budget & Club-Quota Checked)**")
                sq_sorted = opt_sq.sort_values(["position", "price"], ascending=[True, False])
                out_map = {
                    f"{r['web_name']} ({r['team_short']} - {r['position']} - £{r['price']:.1f}m | GW{next_gw}: {r['xP_next']:.1f} xP)": int(r["id"])
                    for _, r in sq_sorted.iterrows()
                }
                sel_out_lbl = st.selectbox(
                    "1. Player Out",
                    options=list(out_map.keys()),
                    key="sb_custom_out",
                )
                sel_out_id = out_map[sel_out_lbl]
                out_p_row = opt_sq[opt_sq["id"].astype(int) == sel_out_id].iloc[0]
                out_pos = out_p_row["position"]
                out_team = int(out_p_row["team"])
                max_swap_budget = round(float(out_p_row["price"]) + working_bank_m, 1)

                curr_club_counts = opt_sq["team"].astype(int).value_counts().to_dict()
                curr_club_counts[out_team] = max(0, curr_club_counts.get(out_team, 1) - 1)
                current_ids = set(opt_sq["id"].astype(int).tolist())

                valid_in_pool = all_players_xp[
                    (~all_players_xp["id"].astype(int).isin(current_ids))
                    & (all_players_xp["position"] == out_pos)
                    & (all_players_xp["price"] <= max_swap_budget)
                    & (all_players_xp["team"].astype(int).map(lambda t: curr_club_counts.get(int(t), 0) < 3))
                ].sort_values(["xP_horizon", "xP_next"], ascending=False)

                if not valid_in_pool.empty:
                    in_map = {
                        f"{r['web_name']} ({r['team_short']} £{r['price']:.1f}m | GW{next_gw}: {r['xP_next']:.1f} xP | {plan_horizon}GW: {r['xP_horizon']:.1f} xP)": int(r["id"])
                        for _, r in valid_in_pool.head(60).iterrows()
                    }
                    sel_in_lbl = st.selectbox(
                        f"2. {out_pos} Replacement In (Max £{max_swap_budget:.1f}m)",
                        options=list(in_map.keys()),
                        key="sb_custom_in",
                    )
                    if st.button("➕ Apply Custom Transfer", use_container_width=True):
                        st.session_state["sandbox_transfers"].append(
                            {"out_id": sel_out_id, "in_id": in_map[sel_in_lbl]}
                        )
                        st.rerun()
                else:
                    st.warning(f"No valid {out_pos} within £{max_swap_budget:.1f}m budget.")

            with st.container(border=True):
                club_quota_series = opt_sq["team_short"].value_counts()
                quota_badges = ", ".join(
                    [f"`{clb}: {cnt}/3`" for clb, cnt in club_quota_series.items() if cnt >= 2]
                )
                st.caption(f"**Club Quotas (≥2 owned)**: {quota_badges if quota_badges else 'None ≥2'}")
                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    if st.button(
                        "↩️ Undo Last",
                        disabled=(num_sb_transfers == 0),
                        use_container_width=True,
                    ):
                        st.session_state["sandbox_transfers"].pop()
                        st.rerun()
                with btn_c2:
                    if st.button(
                        "🔄 Reset Squad",
                        disabled=(num_sb_transfers == 0),
                        use_container_width=True,
                    ):
                        st.session_state["sandbox_transfers"] = []
                        st.rerun()

        st.divider()

        m_sub1, m_sub2, m_sub3, m_sub4, m_sub5 = st.tabs(
            [
                f"📋 Optimal 15-Man Squad Table ({lineup_res.get('formation', '3-4-3')})",
                "🎯 1-FT & 2-FT Transfer Packages",
                "🔭 Top Targets Watchlist",
                "🃏 Chip Strategy Advisor",
                "🏅 Mini-League Standings",
            ]
        )

        with m_sub1:
            sq_disp_cols = [
                c
                for c in [
                    "photo_url",
                    "crest_url",
                    "Recommended_Role",
                    "Current_Role",
                    "web_name",
                    "team_short",
                    "position",
                    "price",
                    "price_trend",
                    "set_piece_duty",
                    "Availability",
                    "next_opp",
                    "xP_next",
                    "Effective_xP",
                    "xP_horizon",
                    "form",
                    "xGI_per_90",
                ]
                if c in opt_sq.columns
            ]
            st.dataframe(
                opt_sq[sq_disp_cols].rename(
                    columns={
                        "Recommended_Role": f"GW {next_gw} Rec. Role",
                        "Current_Role": f"GW {current_gw} Role",
                        "web_name": "Player",
                        "team_short": "Club",
                        "position": "Pos",
                        "price": "£m",
                        "price_trend": "Price Trend",
                        "set_piece_duty": "Set Pieces",
                        "next_opp": f"GW {next_gw} Opp",
                        "xP_next": f"GW {next_gw} xP",
                        "Effective_xP": "Eff. xP (w/ Cap)",
                        "xP_horizon": f"Next {plan_horizon}-GW xP",
                    }
                ),
                column_config={
                    "photo_url": st.column_config.ImageColumn("Photo", width="small"),
                    "crest_url": st.column_config.ImageColumn("Crest", width="small"),
                    "£m": st.column_config.NumberColumn("Price", format="£%.1fm"),
                    f"GW {next_gw} xP": st.column_config.ProgressColumn(
                        f"GW {next_gw} xP", format="%.2f", min_value=0.0, max_value=12.0
                    ),
                    "Eff. xP (w/ Cap)": st.column_config.NumberColumn(
                        "Eff. xP (w/ Cap)", format="%.2f"
                    ),
                    f"Next {plan_horizon}-GW xP": st.column_config.ProgressColumn(
                        f"Next {plan_horizon}-GW xP",
                        format="%.1f",
                        min_value=0.0,
                        max_value=float(max(20.0, opt_sq["xP_horizon"].max() * 1.1)),
                    ),
                },
                use_container_width=True,
                hide_index=True,
                height=540,
            )

        with m_sub2:
            st.markdown("##### 🎯 Best Single Transfers (1 FT)")
            if not single_moves_df.empty:
                visible_single_cols = [c for c in single_moves_df.columns if not c.startswith("_")]
                st.dataframe(
                    single_moves_df[visible_single_cols],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success("Your squad is already well-optimized! No single transfer gains > 0.5 xP.")

            st.markdown("##### ⚡ Best 2-Player Packages (2 Transfers)")
            if not double_moves_df.empty:
                st.dataframe(double_moves_df, use_container_width=True, hide_index=True)
            else:
                st.info("No 2-transfer package exceeds +1.0 Net xP Gain after accounting for transfer cost.")

        with m_sub3:
            if not watchlist_df.empty:
                pos_watch = st.multiselect(
                    "Filter Watchlist Position",
                    options=["GKP", "DEF", "MID", "FWD"],
                    default=["GKP", "DEF", "MID", "FWD"],
                )
                st.dataframe(
                    watchlist_df[watchlist_df["Position"].isin(pos_watch)],
                    use_container_width=True,
                    hide_index=True,
                )

        with m_sub4:
            chip_df = evaluate_chip_strategy(mgr_hist, lineup_res, next_gw=int(next_gw))
            st.dataframe(chip_df, use_container_width=True, hide_index=True)

        with m_sub5:
            if league_id_input > 0:
                try:
                    standings_data = get_classic_league_standings(int(league_id_input))
                    league_name = standings_data.get("league", {}).get("name", f"League {league_id_input}")
                    results_df = pd.DataFrame(standings_data.get("standings", {}).get("results", []))
                    st.markdown(f"##### 🏅 {league_name}")
                    if not results_df.empty:
                        st.dataframe(
                            results_df[["rank", "entry_name", "player_name", "event_total", "total", "entry"]],
                            use_container_width=True,
                            hide_index=True,
                        )
                except Exception as exc:
                    st.error(f"Could not load mini-league ID {league_id_input}: {exc}")
            else:
                st.info("Enter your Classic Mini-League ID in the left sidebar to view live standings.")
