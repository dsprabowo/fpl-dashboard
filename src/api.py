"""FPL Public API client with Streamlit caching."""

from datetime import date
from io import StringIO
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
import streamlit as st

BASE_URL = "https://fantasy.premierleague.com/api"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 15


def _get_json(endpoint: str) -> Dict[str, Any]:
    """Fetch JSON from an FPL API endpoint."""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=900, show_spinner=False)
def get_bootstrap_static() -> Dict[str, Any]:
    """Fetch master FPL dataset (players, teams, events, element_types)."""
    return _get_json("bootstrap-static/")


@st.cache_data(ttl=900, show_spinner=False)
def get_fixtures() -> List[Dict[str, Any]]:
    """Fetch all season fixtures."""
    url = f"{BASE_URL}/fixtures/"
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=900, show_spinner=False)
def get_player_summary(element_id: int) -> Dict[str, Any]:
    """Fetch match-by-match history and upcoming fixtures for a single player."""
    return _get_json(f"element-summary/{element_id}/")


@st.cache_data(ttl=600, show_spinner=False)
def get_event_live(event_id: int) -> Dict[str, Any]:
    """Fetch live or historical player stats for a specific gameweek."""
    return _get_json(f"event/{event_id}/live/")


@st.cache_data(ttl=600, show_spinner=False)
def get_manager_info(manager_id: int) -> Dict[str, Any]:
    """Fetch basic profile and leagues for an FPL manager."""
    return _get_json(f"entry/{manager_id}/")


@st.cache_data(ttl=600, show_spinner=False)
def get_manager_history(manager_id: int) -> Dict[str, Any]:
    """Fetch gameweek history and chips used for an FPL manager."""
    return _get_json(f"entry/{manager_id}/history/")


@st.cache_data(ttl=600, show_spinner=False)
def get_manager_picks(manager_id: int, event_id: int) -> Dict[str, Any]:
    """Fetch a manager's 15-man squad and starting picks for a specific gameweek."""
    return _get_json(f"entry/{manager_id}/event/{event_id}/picks/")


@st.cache_data(ttl=600, show_spinner=False)
def get_classic_league_standings(league_id: int, page: int = 1) -> Dict[str, Any]:
    """Fetch standings for a classic mini-league."""
    return _get_json(f"leagues-classic/{league_id}/standings/?page_standings={page}")


def get_current_or_last_event(bootstrap: Dict[str, Any]) -> int:
    """Return the active gameweek ID, or the most recently finished gameweek."""
    events = bootstrap.get("events", [])
    for ev in events:
        if ev.get("is_current"):
            return int(ev["id"])
    for ev in reversed(events):
        if ev.get("finished"):
            return int(ev["id"])
    for ev in events:
        if ev.get("is_next"):
            return max(1, int(ev["id"]) - 1)
    return 1


CLUBELO_TO_FPL_SHORT: Dict[str, str] = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton": "BHA",
    "Burnley": "BUR",
    "Chelsea": "CHE",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Ipswich": "IPS",
    "Leeds": "LEE",
    "Leicester": "LEI",
    "Liverpool": "LIV",
    "Luton": "LUT",
    "Man City": "MCI",
    "Man United": "MUN",
    "Newcastle": "NEW",
    "Forest": "NFO",
    "Sheffield United": "SHU",
    "Southampton": "SOU",
    "Sunderland": "SUN",
    "Tottenham": "TOT",
    "West Ham": "WHU",
    "Wolves": "WOL",
}


@st.cache_data(ttl=3600, show_spinner=False)
def get_club_elo() -> Dict[str, float]:
    """Fetch live English club Elo ratings from api.clubelo.com, mapped to FPL short_name."""
    try:
        today_str = date.today().isoformat()
        url = f"http://api.clubelo.com/{today_str}"
        resp = requests.get(url, headers=HEADERS, timeout=8)
        resp.raise_for_status()
        elo_df = pd.read_csv(StringIO(resp.text))
        if "Country" in elo_df.columns:
            elo_df = elo_df[elo_df["Country"] == "ENG"]
        elo_map: Dict[str, float] = {}
        for _, row in elo_df.iterrows():
            club_name = str(row.get("Club", "")).strip()
            elo_val = float(row.get("Elo", 0.0))
            short = CLUBELO_TO_FPL_SHORT.get(club_name)
            if short and elo_val > 0:
                elo_map[short] = round(elo_val, 1)
        return elo_map
    except Exception:
        return {}

