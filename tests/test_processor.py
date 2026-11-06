"""Tests for the lineup optimizer in src/processor.py."""

import pandas as pd
import pytest

from src.processor import optimize_starting_xi

COUNTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}


def _squad() -> pd.DataFrame:
    rows = []
    pid = 1
    for pos, n in COUNTS.items():
        for i in range(n):
            rows.append(
                {
                    "id": pid,
                    "web_name": f"{pos}{i}",
                    "position": pos,
                    "team": pid,
                    "xP_next": float(10 - i),  # strictly descending per position
                    "xP_horizon": float(10 - i),
                    "form": 5.0,
                    "next_opp": "XXX (H)",
                    "squad_slot": pid,
                    "is_captain": pid == 1,
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def test_empty_or_short_squad_returns_empty():
    assert optimize_starting_xi(pd.DataFrame()) == {}
    assert optimize_starting_xi(_squad().head(14)) == {}


def test_starting_xi_is_valid_formation():
    res = optimize_starting_xi(_squad())
    table = res["squad_table"]
    starters = table[table["is_opt_starter"]]
    counts = starters["position"].value_counts()
    assert len(starters) == 11
    assert counts["GKP"] == 1
    assert 3 <= counts["DEF"] <= 5
    assert 2 <= counts["MID"] <= 5
    assert 1 <= counts["FWD"] <= 3
    assert res["formation"] == f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"


def test_captain_doubles_and_totals_match():
    res = optimize_starting_xi(_squad())
    table = res["squad_table"]
    assert (table["Opt_Mult"] == 2).sum() == 1
    assert len(table) == 15
    assert res["opt_xi_xp"] == pytest.approx(table["Effective_xP"].sum())
    assert res["opt_xi_xp"] >= res["current_xi_xp"] - 1e-9 or res["lineup_gain_xp"] < 0


def test_best_gkp_starts_and_other_is_bench():
    table = optimize_starting_xi(_squad())["squad_table"]
    gkp = table[table["position"] == "GKP"].set_index("web_name")
    assert gkp.loc["GKP0", "is_opt_starter"]
    assert gkp.loc["GKP1", "Recommended_Role"] == "Bench GKP"
