# 📈 FPL Analytics Dashboard — Improvement Plan & Implementation Status

This document tracks the roadmap and implementation status for the FPL Analytics Dashboard. It covers **Core Bug & Performance Fixes**, **Data & Analytical Improvements**, **UI & UX Improvements**, and **External Data Sources**.

---

## ✅ Completed Core Bug & Performance Fixes (Step 1)

* **✅ Removed Streamlit Cache-Busting Reloads**: Removed `importlib.reload(src.api)` and `importlib.reload(src.processor)` in `app.py` so `@st.cache_data` persists properly across user interactions.
* **✅ Minutes-Normalized Club `xGC`**: Replaced raw player `xGC` summation in `build_club_and_opponent_stats()` with Goalkeeper `xGC` normalized by Goalkeeper minutes played (`raw_gkp_xgc / (raw_gkp_mins / 90.0)`), eliminating inflation from substitutes.
* **✅ Empirical Bayes Shrinkage & Position-Specific FPL Scoring**: Upgraded `compute_player_xp()` to shrink per-90 rates toward positional means using a 270-minute (`3.0` nineties) prior (`n90 / (n90 + 3.0)`), preventing 10-minute cameo outliers. Replaced uniform goal weights with exact FPL position rules (`GKP/DEF: 6 pts`, `MID: 5 pts`, `FWD: 4 pts`, `Clean Sheet: 4 pts` for GKP/DEF and `1 pt` for MID, expected goalkeeper save points, and `-1 pt` per 2 expected goals conceded).
* **✅ Vectorized `compute_player_xp()`**: Replaced row-by-row `.iterrows()` over ~800 players with vectorized NumPy array operations.
* **✅ Dual Half-Season Wildcard Tracking**: Updated `evaluate_chip_strategy()` to track **Wildcard 1 (GW1–19)** and **Wildcard 2 (GW20–38)** separately.
* **✅ True Starting XI Gain in Transfer Recommender**: Updated `recommend_transfers()` to re-run `optimize_starting_xi()` on the post-transfer 15-man squad and report `XI Next GW Gain` alongside `Selling Price` support.

---

## Part 1: Data & Analytical Improvements

### 1.1 Poisson Goal & Clean Sheet Probability Model — ✅ IMPLEMENTED
* **Previous State**: Used a linear average of team $xG$ and opponent $xGC$, paired with FPL's single `1–5` Fixture Difficulty Rating (FDR).
* **Implemented Upgrade** (`src/processor.py` & `app.py`):
  * **Multiplicative Poisson Expected Goals ($\lambda$)**:
    $$\lambda_{\text{Home}} = \text{Attack}_{\text{Home}} \times \text{Defense}_{\text{Away}} \times \overline{\text{GF}}_{\text{League, Home}} \times \text{EloFactor}$$
    $$\lambda_{\text{Away}} = \text{Attack}_{\text{Away}} \times \text{Defense}_{\text{Home}} \times \overline{\text{GF}}_{\text{League, Away}} \times \text{EloFactor}$$
  * **Exact Clean Sheet Probability**:
    $$P(\text{CS}) = e^{-\lambda_{\text{Opponent}}}$$
    Exposed as `Home CS %` and `Away CS %` in Tab 3 (`build_gameweek_matchups`) and integrated directly into GKP/DEF/MID clean-sheet expected points in `compute_player_xp()`.
  * **Split Attacking vs. Defensive FDR**: Added `att_fdr_vs` and `def_fdr_vs` in `build_club_and_opponent_stats()` and an interactive **FDR Rating Mode** selector (`Overall FDR`, `Attacking FDR`, `Defensive FDR`) in Tab 3 (`build_fixture_ticker`).

### 1.2 Rolling Player Form (`Last 4 GWs`) & Local Storage — ⏳ PLANNED
* **Current State**: Computes per-90 metrics (`xG_per_90`, `xA_per_90`, `xGI_per_90`) from full-season totals in `bootstrap-static/` (with Empirical Bayes shrinkage and FPL's 30-day `form` metric).
* **Target Upgrade**:
  * Add a local **DuckDB** or **Parquet** cache (`data/player_history.parquet`) that stores `element-summary/{id}/` match logs across all active players.
  * Compute **Rolling 4-GW Per-90 Metrics** (`last4_xGI_per_90`, `last4_mins_avg`, `last4_bps_per_90`) and **Home vs. Away Player Splits**.
  * Detect **Role Shifts & Minutes Trends** (e.g., players moving from bench cameos to 90-minute starters).

### 1.3 Set-Piece & Penalty Taker Weighting — ✅ IMPLEMENTED
* **Previous State**: Treated all expected goal involvements ($xGI$) equally.
* **Implemented Upgrade** (`src/processor.py` & `app.py`):
  * Extracted `penalties_order`, `corners_and_indirect_freekicks_order`, and `direct_freekicks_order` in `build_players_df()`.
  * Applied `sp_multiplier` (`+8%` PEN #1, `+3%` PEN #2, `+4%` CRN #1, `+1.5%` CRN #2, `+2%` FK #1) to attacking expected points in `compute_player_xp()`.
  * Added `set_piece_duty` (`PEN #1, CRN #1, FK #1`) to the Master Player Table (Tab 1), Optimal Lineup Builder (Tab 4), and Transfer Watchlist (Tab 4).

### 1.4 Exact Selling Price & Price Change Velocity Predictor — ✅ IMPLEMENTED
* **Implemented (`src/processor.py` & `app.py`)**:
  * **Selling Price Support**: `recommend_transfers()` supports a `selling_price` column (defaulting to current `price` when unauthenticated public endpoints are used).
  * **Price Change Velocity & Trend Indicator**: Computes `transfer_velocity` (net gameweek transfers normalized by effective ownership floor) and assigns `price_trend` badges (`📈 Likely Rise (+£0.1m)`, `↗️ Trending Up`, `📉 Likely Fall (-£0.1m)`, `↘️ Trending Down`, `➖ Stable`) in `build_players_df()`, exposed in the Master Player Table and Player Radar Comparison.

### 1.5 Multi-Gameweek Linear Programming (MILP) Transfer Planner — ⏳ PLANNED
* **Current State**: Evaluates 1-FT and 2-FT moves across a user-selected horizon (`3–8` GWs) with Starting XI re-optimization.
* **Target Upgrade**:
  * Use `scipy.optimize.milp` to solve a **Multi-Week Rollover Plan** across `GW+1` to `GW+5`.
  * Models **Free Transfer banking** (up to 5 stored FTs), exact budget constraints, club limits (`max 3 per club`), and when to roll a transfer versus taking a `-4 pt` hit.

---

## Part 2: UI & User Experience (UX) Improvements

### 2.1 Visual 2D Football Pitch for Lineup Builder — ✅ IMPLEMENTED
* **Previous State**: Displayed the recommended Starting XI and Bench only in a tabular dataframe and bar chart.
* **Implemented Upgrade** (`render_pitch_view` in `app.py`):
  * Renders an interactive **2D Football Pitch** in Tab 4 (`Manager & Lineup Builder`) with a toggle between `⚽ 2D Starting XI Pitch` and `📊 Expected Points (xP) Bar Chart`.
  * Arranges the Starting XI visually in their recommended formation (`GKP`, `DEF`, `MID`, `FWD`) with penalty boxes, halfway line, role color-coding (**Captain `©`**, **Vice-Captain `(VC)`**, **Starter**), opponent tags, and projected effective $xP$.
  * Displays the 4-man Bench (`Bench 1 (GK)`, `Bench 1–3`) in a shaded dugout zone underneath the pitch.

### 2.2 Interactive "What-If" Squad & Transfer Sandbox — ✅ IMPLEMENTED
* **Previous State**: Displayed recommended transfers and Starting XI gain in a read-only table.
* **Implemented Upgrade** (`app.py` & `src/processor.py`):
  * Added a live **`st.session_state["sandbox_transfers"]` Transfer Sandbox** in Tab 4 (`Manager & Lineup Builder`).
  * Supports **Quick-Apply Recommended Move** (one-click application of top 1-FT recommendations) and **Custom Player Swap** (position-matched dropdowns pre-filtered by live budget and the max 3-per-club rule).
  * Automatically recalculates **Sandbox Bank (`£m`)**, **Point Hit (`-4 pts` beyond free transfers)**, **Club Quotas (`x/3`)**, **Optimal Starting XI Table**, **2D Starting XI Pitch View**, and subsequent **Transfer Recommendations**, with one-click **Undo** and **Reset to Live Squad** controls.

### 2.3 Rich Table Formatting (`st.column_config` & Player Photos) — ✅ IMPLEMENTED
* **Previous State**: Standard text and numeric columns in `st.dataframe`.
* **Implemented Upgrade** (`src/processor.py` & `app.py`):
  * Added `photo_url` in `build_players_df()` and `st.column_config.ImageColumn` in the Master Player Table to display **Player Headshots** (`https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png`).
  * Added `st.column_config.ProgressColumn` for `TSB %` (`selected_by_percent`), `xGI / 90`, `Home CS %`, `Away CS %`, `GW xP`, and `Horizon xP`, plus `£%.1fm` currency formatting.

### 2.4 Head-to-Head Player Percentile Radar Comparison — ✅ IMPLEMENTED
* **Previous State**: Compared players only on a 2-axis scatter plot.
* **Implemented Upgrade** (`build_player_radar_df` in `src/processor.py` & `app.py`):
  * Added a **🕸️ Head-to-Head Player Percentile Radar Comparison** section in Tab 1 (`Player Analytics`) supporting `2–4` players simultaneously.
  * Plots overlapping **Percentile Radar (`Scatterpolar`) Polygons (`0–100%`)** benchmarked against all active Premier League players (`minutes >= 180`) across 6 dimensions: `Goal Threat (xG/90)`, `Creativity (xA/90)`, `Bonus Magnet (BPS/90)`, `Recent Form`, `Value (Pts / £m)`, and `Fixture Ease (Next 5 GW)`.
  * Includes a side-by-side raw metric, set-piece duty, and `price_trend` comparison table alongside the radar chart.

### 2.5 Mini-League Rival Spy & Differential Matrix — ⏳ PLANNED
* **Current State**: Displays basic mini-league standings (rank, GW points, total points).
* **Target Upgrade**:
  * Fetch `entry/{rival_id}/event/{gw}/picks/` for the **Top 5–10 Managers** in your mini-league to build a **Mini-League Effective Ownership (EO%) Matrix** (**Shields** vs. **Swords**).

---

## Part 3: External Data Sources & Enrichment Architecture

### 3.1 Understat (`npxG`, Shot Locations, & `xGChain`) — ⏳ PLANNED
* **Unique Metrics**: Non-Penalty $xG$ (`npxG` & `npxG/90`), Big Chances, Shot Coordinates $(X, Y)$, `xGChain`, and `xGBuildup`.

### 3.2 FBref / Opta Advanced Logs (Box Touches & Defensive Contributions) — ⏳ PLANNED
* **Unique Metrics**: Touches in the Penalty Area (`Touches_Att_Pen`), Shots inside the Box (`SiB`), Progressive Carries/Passes (`PrgC`/`PrgP`), and `CBIT` defensive actions.

### 3.3 ClubElo API (Continuous Dynamic Club Strength) — ✅ IMPLEMENTED
* **Implemented (`get_club_elo()` in `src/api.py` & `src/processor.py`)**:
  * Fetches live continuous **Elo Ratings** from `http://api.clubelo.com/YYYY-MM-DD` (cached for 1 hour with automatic fallback to official FPL team strength if offline).
  * Integrated into `build_club_and_opponent_stats()`, `build_gameweek_matchups()`, and `compute_player_xp()` to adjust Poisson goal expectancy ($\lambda$) based on Elo differentials ($\Delta \text{Elo}$).

### 3.4 Football-Data.co.uk & The Odds API (Bookmaker Implied Probabilities) — ⏳ PLANNED
* **Unique Metrics**: Market-Implied Clean Sheet Probability (`Implied_CS_%`), Team Total Goals (`Implied_Team_xG`), and Anytime Goalscorer (`ATGS_%`) odds.

### 3.5 Vaastav FPL Repository (Player ID Crosswalk & Multi-Season Ground Truth) — ⏳ PLANNED
* **Unique Metrics**: Master ID Dictionary (`id_dict.csv`) mapping `FPL_ID` $\leftrightarrow$ `Understat_ID` $\leftrightarrow$ `FBref_ID`.

### 3.6 Summary Comparison Matrix

| Analytical Question | FPL API Alone | Enriched External Source | Status |
| :--- | :--- | :--- | :---: |
| Is a player's $xG$ inflated by penalties? | ❌ No (`xG` includes pens) | **Understat** (`npxG` vs. `xG`) | ⏳ Planned (Set-Piece orders ✅ Live) |
| Who gets the most touches inside the opponent's 18-yard box? | ❌ No | **FBref** (`Touches_Att_Pen`) | ⏳ Planned |
| What is the market-implied Clean Sheet probability this GW? | ❌ Estimated from past $xGC$ | **Football-Data / Odds API** | ⏳ Planned (Poisson CS % ✅ Live) |
| How strong is an opponent on a continuous scale vs. 1–5 FDR? | ❌ Static 1–5 integer | **ClubElo API** (Continuous Elo) | ✅ **Implemented** |
| How to join FPL players to Understat/FBref reliably? | ❌ Name mismatches | **Vaastav `id_dict.csv`** | ⏳ Planned |

---

## Part 4: Implementation Priority Matrix & Current Status

| Priority | Feature | Category | Complexity | Status |
| :---: | :--- | :---: | :---: | :---: |
| **Core** | **Cache Persistence, GK-Normalized `xGC`, Bayes Shrinkage, Vectorized `xP`, 2 Wildcards, XI Transfer Gain** | Core Engine | Medium | ✅ **Implemented** |
| **P1** | **Poisson Clean Sheet & Split Attack/Defense FDR** | Data + UI | Low | ✅ **Implemented** |
| **P1** | **Set-Piece (`PEN`/`CRN`/`FK`) Indicators & Weighting** | Data + UI | Low | ✅ **Implemented** |
| **P1** | **ClubElo Dynamic Opponent Strength Integration** | External Data | Low | ✅ **Implemented** |
| **P1** | **Rich Tables (`st.column_config`) + Player Photos** | UI | Low | ✅ **Implemented** |
| **P1** | **Price Change Velocity & Rise/Fall Predictor** | Data + UI | Low | ✅ **Implemented** |
| **P2** | **Visual 2D Pitch View for Lineup Builder** | UI | Medium | ✅ **Implemented** |
| **P2** | **Player Percentile Radar Comparison** | Data + UI | Low | ✅ **Implemented** |
| **P2** | **Interactive "What-If" Transfer Sandbox** | UI + Data | Medium | ✅ **Implemented** |
| **P2** | **Understat `npxG` + Vaastav ID Crosswalk** | External Data | Medium | ⏳ Planned |
| **P3** | **Mini-League Rival Spy & EO% Matrix** | Data + UI | Medium | ⏳ Planned |
| **P3** | **DuckDB Cache + Rolling 4-GW Player Stats** | Data | Medium | ⏳ Planned |
| **P4** | **Multi-GW Linear Programming (MILP) Solver** | Data | High | ⏳ Planned |
