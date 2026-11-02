# ⚽ FPL Comprehensive Analytics Dashboard

An interactive Fantasy Premier League (FPL) analytics and decision-support dashboard built with **Python**, **Streamlit**, **Pandas**, and **Plotly**. It pulls live data directly from the official FPL public API.

---

## 📂 Project Structure

```text
fpl-dashboard/
├── app.py               # Main Streamlit dashboard UI (4 analytical tabs)
├── requirements.txt     # Python dependencies (streamlit, pandas, plotly, requests)
├── README.md            # Project documentation & metric formulas
└── src/
    ├── __init__.py      # Package initializer
    ├── api.py           # Cached FPL API client (master data, fixtures, live GW, manager)
    └── processor.py     # Per-90 metrics, club vulnerability, xP engine, lineup & transfer solver
```

---

## 🚀 Quick Start (Standard Python `venv`)

1. Clone the repository and navigate to the project directory:
   ```powershell
   git clone https://github.com/dsprabowo/fpl-dashboard.git
   cd fpl-dashboard
   ```
2. Create and activate the virtual environment (if not already created):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Launch the dashboard:
   ```powershell
   streamlit run app.py
   ```

---

## 🔍 How to Find Your FPL Manager ID & Mini-League ID

### 1. Finding Your FPL Manager ID
1. Log in to [fantasy.premierleague.com](https://fantasy.premierleague.com) in your web browser.
2. Click the **Points** tab (or click **View Gameweek history** on the right side of the **Pick Team** page).
3. Check the URL in your browser's address bar. Your **FPL Manager ID** is the number directly after `/entry/`:
   ```text
   https://fantasy.premierleague.com/entry/1234567/event/1
                                           ^^^^^^^
                                      Your Manager ID
   ```
4. Enter that number into the **FPL Manager ID** field in the left sidebar of the dashboard.

### 2. Finding Your Classic Mini-League ID (Optional)
1. Go to **Leagues & Cups** on the official FPL website and click on any Classic Mini-League.
2. Check the URL in your browser's address bar. Your **Mini-League ID** is the number after `/leagues/`:
   ```text
   https://fantasy.premierleague.com/leagues/987654/standings/c
                                             ^^^^^^
                                       Your Mini-League ID
   ```

---

## 📊 Dashboard Modules (4 Tabs)

### 1. 🏃 Player Analytics
* **Multi-Filter Explorer**: Filter all Premier League players by position (`GKP`, `DEF`, `MID`, `FWD`), club, maximum price (`£m`), minimum minutes played, or player name.
* **KPI Cards**: Highlights the top points scorer, highest `xGI/90`, best value (`Points per £m`), and hottest form player in the active filter view.
* **Interactive Scatter Plot**: Compare any two metrics (e.g., `xGI_per_90` vs. `pts_per_90`) with customizable bubble size (`price`, `minutes`, `selected_by_percent`).
* **Rich Master Player Table**: Includes official Premier League player headshots (`photo_url`), **Set-Piece Duty badges** (`PEN #1`, `CRN #1`, `FK #1`), **Price Trend & Net GW Transfers** (`📈 Likely Rise`, `📉 Likely Fall`), and visual progress bars (`st.column_config`) for ownership (`TSB %`) and `xGI / 90`.
* **Head-to-Head Player Percentile Radar Comparison**: Select `2–4` players to compare overlapping `0–100%` percentile polygons (`Goal Threat (xG/90)`, `Creativity (xA/90)`, `Bonus Magnet (BPS/90)`, `Recent Form`, `Value (Pts / £m)`, and `Fixture Ease (Next 5 GW)`) alongside a side-by-side raw metric table.
* **Single-Player Deep Dive**: View a player's gameweek-by-gameweek Points vs. `xGI` chart, full match log, and upcoming 8-gameweek fixture difficulty schedule.

### 2. 🛡️ Club & Opponent Analysis
* **Rolling Form Window**: Adjust the rolling match window (`3–10` matches) from the sidebar to capture recent club momentum.
* **ClubElo Live Ratings**: Integrates continuous dynamic ClubElo ratings (`Elo`) alongside minutes-normalized goalkeeper `xGC/match`.
* **Attack vs. Defense Quadrant Chart**: Plots every club's `xG/match` against `xGC/match` (with inverted Y-axis so stronger defenses appear at the top).
* **Opponent Vulnerability Tables**:
  * **Top 8 Weakest Defenses (`Target Index` & `Att FDR`)**: Clubs conceding the highest rate of goals and `xGC`—target these opponents when selecting attackers and captains.
  * **Top 8 Weakest Attacks (`CS Target Index` & `Def FDR`)**: Clubs with the lowest goal output, `xG`, and highest Failed-to-Score (`FTS`) counts—target these opponents with your Goalkeepers and Defenders.
* **Club Table & Home/Away Splits**: Full standings with `Elo`, `att_fdr_vs`, `def_fdr_vs`, and Home vs. Away `GF/match` and `GA/match`.

### 3. 📅 Gameweek Matchups & Fixture Ticker
* **Multi-Gameweek Split FDR Heatmap**: Color-coded matrix of upcoming fixtures (`GWstart` to `GWstart + N`), sorted from easiest average Fixture Difficulty Rating (`Avg FDR`) to hardest. Supports switching between **Overall FDR (Official FPL)**, **Attacking FDR (Target Weak Defenses)**, and **Defensive FDR (Target Clean Sheets)**, while handling Blank (`BGW`) and Double (`DGW`) gameweeks.
* **Single-Gameweek Multiplicative Poisson Projections**: Select any Gameweek (`1–38`) to compare Home and Away form, `xG/m`, `xGC/m`, Poisson projected goal expectancy ($\lambda_{\text{Home}}, \lambda_{\text{Away}}$), exact Poisson Clean Sheet probabilities (`Home CS %`, `Away CS %`), and key form players.
* **Gameweek Player Performance Table**: Pulls live/historical gameweek stats (`event/{gw}/live/`) to inspect player minutes, points, goals, assists, `xG`, `xA`, `xGI`, `xGC`, and `BPS` against their specific opponent.

### 4. 🏆 Manager Squad, Lineup Builder & Transfer Advisor
* **Manager Profile & Budget**: Fetches your live squad, overall rank, overall points, **Squad Value (`£m`)**, and **Bank ITB (`£m`)** using your FPL Manager ID.
* **Optimal Starting XI, 2D Pitch View & Captaincy Builder**:
  * Evaluates all valid FPL formations (`3-4-3`, `3-5-2`, `4-4-2`, `4-3-3`, `4-5-1`, `5-3-2`, `5-4-1`) and selects the legal Starting XI that maximizes next-gameweek Expected Points (`xP`).
  * Assigns **© Captain** (highest `xP`), **Vice-Captain** (second-highest `xP`), **Bench GKP**, and ordered outfield substitutes (**Bench 1–3**).
  * Includes an interactive **2D Starting XI Pitch View** (`render_pitch_view`) showing your formation, bench zone, captain badges, upcoming opponents, and effective `xP`.
* **Interactive "What-If" Squad & Transfer Sandbox**:
  * Test multiple transfers directly in the UI using **Quick-Apply Recommended Move** or **Custom Player Swap** (enforcing exact bank balance and the max 3-per-club limit).
  * Automatically updates your **Optimal Starting XI Table**, **2D Pitch View**, **Remaining Bank (`£m`)**, **Point Hit (`-4 pts` per extra transfer)**, and subsequent transfer recommendations, with one-click **Undo** and **Reset to Live Squad** buttons.
* **Chip Status & Strategy Advisor**: Tracks used vs. available chips across both half-season Wildcard windows (`Wildcard 1 (GW1–19)` and `Wildcard 2 (GW20–38)`), `Free Hit`, `Bench Boost`, and `Triple Captain`.
* **Smart Transfer & Target Recommender**:
  * **1-Transfer Moves**: Finds the best single-player replacements within your exact budget (`Selling Price + Bank`), same position, and the **max 3 players per club** rule, and calculates true **Starting XI Next GW Gain** by re-optimizing the 15-man squad after the swap.
  * **2-Player Packages**: Identifies double upgrades (automatically deducting a `-4 pt` hit if you have only `1 FT`).
  * **Top Targets Watchlist**: Lists the top 5 external targets per position sorted by multi-GW `xP`, including `Set-Piece Duty`.

---

## 🧮 Key Analytical Formulas

| Metric | Formula & Definition |
| :--- | :--- |
| **Per-90 Stats (`xG_per_90`, `xA_per_90`, `xGI_per_90`)** | $\frac{\text{Stat}}{\text{Minutes} / 90}$ |
| **Empirical Bayes Shrunk Per-90 (`shrunk_xG90`, `shrunk_xA90`)** | $\frac{\text{Nineties}}{\text{Nineties} + 3.0} \times \text{Player Per-90} + \frac{3.0}{\text{Nineties} + 3.0} \times \text{Positional Mean Per-90}$ *(shrinks low-minute outliers over a 270-minute prior)* |
| **Minutes-Normalized Club `xGC/m`** | $\frac{\sum \text{Goalkeeper } xGC}{\max(\text{Matches Played},\; \sum \text{Goalkeeper Minutes} / 90)}$ |
| **Poisson Goal Expectancy ($\lambda_{\text{Home}}, \lambda_{\text{Away}}$)** | $\lambda_{\text{Home}} = \text{Att}_{\text{Home}} \times \text{Def}_{\text{Away}} \times \overline{\text{GF}}_{\text{Home}} \times \text{EloFactor}$ |
| **Exact Clean Sheet Probability (`CS %`)** | $P(\text{CS}) = e^{-\lambda_{\text{Opponent}}}$ |
| **Set-Piece Multiplier (`sp_multiplier`)** | $1.0 + 0.08 \cdot \mathbb{I}(\text{PEN \#1}) + 0.03 \cdot \mathbb{I}(\text{PEN \#2}) + 0.04 \cdot \mathbb{I}(\text{CRN \#1}) + 0.015 \cdot \mathbb{I}(\text{CRN \#2}) + 0.02 \cdot \mathbb{I}(\text{FK \#1})$ |
| **Position-Specific Player `xP`** | Combines appearance points, position-weighted goal points (`GKP/DEF: 6`, `MID: 5`, `FWD: 4`), assist points (`3`), Poisson clean sheet points (`GKP/DEF: 4`, `MID: 1`), expected save points (`GKP`), expected `-1` per 2 goals conceded penalty (`GKP/DEF`), `BPS/90` bonus, and `sp_multiplier`. |

---

## 🔌 Official FPL & External API Endpoints Used

All requests in `src/api.py` use `@st.cache_data` (`10–60 min` TTL) to keep navigation fast and avoid rate limits:
* `https://fantasy.premierleague.com/api/bootstrap-static/` — Master players, clubs, set-piece orders, and gameweek events.
* `https://fantasy.premierleague.com/api/fixtures/` — Full season fixture list, scores, and FDR ratings.
* `https://fantasy.premierleague.com/api/element-summary/{element_id}/` — Single-player GW match log and upcoming schedule.
* `https://fantasy.premierleague.com/api/event/{event_id}/live/` — Live/historical player stats for a specific gameweek.
* `https://fantasy.premierleague.com/api/entry/{manager_id}/` — Manager profile, rank, and bank balance.
* `https://fantasy.premierleague.com/api/entry/{manager_id}/history/` — Manager GW history and chips used.
* `https://fantasy.premierleague.com/api/entry/{manager_id}/event/{event_id}/picks/` — Manager 15-man squad and captaincy picks.
* `https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/` — Classic mini-league standings.
* `http://api.clubelo.com/YYYY-MM-DD` — Live continuous ClubElo ratings for Premier League clubs (with automatic fallback to FPL strength ratings if offline).
