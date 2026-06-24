import datetime
import math
import random
import duckdb
import numpy as np
from typing import Dict, List, Tuple, Any

import config
from dhan_client import black_scholes_greeks, std_norm_cdf
from ml_engine import MLEngine

# ==============================================================================
# INDIAN REGULATORY TAXES & BROKERAGE CALCULATOR
# ==============================================================================

def calculate_transaction_costs(
    premium: float, qty: int, side: str = "BUY"
) -> Tuple[float, float, float, float, float, float]:
    """
    Computes standard brokerage and exchange fees for Indian Options contracts.
    Returns: (Brokerage, STT, Transaction Charges, GST, Stamp Duty, Total Cost)
    """
    turnover = premium * qty
    brokerage = 20.0
    stt = 0.000625 * turnover if side.upper() == "SELL" else 0.0
    exchange_charges = 0.00053 * turnover
    gst = 0.18 * (brokerage + exchange_charges)
    sebi_charges = 0.000001 * turnover
    stamp_duty = 0.00003 * turnover if side.upper() == "BUY" else 0.0
    total_cost = brokerage + stt + exchange_charges + gst + sebi_charges + stamp_duty
    return brokerage, stt, exchange_charges, gst, stamp_duty, total_cost


def get_nse_option_symbol(symbol: str, expiry_date: datetime.date,
                          strike: float, option_type: str) -> str:
    """Generates standard NSE option contract name (e.g. NIFTY2661623900CE)."""
    year_str = expiry_date.strftime("%y")
    month_char = str(expiry_date.month)
    if expiry_date.month == 10:
        month_char = "O"
    elif expiry_date.month == 11:
        month_char = "N"
    elif expiry_date.month == 12:
        month_char = "D"
    day_str = expiry_date.strftime("%d")
    strike_str = str(int(strike))
    return f"{symbol}{year_str}{month_char}{day_str}{strike_str}{option_type}"

# ==============================================================================
# EVENT-DRIVEN BACKTESTER ENGINE
# ==============================================================================

class OptionsBacktester:
    def __init__(self, db_path: str = None):
        # Always use the dedicated backtest DB — never the live feed DB
        self.db_path = db_path if db_path else config.BACKTEST_DB_PATH
        self.ml_engine = MLEngine(self.db_path)
        self.starting_capital = 500000.0  # 5 Lakh INR

    # ------------------------------------------------------------------
    # BOOTSTRAP  (5-min bars + batch inserts — runs in < 1 second)
    # ------------------------------------------------------------------
    def bootstrap_historical_data(self):
        """
        Generates deterministic June 2026 backtest data.

        Performance decisions:
          • 5-minute bars (300 s) instead of 30-second ticks
            → 75 bars/day × 20 days = 1 500 rows  (was 15 000)
          • executemany() batch-inserts all rows in 3 calls
            → eliminates ~60 000 individual DB round-trips
          • Single connection kept open throughout
        """
        TICK_SEC = 300          # 5-minute candles
        BASE_DATE = datetime.date(2026, 6, 1)

        con = duckdb.connect(self.db_path)

        # ── Create tables if missing ──────────────────────────────────
        con.execute("""
            CREATE TABLE IF NOT EXISTS spot_data (
                timestamp TIMESTAMP, symbol VARCHAR,
                ltp DOUBLE, volume DOUBLE, vwap DOUBLE
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS order_book (
                timestamp TIMESTAMP, symbol VARCHAR, ltp DOUBLE,
                bid_imbalance DOUBLE, density DOUBLE,
                bid_wall_ratio DOUBLE, ask_wall_ratio DOUBLE
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS option_chain (
                timestamp TIMESTAMP, symbol VARCHAR, strike DOUBLE,
                option_type VARCHAR, ltp DOUBLE, iv DOUBLE,
                delta DOUBLE, gamma DOUBLE, vega DOUBLE, theta DOUBLE,
                oi DOUBLE, volume DOUBLE
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS options_buying_trades (
                timestamp TIMESTAMP, entry_time TIMESTAMP,
                contract VARCHAR, strike DOUBLE, option_type VARCHAR,
                entry_price DOUBLE, exit_price DOUBLE, quantity INTEGER,
                pnl DOUBLE, outcome VARCHAR, capital DOUBLE, allocation_pct DOUBLE
            )
        """)

        # ── Always wipe — never let live-feed data pollute backtest ───
        con.execute("DELETE FROM spot_data")
        con.execute("DELETE FROM order_book")
        con.execute("DELETE FROM option_chain")
        con.execute("DELETE FROM options_buying_trades")

        print("[BACKTESTER] Generating June 2026 data (5-min bars, batch inserts)…")

        spot = 24021.65
        random.seed(42)

        spot_rows, ob_rows, oc_rows = [], [], []

        for d in range(30):
            current_day = BASE_DATE + datetime.timedelta(days=d)
            if current_day.month != 6:
                break
            if current_day.weekday() >= 5:          # skip Sat/Sun
                continue

            t = datetime.datetime.combine(current_day, datetime.time(9, 15))
            end = datetime.datetime.combine(current_day, datetime.time(15, 30))

            cum_vol = 10_000.0
            cum_pv  = spot * cum_vol

            while t <= end:
                # ── Spot random walk ──────────────────────────────────
                elapsed = (t - datetime.datetime.combine(
                    current_day, datetime.time(9, 15))).total_seconds()
                drift = 0.04 * math.sin(elapsed / 3600.0)
                shock = random.normalvariate(0.0, 3.0)   # larger σ per 5-min bar
                spot = max(20_000.0, spot + drift + shock)

                vol = float(random.randint(5_000, 25_000))
                cum_vol += vol
                cum_pv  += spot * vol
                vwap = cum_pv / cum_vol

                spot_rows.append((t, "NIFTY-50", round(spot, 2), vol, round(vwap, 2)))

                # ── Order book depth ──────────────────────────────────
                bid_vol = float(random.randint(100_000, 200_000))
                ask_vol = float(random.randint(100_000, 200_000))
                imbalance  = (bid_vol - ask_vol) / (bid_vol + ask_vol)
                density    = bid_vol + ask_vol
                bid_wall   = round(random.uniform(0.1, 0.4), 4)
                ask_wall   = round(random.uniform(0.1, 0.4), 4)

                ob_rows.append((t, "NIFTY-50", round(spot, 2),
                                round(imbalance, 4), density, bid_wall, ask_wall))

                # ── ATM options (CE + PE) ─────────────────────────────
                atm = round(spot / 50.0) * 50
                r   = 0.07
                expiry_dt = datetime.datetime.combine(current_day, datetime.time(15, 30))
                rem_t = max((expiry_dt - t).total_seconds() / (365.0 * 24.0 * 3600.0), 1e-5)

                c_p, c_d, c_g, c_v, c_t = black_scholes_greeks(
                    spot, atm + 50, rem_t, r, 0.15, "CE")
                p_p, p_d, p_g, p_v, p_t = black_scholes_greeks(
                    spot, atm - 50, rem_t, r, 0.15, "PE")

                oc_rows.append((t, f"NIFTY-50_STRIKE_{atm+50}",
                                atm + 50, "CE",
                                round(c_p, 2), 0.15,
                                round(c_d, 4), round(c_g, 6),
                                round(c_v, 4), round(c_t, 6),
                                500_000.0, 5_000.0))
                oc_rows.append((t, f"NIFTY-50_STRIKE_{atm-50}",
                                atm - 50, "PE",
                                round(p_p, 2), 0.15,
                                round(p_d, 4), round(p_g, 6),
                                round(p_v, 4), round(p_t, 6),
                                500_000.0, 5_000.0))

                t += datetime.timedelta(seconds=TICK_SEC)

        # ── Single batch-insert per table ─────────────────────────────
        con.executemany("INSERT INTO spot_data  VALUES (?,?,?,?,?)", spot_rows)
        con.executemany("INSERT INTO order_book VALUES (?,?,?,?,?,?,?)", ob_rows)
        con.executemany("INSERT INTO option_chain VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", oc_rows)
        con.close()

        unique_days = len({r[0].date() for r in spot_rows})
        print(f"[BACKTESTER] Bootstrap done: {len(spot_rows)} bars across "
              f"{unique_days} trading days.")

    # ------------------------------------------------------------------
    # OPTIONAL: download real historical data from Dhan
    # ------------------------------------------------------------------
    def download_historical_from_dhan(self, client_id: str, access_token: str) -> bool:
        """Downloads real historical minute-level data from Dhan API and seeds DuckDB."""
        try:
            from dhanhq import dhanhq
        except ImportError:
            print("[BACKTESTER] dhanhq SDK not installed. Skipping live download.")
            return False

        if client_id in (None, "MOCK_CLIENT_ID") or access_token in (None, "MOCK_ACCESS_TOKEN"):
            print("[BACKTESTER] Mock credentials. Skipping live download.")
            return False

        print("[BACKTESTER] Connecting to Dhan REST API for historical download…")
        try:
            try:
                dhan = dhanhq(client_id=client_id, access_token=access_token)
            except TypeError:
                dhan = dhanhq(client_id, access_token)
        except Exception as e:
            print(f"[BACKTESTER] Failed to connect to Dhan: {e}")
            return False

        try:
            con = duckdb.connect(self.db_path)
            con.execute("""
                CREATE TABLE IF NOT EXISTS spot_data (
                    timestamp TIMESTAMP, symbol VARCHAR,
                    ltp DOUBLE, volume DOUBLE, vwap DOUBLE
                )
            """)
            con.execute("DELETE FROM spot_data")

            today = datetime.date.today()
            from_date = (today - datetime.timedelta(days=60)).strftime("%Y-%m-%d")
            to_date   = today.strftime("%Y-%m-%d")

            resp = dhan.intraday_daily_minute_charts(
                security_id="13",
                exchange_segment="IDX_I",
                instrument_type="INDEX",
                expiry_code=0,
                from_date=from_date,
                to_date=to_date
            )

            rows = []
            if resp.get("status") == "success" and "data" in resp:
                for candle in resp["data"]:
                    ts = datetime.datetime.fromisoformat(candle.get("start_Time", candle.get("time", "")))
                    ltp = float(candle.get("close", candle.get("ltp", 0.0)))
                    vol = float(candle.get("volume", 0.0))
                    rows.append((ts, "NIFTY-50", ltp, vol, ltp))

            if rows:
                con.executemany("INSERT INTO spot_data VALUES (?,?,?,?,?)", rows)
                con.close()
                print(f"[BACKTESTER] Downloaded {len(rows)} real candles from Dhan.")
                return True
            con.close()
        except Exception as e:
            print(f"[BACKTESTER] Failed to download historical data: {e}")
        return False

    # ------------------------------------------------------------------
    # MAIN BACKTEST LOOP
    # ------------------------------------------------------------------
    def run_backtest(
        self,
        probability_threshold: float = 0.55,
        slippage_pct: float = 0.005,
        client_id: str = None,
        access_token: str = None
    ) -> Dict[str, Any]:
        """Runs the event-driven backtest on 5-min bar data."""

        # Step 1: Populate data
        downloaded = False
        if client_id and access_token:
            downloaded = self.download_historical_from_dhan(client_id, access_token)
        if not downloaded:
            self.bootstrap_historical_data()

        # Step 2: Open a FRESH connection AFTER bootstrap writes are committed
        TICK_SEC = 300.0   # must match bootstrap tick interval

        con = duckdb.connect(self.db_path)

        con.execute("""
            CREATE TABLE IF NOT EXISTS options_buying_trades (
                timestamp TIMESTAMP, entry_time TIMESTAMP,
                contract VARCHAR, strike DOUBLE, option_type VARCHAR,
                entry_price DOUBLE, exit_price DOUBLE, quantity INTEGER,
                pnl DOUBLE, outcome VARCHAR, capital DOUBLE, allocation_pct DOUBLE
            )
        """)
        con.execute("DELETE FROM options_buying_trades")

        # Step 3: Load ticks (joined spot + order_book)
        ticks = con.execute("""
            SELECT s.timestamp, s.ltp, s.vwap, s.volume,
                   o.bid_imbalance, o.density, o.bid_wall_ratio, o.ask_wall_ratio
            FROM spot_data s
            JOIN order_book o ON s.timestamp = o.timestamp
            WHERE CAST(strftime(s.timestamp, '%H:%M') AS TIME) >= '09:15'
              AND CAST(strftime(s.timestamp, '%H:%M') AS TIME) <= '15:30'
            ORDER BY s.timestamp ASC
        """).fetchall()

        if not ticks:
            con.close()
            return {"error": "No historical ticks found after bootstrap."}

        print(f"[BACKTESTER] Loaded {len(ticks)} bars. Running event-driven loop…")

        capital      = self.starting_capital
        equity_curve = []
        trades       = []
        trade_rows   = []    # batched for final executemany

        active_trade  = None
        current_date  = None
        trades_today  = 0
        LOOKBACK      = 5

        for idx in range(LOOKBACK, len(ticks)):
            t_curr, spot, vwap, vol, imbalance, density, bid_wall, ask_wall = ticks[idx]

            # ── Reset daily counter ───────────────────────────────────
            t_date = t_curr.date() if hasattr(t_curr, "date") else t_curr
            if current_date != t_date:
                current_date = t_date
                trades_today = 0

            # ── Feature construction ──────────────────────────────────
            iv_skew  = 0.01 + 0.02 * math.sin(spot / 100.0)
            iv_rank  = 0.50 + 0.10 * math.cos(spot / 50.0)
            pcr_div  = 0.02 * math.sin(idx / 10.0)
            oi_vel   = 1.05
            vwap_dist = (spot - vwap) / vwap

            spot_prev   = ticks[idx - 1][1]
            spot_prev_5 = ticks[idx - 5][1]
            lag1 = (spot - spot_prev)   / (spot_prev   + 1e-9)
            lag5 = (spot - spot_prev_5) / (spot_prev_5 + 1e-9)

            feature_row = [imbalance, density, bid_wall, ask_wall,
                           iv_skew, iv_rank, pcr_div, oi_vel,
                           vwap_dist, vol, lag1, lag5]

            prob = self.ml_engine.predictive_core.predict_breakout_prob(feature_row)

            # ── Monitor open trade ────────────────────────────────────
            if active_trade is not None:
                # Decay remaining time by one 5-min bar
                active_trade["rem_t"] = max(
                    active_trade["rem_t"] - TICK_SEC / (365.0 * 24.0 * 3600.0),
                    1e-5
                )
                opt_price, *_ = black_scholes_greeks(
                    spot, active_trade["strike"],
                    active_trade["rem_t"], 0.07, 0.15, active_trade["type"]
                )

                exit_reason = None
                if opt_price <= active_trade["stop_loss"]:
                    exit_reason = "LOSS"
                elif opt_price >= active_trade["target"]:
                    exit_reason = "WIN"
                elif hasattr(t_curr, "time") and t_curr.time() >= datetime.time(15, 0):
                    exit_reason = "MIS_CLEAR"

                if exit_reason:
                    slip = opt_price * slippage_pct
                    exit_px = max(0.10, opt_price - slip)
                    *_, exit_costs = calculate_transaction_costs(exit_px, active_trade["qty"], "SELL")
                    gross  = (exit_px - active_trade["entry_price"]) * active_trade["qty"]
                    net    = gross - active_trade["entry_costs"] - exit_costs
                    capital += net

                    trade = {
                        "entry_time": active_trade["entry_time"],
                        "exit_time":  t_curr,
                        "type":       active_trade["type"],
                        "entry":      active_trade["entry_price"],
                        "exit":       exit_px,
                        "pnl":        net,
                        "outcome":    exit_reason,
                        "contract":   active_trade["contract"]
                    }
                    trades.append(trade)

                    # Queue row for batch insert
                    trade_rows.append((
                        t_curr, active_trade["entry_time"],
                        active_trade["contract"], active_trade["strike"],
                        active_trade["type"], active_trade["entry_price"],
                        exit_px, active_trade["qty"], net, exit_reason,
                        active_trade["capital"], active_trade["allocation_pct"]
                    ))
                    active_trade = None

            # ── Entry signal check ────────────────────────────────────
            else:
                t_time = t_curr.time() if hasattr(t_curr, "time") else t_curr
                is_valid = datetime.time(9, 45) <= t_time <= datetime.time(15, 0)

                if is_valid and prob >= probability_threshold and trades_today < 3:
                    if   bid_wall > ask_wall and imbalance >  0.15:
                        opt_type = "CE"
                    elif ask_wall > bid_wall and imbalance < -0.15:
                        opt_type = "PE"
                    else:
                        equity_curve.append({"timestamp": str(t_curr), "equity": capital})
                        continue

                    strike    = round(spot / 50.0) * 50
                    expiry_dt = datetime.datetime.combine(
                        t_curr.date() if hasattr(t_curr, "date") else t_curr,
                        datetime.time(15, 30))
                    rem_t = max(
                        (expiry_dt - t_curr).total_seconds() / (365.0 * 24.0 * 3600.0),
                        1e-5
                    )
                    opt_price, *_ = black_scholes_greeks(spot, strike, rem_t, 0.07, 0.15, opt_type)

                    if opt_price > 5.0:
                        qty = config.TRADE_LOTS * config.LOT_SIZE
                        entry_px = opt_price * (1.0 + slippage_pct)
                        *_, entry_costs = calculate_transaction_costs(entry_px, qty, "BUY")

                        stop_loss = entry_px * (1.0 - config.STOP_LOSS_PCT)
                        target    = entry_px * (1.0 + config.TARGET_PCT)
                        contract  = get_nse_option_symbol(
                            "NIFTY",
                            expiry_dt.date() if hasattr(expiry_dt, "date") else expiry_dt,
                            strike, opt_type)

                        active_trade = {
                            "entry_time":   t_curr,
                            "type":         opt_type,
                            "strike":       strike,
                            "qty":          qty,
                            "entry_price":  entry_px,
                            "entry_costs":  entry_costs,
                            "stop_loss":    stop_loss,
                            "target":       target,
                            "rem_t":        rem_t,
                            "contract":     contract,
                            "capital":      capital,
                            "allocation_pct": (entry_px * qty) / capital * 100
                        }
                        trades_today += 1

            equity_curve.append({"timestamp": str(t_curr), "equity": capital})

        # ── Batch-insert all completed trades ─────────────────────────
        if trade_rows:
            con.executemany("""
                INSERT INTO options_buying_trades VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?)
            """, trade_rows)

        con.close()

        # ── Performance statistics ────────────────────────────────────
        total_trades = len(trades)
        wins   = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]

        win_rate      = len(wins) / total_trades if total_trades > 0 else 0.0
        total_pnl     = capital - self.starting_capital
        pnl_pct       = (total_pnl / self.starting_capital) * 100.0
        gross_profits = sum(t["pnl"] for t in wins)
        gross_losses  = sum(abs(t["pnl"]) for t in losses)
        profit_factor = (gross_profits / gross_losses) if gross_losses > 0 \
                        else (gross_profits if gross_profits > 0 else 1.0)

        equities = [eq["equity"] for eq in equity_curve]
        peak, max_dd = self.starting_capital, 0.0
        for eq in equities:
            peak = max(peak, eq)
            dd   = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        trade_returns = [t["pnl"] / self.starting_capital for t in trades]
        sharpe = 0.0
        if len(trade_returns) > 2:
            mean_r = np.mean(trade_returns)
            std_r  = np.std(trade_returns) + 1e-9
            sharpe = (mean_r / std_r) * math.sqrt(252 * 3)

        print(f"[BACKTESTER] Complete. {total_trades} trades | "
              f"Win rate: {win_rate*100:.1f}% | PnL: {pnl_pct:+.2f}%")

        return {
            "starting_capital":  self.starting_capital,
            "ending_capital":    capital,
            "total_pnl":         total_pnl,
            "total_pnl_pct":     pnl_pct,
            "total_trades":      total_trades,
            "win_rate":          win_rate,
            "profit_factor":     profit_factor,
            "max_drawdown_pct":  max_dd * 100.0,
            "sharpe_ratio":      sharpe,
            "trades":            trades,
            "equity_curve":      equity_curve
        }
