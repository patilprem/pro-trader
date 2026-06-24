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
    
    # 1. Flat Brokerage (Dhan standard: Rs 20 per executed order)
    brokerage = 20.0
    
    # 2. STT (Securities Transaction Tax)
    # 0.0625% on Sell side option premium (no STT on BUY side options)
    stt = 0.000625 * turnover if side.upper() == "SELL" else 0.0
    
    # 3. Exchange Transaction Charges (NSE standard: 0.053% of premium turnover)
    exchange_charges = 0.00053 * turnover
    
    # 4. GST: 18% of (Brokerage + Exchange Charges)
    gst = 0.18 * (brokerage + exchange_charges)
    
    # 5. SEBI Turnover Fees: Rs 10 per crore (0.0001% of turnover)
    sebi_charges = 0.000001 * turnover
    
    # 6. Stamp Duty: 0.003% on Buy side (no stamp duty on SELL side)
    stamp_duty = 0.00003 * turnover if side.upper() == "BUY" else 0.0
    
    total_cost = brokerage + stt + exchange_charges + gst + sebi_charges + stamp_duty
    return brokerage, stt, exchange_charges, gst, stamp_duty, total_cost


# ==============================================================================
# EVENT-DRIVEN BACKTESTER ENGINE
# ==============================================================================

class OptionsBacktester:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.ml_engine = MLEngine(db_path)
        self.starting_capital = 500000.0  # 5 Lakh INR

    def bootstrap_historical_data(self, days: int = 5):
        """Bootstraps realistic mock data in DuckDB for backtesting if database is empty."""
        con = duckdb.connect(self.db_path)
        count = con.execute("SELECT COUNT(*) FROM spot_data").fetchone()[0]
        if count > 1000:
            con.close()
            return
        
        print(f"[BACKTESTER] Bootstrapping {days} days of historical market depth data...")
        con.execute("DELETE FROM spot_data")
        con.execute("DELETE FROM order_book")
        con.execute("DELETE FROM option_chain")
        
        # Start 5 days ago
        base_time = datetime.datetime.now() - datetime.timedelta(days=days)
        spot = 22000.0
        
        for d in range(days):
            current_day = base_time + datetime.timedelta(days=d)
            # Market opens at 09:15 and closes at 15:30
            trade_time = datetime.datetime.combine(current_day.date(), datetime.time(9, 15))
            end_time = datetime.datetime.combine(current_day.date(), datetime.time(15, 30))
            
            # Avoid weekends
            if current_day.weekday() >= 5:
                continue
                
            cum_vol = 10000.0
            cum_pv = spot * cum_vol
            
            print(f"  Generating day {d+1} ({current_day.strftime('%Y-%m-%d')})...")
            
            # Tick every 30 seconds
            while trade_time <= end_time:
                # Random walk spot
                drift = 0.04 * math.sin((trade_time - datetime.datetime.combine(current_day.date(), datetime.time(9, 15))).total_seconds() / 3600.0)
                shock = random.normalvariate(0.0, 1.5)
                spot += drift + shock
                
                vol = float(random.randint(1000, 5000))
                cum_vol += vol
                cum_pv += spot * vol
                vwap = cum_pv / cum_vol
                
                # Insert Spot
                con.execute(
                    "INSERT INTO spot_data VALUES (?, ?, ?, ?, ?)",
                    (trade_time, "NIFTY-50", spot, vol, vwap)
                )
                
                # Order depth
                bid_vol = float(random.randint(100000, 200000))
                ask_vol = float(random.randint(100000, 200000))
                imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
                density = bid_vol + ask_vol
                
                # Add institutional wall ratios (simulated edge)
                bid_wall = float(random.uniform(0.1, 0.4))
                ask_wall = float(random.uniform(0.1, 0.4))
                
                con.execute(
                    "INSERT INTO order_book VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (trade_time, "NIFTY-50", spot, imbalance, density, bid_wall, ask_wall)
                )
                
                # Option chain (ATM call and put)
                atm = round(spot / 50.0) * 50
                r = 0.07
                expiry_dt = datetime.datetime.combine(current_day.date(), datetime.time(15, 30))
                rem_t = max((expiry_dt - trade_time).total_seconds() / (365.0 * 24.0 * 3600.0), 1e-5)
                
                # Option CE / PE Pricing
                c_p, c_d, c_g, c_v, c_t = black_scholes_greeks(spot, atm + 50, rem_t, r, 0.15, "CE")
                p_p, p_d, p_g, p_v, p_t = black_scholes_greeks(spot, atm - 50, rem_t, r, 0.15, "PE")
                
                con.execute(
                    "INSERT INTO option_chain VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (trade_time, "NIFTY-50_STRIKE_" + str(atm+50), atm+50, "CE", c_p, 0.15, c_d, c_g, c_v, c_t, 500000.0, 5000.0)
                )
                con.execute(
                    "INSERT INTO option_chain VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (trade_time, "NIFTY-50_STRIKE_" + str(atm-50), atm-50, "PE", p_p, 0.15, p_d, p_g, p_v, p_t, 500000.0, 5000.0)
                )
                
                trade_time += datetime.timedelta(seconds=30)
                
        con.close()
        print("[BACKTESTER] Bootstrap completed successfully.")

    def run_backtest(
        self, probability_threshold: float = 0.55, slippage_pct: float = 0.005
    ) -> Dict[str, Any]:
        """Runs the event-driven backtest on DuckDB records."""
        # Ensure database is populated
        self.bootstrap_historical_data()
        
        con = duckdb.connect(self.db_path)
        
        # Load timeseries ticks
        ticks = con.execute("""
            SELECT s.timestamp, s.ltp, s.vwap, s.volume, o.bid_imbalance, o.density, o.bid_wall_ratio, o.ask_wall_ratio
            FROM spot_data s JOIN order_book o ON s.timestamp = o.timestamp
            ORDER BY s.timestamp ASC
        """).fetchall()
        
        if not ticks:
            con.close()
            return {"error": "No database historical ticks found."}
            
        capital = self.starting_capital
        equity_curve = []
        trades = []
        
        active_trade = None
        
        print(f"[BACKTESTER] Initiating backtest. Starting Capital: {capital:.2f} INR")
        
        for idx in range(10, len(ticks)):
            t_curr, spot, vwap, vol, imbalance, density, bid_wall, ask_wall = ticks[idx]
            
            # Simple feature construction for the model prediction
            # Calculate mock skew & IV rank to fit ml_engine requirements
            iv_skew = 0.01 + 0.02 * math.sin(spot / 100.0)
            iv_rank = 0.5 + 0.1 * math.cos(spot / 50.0)
            pcr_div = 0.02 * math.sin(idx / 10.0)
            oi_vel = 1.05
            vwap_dist = (spot - vwap) / vwap
            
            # momentum lags
            spot_prev = ticks[idx - 1][1]
            spot_prev_5 = ticks[idx - 5][1]
            lag1 = (spot - spot_prev) / spot_prev
            lag5 = (spot - spot_prev_5) / spot_prev_5
            
            feature_row = [
                imbalance, density, bid_wall, ask_wall,
                iv_skew, iv_rank, pcr_div, oi_vel,
                vwap_dist, vol, lag1, lag5
            ]
            
            # Predict probability of breakout
            prob = self.ml_engine.predictive_core.predict_breakout_prob(feature_row)
            
            # Trade Monitor Loop
            if active_trade is not None:
                # Update option price
                rem_t = active_trade["rem_t"] - (30.0 / (365.0 * 24.0 * 3600.0))  # subtract 30s
                active_trade["rem_t"] = max(rem_t, 1e-5)
                
                # Fetch fresh option premium using Black-Scholes
                opt_price, _, _, _, _ = black_scholes_greeks(
                    spot, active_trade["strike"], active_trade["rem_t"], 0.07, 0.15, active_trade["type"]
                )
                
                # Check exit bounds
                if opt_price <= active_trade["stop_loss"]:
                    # Exit - Stop Loss Triggered
                    slippage = opt_price * slippage_pct
                    exit_price = max(0.1, opt_price - slippage)
                    
                    _, _, _, _, _, exit_costs = calculate_transaction_costs(exit_price, active_trade["qty"], "SELL")
                    gross_pnl = (exit_price - active_trade["entry_price"]) * active_trade["qty"]
                    net_pnl = gross_pnl - active_trade["entry_costs"] - exit_costs
                    
                    capital += net_pnl
                    trades.append({
                        "entry_time": active_trade["entry_time"],
                        "exit_time": t_curr,
                        "type": active_trade["type"],
                        "entry": active_trade["entry_price"],
                        "exit": exit_price,
                        "pnl": net_pnl,
                        "outcome": "LOSS"
                    })
                    active_trade = None
                    
                elif opt_price >= active_trade["target"]:
                    # Exit - Take Profit Triggered
                    slippage = opt_price * slippage_pct
                    exit_price = opt_price - slippage
                    
                    _, _, _, _, _, exit_costs = calculate_transaction_costs(exit_price, active_trade["qty"], "SELL")
                    gross_pnl = (exit_price - active_trade["entry_price"]) * active_trade["qty"]
                    net_pnl = gross_pnl - active_trade["entry_costs"] - exit_costs
                    
                    capital += net_pnl
                    trades.append({
                        "entry_time": active_trade["entry_time"],
                        "exit_time": t_curr,
                        "type": active_trade["type"],
                        "entry": active_trade["entry_price"],
                        "exit": exit_price,
                        "pnl": net_pnl,
                        "outcome": "WIN"
                    })
                    active_trade = None
                    
                elif t_curr.time() >= datetime.time(15, 0):
                    # Intraday MIS Clearout (3:00 PM safety rule)
                    slippage = opt_price * slippage_pct
                    exit_price = max(0.1, opt_price - slippage)
                    
                    _, _, _, _, _, exit_costs = calculate_transaction_costs(exit_price, active_trade["qty"], "SELL")
                    gross_pnl = (exit_price - active_trade["entry_price"]) * active_trade["qty"]
                    net_pnl = gross_pnl - active_trade["entry_costs"] - exit_costs
                    
                    capital += net_pnl
                    trades.append({
                        "entry_time": active_trade["entry_time"],
                        "exit_time": t_curr,
                        "type": active_trade["type"],
                        "entry": active_trade["entry_price"],
                        "exit": exit_price,
                        "pnl": net_pnl,
                        "outcome": "MIS_CLEAR"
                    })
                    active_trade = None
            
            # Entry Signal Check
            else:
                # Time validation (restrict open/close zones)
                is_valid_time = datetime.time(9, 45) <= t_curr.time() <= datetime.time(15, 0)
                
                if is_valid_time and prob >= probability_threshold:
                    # Prop Trading Microstructure filters:
                    # Only buy CALL if bid walls > ask walls, indicating institutional buying base.
                    # Only buy PUT if ask walls > bid walls, indicating institutional selling cap.
                    opt_type = "CE" if bid_wall > ask_wall else "PE"
                    
                    # Determine target strike
                    strike = round(spot / 50.0) * 50
                    strike = (strike + 50) if opt_type == "CE" else (strike - 50)
                    
                    # Calculate BS premium
                    expiry_dt = datetime.datetime.combine(t_curr.date(), datetime.time(15, 30))
                    rem_t = max((expiry_dt - t_curr).total_seconds() / (365.0 * 24.0 * 3600.0), 1e-5)
                    
                    opt_price, _, _, _, _ = black_scholes_greeks(spot, strike, rem_t, 0.07, 0.15, opt_type)
                    
                    if opt_price > 5.0:  # avoid illiquid zero value options
                        qty = config.TRADE_LOTS * config.LOT_SIZE # 250 contracts
                        
                        # Apply slippage on entry execution
                        slippage = opt_price * slippage_pct
                        entry_price = opt_price + slippage
                        
                        _, _, _, _, _, entry_costs = calculate_transaction_costs(entry_price, qty, "BUY")
                        
                        stop_loss = entry_price * (1.0 - config.STOP_LOSS_PCT)
                        target = entry_price * (1.0 + config.TARGET_PCT)
                        
                        active_trade = {
                            "entry_time": t_curr,
                            "type": opt_type,
                            "strike": strike,
                            "qty": qty,
                            "entry_price": entry_price,
                            "entry_costs": entry_costs,
                            "stop_loss": stop_loss,
                            "target": target,
                            "rem_t": rem_t
                        }
            
            # Log equity tick hourly / daily
            if idx % 120 == 0 or idx == len(ticks) - 1:
                equity_curve.append({
                    "timestamp": t_curr.strftime("%Y-%m-%d %H:%M"),
                    "equity": capital
                })

        con.close()
        
        # Calculate backtest performance stats
        total_trades = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        
        win_rate = (len(wins) / total_trades) if total_trades > 0 else 0.0
        
        total_pnl = capital - self.starting_capital
        pnl_pct = (total_pnl / self.starting_capital) * 100.0
        
        gross_profits = sum(t["pnl"] for t in wins)
        gross_losses = sum(abs(t["pnl"]) for t in losses)
        profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (gross_profits if gross_profits > 0 else 1.0)
        
        # Max drawdown calculation
        equities = [eq["equity"] for eq in equity_curve]
        peak = self.starting_capital
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Sharpe ratio (simple approximation based on trade return variance)
        trade_returns = [t["pnl"] / self.starting_capital for t in trades]
        sharpe = 0.0
        if len(trade_returns) > 2:
            mean_ret = np.mean(trade_returns)
            std_ret = np.std(trade_returns) + 1e-9
            # annualized multiplier based on trade frequency (approx 5 trades a day)
            sharpe = (mean_ret / std_ret) * math.sqrt(252 * 5)

        return {
            "starting_capital": self.starting_capital,
            "ending_capital": capital,
            "total_pnl": total_pnl,
            "total_pnl_pct": pnl_pct,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_dd * 100.0,
            "sharpe_ratio": sharpe,
            "trades": trades,
            "equity_curve": equity_curve
        }
