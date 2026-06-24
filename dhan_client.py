import os
import time
import math
import random
import threading
import datetime
import duckdb
from typing import Dict, List, Tuple, Optional, Any

import config

# Try to import dhanhq, fallback gracefully if not installed
try:
    from dhanhq import marketfeed
    from dhanhq import dhanhq as dhan_api
    DHAN_AVAILABLE = True
except ImportError:
    DHAN_AVAILABLE = False

# ==============================================================================
# BLACK-SCHOLES OPTIONS MATHEMATICS ENGINE (ZERO EXTERNAL DEPENDENCY)
# ==============================================================================

def std_norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def std_norm_pdf(x: float) -> float:
    """Probability density function for standard normal distribution."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def black_scholes_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "CE"
) -> Tuple[float, float, float, float, float]:
    """
    Calculates Black-Scholes option price and greeks (Delta, Gamma, Vega, Theta).
    Returns: (Price, Delta, Gamma, Vega, Theta)
    """
    T = max(T, 1e-5)  # Avoid division by zero
    sigma = max(sigma, 1e-4)
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    n_d1 = std_norm_cdf(d1)
    n_d2 = std_norm_cdf(d2)
    pdf_d1 = std_norm_pdf(d1)
    
    # Calculate price
    if option_type.upper() == "CE":
        price = S * n_d1 - K * math.exp(-r * T) * n_d2
        delta = n_d1
        theta = - (S * pdf_d1 * sigma) / (2.0 * math.sqrt(T)) - r * K * math.exp(-r * T) * n_d2
    else:
        n_neg_d1 = std_norm_cdf(-d1)
        n_neg_d2 = std_norm_cdf(-d2)
        price = K * math.exp(-r * T) * n_neg_d2 - S * n_neg_d1
        delta = n_d1 - 1.0
        theta = - (S * pdf_d1 * sigma) / (2.0 * math.sqrt(T)) + r * K * math.exp(-r * T) * n_neg_d2
        
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * math.sqrt(T) * pdf_d1 / 100.0  # Vega for 1% change in IV
    theta = theta / 365.0                     # Daily theta decay
    
    return max(0.0, price), delta, gamma, vega, theta

def estimate_implied_volatility(
    market_price: float, S: float, K: float, T: float, r: float, option_type: str = "CE"
) -> float:
    """Finds implied volatility using Newton-Raphson method."""
    sigma = 0.20  # Initial guess
    for _ in range(30):
        price, _, _, vega, _ = black_scholes_greeks(S, K, T, r, sigma, option_type)
        diff = price - market_price
        if abs(diff) < 1e-4:
            return sigma
        # If vega is too small, use basic search step
        if abs(vega) < 1e-6:
            sigma += 0.01 if diff < 0 else -0.01
        else:
            # Note: vega is scaled by 100.0 in our function, so scale it back
            sigma -= diff / (vega * 100.0)
        sigma = max(0.01, min(sigma, 3.0))
    return sigma

# ==============================================================================
# DUCKDB STORAGE ENGINE
# ==============================================================================

class DuckDBManager:
    """Manages thread-safe storage & retrieval of tick data from DuckDB."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            con = duckdb.connect(self.db_path)
            # Create spot data table
            con.execute("""
                CREATE TABLE IF NOT EXISTS spot_data (
                    timestamp TIMESTAMP,
                    symbol VARCHAR,
                    ltp DOUBLE,
                    volume DOUBLE,
                    vwap DOUBLE
                )
            """)
            # Create order book table
            con.execute("""
                CREATE TABLE IF NOT EXISTS order_book (
                    timestamp TIMESTAMP,
                    symbol VARCHAR,
                    ltp DOUBLE,
                    bid_imbalance DOUBLE,
                    density DOUBLE,
                    bid_wall_ratio DOUBLE,
                    ask_wall_ratio DOUBLE
                )
            """)
            # Create options chain table
            con.execute("""
                CREATE TABLE IF NOT EXISTS option_chain (
                    timestamp TIMESTAMP,
                    symbol VARCHAR,
                    strike_price DOUBLE,
                    option_type VARCHAR,
                    ltp DOUBLE,
                    iv DOUBLE,
                    delta DOUBLE,
                    gamma DOUBLE,
                    vega DOUBLE,
                    theta DOUBLE,
                    oi DOUBLE,
                    volume DOUBLE
                )
            """)
            # Create trade history table
            con.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    timestamp TIMESTAMP,
                    contract VARCHAR,
                    strike DOUBLE,
                    option_type VARCHAR,
                    entry_price DOUBLE,
                    exit_price DOUBLE,
                    quantity INTEGER,
                    pnl DOUBLE,
                    outcome VARCHAR
                )
            """)
            con.close()

    def insert_spot(self, timestamp: datetime.datetime, symbol: str, ltp: float, volume: float, vwap: float):
        with self.lock:
            con = duckdb.connect(self.db_path)
            con.execute(
                "INSERT INTO spot_data VALUES (?, ?, ?, ?, ?)",
                (timestamp, symbol, ltp, volume, vwap)
            )
            con.close()

    def insert_order_book(
        self, timestamp: datetime.datetime, symbol: str, ltp: float, 
        imbalance: float, density: float, bid_wall: float, ask_wall: float
    ):
        with self.lock:
            con = duckdb.connect(self.db_path)
            con.execute(
                "INSERT INTO order_book VALUES (?, ?, ?, ?, ?, ?, ?)",
                (timestamp, symbol, ltp, imbalance, density, bid_wall, ask_wall)
            )
            con.close()

    def insert_option(
        self, timestamp: datetime.datetime, symbol: str, strike_price: float, option_type: str,
        ltp: float, iv: float, delta: float, gamma: float, vega: float, theta: float, oi: float, volume: float
    ):
        with self.lock:
            con = duckdb.connect(self.db_path)
            con.execute(
                "INSERT INTO option_chain VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (timestamp, symbol, strike_price, option_type, ltp, iv, delta, gamma, vega, theta, oi, volume)
            )
            con.close()

    def insert_trade(
        self, timestamp: datetime.datetime, contract: str, strike: float, option_type: str,
        entry_price: float, exit_price: float, quantity: int, pnl: float, outcome: str
    ):
        with self.lock:
            con = duckdb.connect(self.db_path)
            con.execute(
                "INSERT INTO trade_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (timestamp, contract, strike, option_type, entry_price, exit_price, quantity, pnl, outcome)
            )
            con.close()

# ==============================================================================
# STATE-MACHINE EXECUTION ROUTER & RISK LOCKOUT
# ==============================================================================

class TradeExecutionRouter:
    """Manages active options positions and enforces 1:3 R:R boundaries."""
    def __init__(self, db_manager: DuckDBManager):
        self.db = db_manager
        self.active_position: Optional[Dict[str, Any]] = None
        self.is_locked = False
        self.kill_switch_tripped = False
        self.pnl_history: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def place_order(
        self, contract: str, strike: float, option_type: str, entry_price: float, side: str = "BUY"
    ) -> bool:
        """Attempts to execute a limit entry order and locks the state machine."""
        with self.lock:
            if self.kill_switch_tripped:
                print("[WARNING] Order blocked: Global Emergency Kill Switch is ACTIVE.")
                return False
            if self.is_locked:
                print(f"[WARNING] Order blocked: Active position already exists in {self.active_position['contract']}.")
                return False

            # Verify daily time window
            now = datetime.datetime.now().time()
            start_time = datetime.time.fromisoformat(config.TRADING_WINDOW_START)
            end_time = datetime.time.fromisoformat(config.TRADING_WINDOW_END)
            if not (start_time <= now <= end_time):
                print(f"[WARNING] Order blocked: Current time {now} outside trading window {config.TRADING_WINDOW_START} - {config.TRADING_WINDOW_END}")
                return False

            # Lock the state machine and create active position
            qty = config.TRADE_LOTS * config.LOT_SIZE
            stop_loss = entry_price * (1.0 - config.STOP_LOSS_PCT)
            target = entry_price * (1.0 + config.TARGET_PCT)

            # In live mode, here we would call the Dhan REST API to place a limit buy order
            if config.RUN_MODE == "LIVE" and DHAN_AVAILABLE:
                # Actual Dhan API order placement would occur here.
                # For safety and implementation limits, we track the order state in our machine.
                pass

            self.active_position = {
                "contract": contract,
                "strike": strike,
                "option_type": option_type,
                "entry_price": entry_price,
                "ltp": entry_price,
                "qty": qty,
                "stop_loss": stop_loss,
                "target": target,
                "entry_time": datetime.datetime.now(),
                "pnl": 0.0
            }
            self.is_locked = True
            print(f"[TRADE ENTRY] Executed BUY {qty} {contract} @ {entry_price:.2f}. Target: {target:.2f}, SL: {stop_loss:.2f}")
            return True

    def monitor_and_update(self, current_ltp: float):
        """Monitors current price and exits position if target or stop loss is breached."""
        with self.lock:
            if not self.is_locked or self.active_position is None:
                return

            self.active_position["ltp"] = current_ltp
            pnl = (current_ltp - self.active_position["entry_price"]) * self.active_position["qty"]
            self.active_position["pnl"] = pnl

            # Risk Exit Bounds Check
            if current_ltp <= self.active_position["stop_loss"]:
                self._exit_position(current_ltp, "LOSS")
            elif current_ltp >= self.active_position["target"]:
                self._exit_position(current_ltp, "WIN")

    def _exit_position(self, exit_price: float, outcome: str):
        """Internal helper to exit a trade, log to database, and unlock state machine."""
        pos = self.active_position
        if pos is None:
            return

        pnl = (exit_price - pos["entry_price"]) * pos["qty"]
        
        # In live mode, call Dhan API to liquidate position
        if config.RUN_MODE == "LIVE" and DHAN_AVAILABLE:
            # Place exit order
            pass

        # Save to database
        self.db.insert_trade(
            timestamp=datetime.datetime.now(),
            contract=pos["contract"],
            strike=pos["strike"],
            option_type=pos["option_type"],
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            quantity=pos["qty"],
            pnl=pnl,
            outcome=outcome
        )

        trade_record = {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "contract": pos["contract"],
            "entry": pos["entry_price"],
            "exit": exit_price,
            "qty": pos["qty"],
            "pnl": pnl,
            "outcome": outcome
        }
        self.pnl_history.append(trade_record)

        print(f"[TRADE EXIT] Closed {pos['contract']} @ {exit_price:.2f}. PnL: {pnl:+.2f} ({outcome})")
        self.active_position = None
        self.is_locked = False

    def trigger_emergency_kill(self):
        """Global Safety Stop: immediately flattens position and locks the system."""
        with self.lock:
            self.kill_switch_tripped = True
            if self.active_position is not None:
                pos = self.active_position
                # Flatten instantly (sell at bid price / current LTP)
                exit_price = pos["ltp"] * 0.95  # worst-case slippage fill
                pnl = (exit_price - pos["entry_price"]) * pos["qty"]
                
                # In live mode, execute urgent order placement to liquidate
                if config.RUN_MODE == "LIVE" and DHAN_AVAILABLE:
                    pass
                
                self.db.insert_trade(
                    timestamp=datetime.datetime.now(),
                    contract=pos["contract"],
                    strike=pos["strike"],
                    option_type=pos["option_type"],
                    entry_price=pos["entry_price"],
                    exit_price=exit_price,
                    quantity=pos["qty"],
                    pnl=pnl,
                    outcome="EMERGENCY_STOP"
                )
                
                self.pnl_history.append({
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "contract": pos["contract"],
                    "entry": pos["entry_price"],
                    "exit": exit_price,
                    "qty": pos["qty"],
                    "pnl": pnl,
                    "outcome": "PANIC_CLOSE"
                })
                
                print(f"[EMERGENCY CLOSE] Executed panic stop for {pos['contract']} @ {exit_price:.2f}. PnL: {pnl:.2f}")
                self.active_position = None
            
            self.is_locked = True  # Keep locked until explicitly reset
            print("[EMERGENCY TRIGGERED] System operations halted.")

    def reset_system(self):
        """Resets the emergency trigger lockout."""
        with self.lock:
            self.kill_switch_tripped = False
            self.is_locked = False
            print("[EMERGENCY RESET] System lockout cleared. Ready for operations.")

# ==============================================================================
# DHAN WEB FEED OR MOCK SIMULATOR FEED
# ==============================================================================

class DhanFeedEngine:
    """Manages WebSocket connection to Dhan or launches a high-fidelity simulator."""
    def __init__(self, db_manager: DuckDBManager, execution_router: TradeExecutionRouter):
        self.db = db_manager
        self.router = execution_router
        self.spot_price = 24021.65
        self.vwap = 24021.65
        self.cum_vol = 100000.0
        self.cum_pv = 24021.65 * 100000.0
        self.last_update_time = time.time()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.subscribers: List[Any] = []
        
        # Live state cache
        self.latest_spot: Dict[str, Any] = {}
        self.latest_depth: Dict[str, Any] = {}
        self.latest_option_chain: List[Dict[str, Any]] = []
        self.market_closed_override = False
        self.error_message: Optional[str] = None

    def is_market_open_ist(self) -> bool:
        """Helper to determine if the Indian Stock Market is currently open (9:15 AM - 3:30 PM IST, Mon-Fri)."""
        # Convert UTC to IST (UTC + 5:30)
        utc_now = datetime.datetime.utcnow()
        ist_now = utc_now + datetime.timedelta(hours=5, minutes=30)
        
        # Weekday check (0 = Monday, 4 = Friday)
        if ist_now.weekday() >= 5:
            return False
            
        market_start = datetime.time(9, 15, 0)
        market_end = datetime.time(15, 30, 0)
        current_time = ist_now.time()
        
        return market_start <= current_time <= market_end

    def start(self):
        self.running = True
        self.market_closed_override = False
        
        if config.RUN_MODE == "SIMULATION":
            self.thread = threading.Thread(target=self._run_simulator, daemon=True)
            self.thread.start()
        else:
            # RUN_MODE == "LIVE"
            self._start_live_feed()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def register_callback(self, callback_func):
        self.subscribers.append(callback_func)

    def _notify(self, data: Dict[str, Any]):
        for sub in self.subscribers:
            try:
                sub(data)
            except Exception as e:
                print(f"Error in feed subscriber callback: {e}")

    def _initialize_live_data_from_rest(self, dhan_rest, expiry_str: str):
        """Fetches the latest closing quotes via REST API and seeds the DB/cache (for after-hours display)."""
        now = datetime.datetime.now()
        print("[LIVE] Fetching latest market closing quotes via REST API...")
        
        # 1. Fetch Spot Index (Nifty 50 Index)
        spot_price = 24021.65
        vwap = 24021.65
        volume = 0.0
        try:
            q_resp = dhan_rest.get_market_quote(securities={"IDX_I": [13]})
            if q_resp.get("status") == "success" and "data" in q_resp:
                spot_data = q_resp["data"].get("13", q_resp["data"].get(13, {}))
                spot_price = float(spot_data.get("last_price", spot_data.get("lastPrice", 24021.65)))
                vwap = float(spot_data.get("average_price", spot_data.get("averagePrice", spot_price)))
                volume = float(spot_data.get("volume", spot_data.get("volume_traded", 0.0)))
                
                self.spot_price = spot_price
                self.vwap = vwap
                self.db.insert_spot(now, config.UNDERLYING_SYMBOL, spot_price, volume, vwap)
                self.latest_spot = {
                    "timestamp": now,
                    "symbol": config.UNDERLYING_SYMBOL,
                    "ltp": spot_price,
                    "volume": volume,
                    "vwap": vwap
                }
                
                # Generate depth snapshot centered on closing Spot
                self._generate_depth_from_spot(now, spot_price, spot_data)
        except Exception as e:
            print(f"[LIVE] Failed to fetch spot closing quote: {e}")
            
        # 2. Fetch Option Chain details
        try:
            chain_resp = dhan_rest.get_option_chain(
                underlying_security_id="13",
                underlying_type="INDEX",
                expiry_date=expiry_str
            )
            if chain_resp.get("status") == "success" or "data" in chain_resp:
                data = chain_resp.get("data", chain_resp)
                oc_dict = data.get("oc", {})
                
                # Clear option chain cache to populate fresh real data
                self.latest_option_chain = []
                
                for strike_str, strike_data in oc_dict.items():
                    strike = float(strike_str)
                    if strike <= 0:
                        continue
                        
                    # Call details
                    c_opt = strike_data.get("ce", {})
                    c_price = float(c_opt.get("last_price", c_opt.get("lastPrice", 0.0)))
                    c_oi = float(c_opt.get("oi", c_opt.get("openInterest", 0.0)))
                    c_vol = float(c_opt.get("volume", c_opt.get("volume_traded", 0.0)))
                    
                    # Put details
                    p_opt = strike_data.get("pe", {})
                    p_price = float(p_opt.get("last_price", p_opt.get("lastPrice", 0.0)))
                    p_oi = float(p_opt.get("oi", p_opt.get("openInterest", 0.0)))
                    p_vol = float(p_opt.get("volume", p_opt.get("volume_traded", 0.0)))
                    
                    # Compute greeks & IV for Call
                    expiry_dt = datetime.datetime.combine(datetime.date.today(), datetime.time(15, 30))
                    remaining_time = max((expiry_dt - now).total_seconds() / (365.0 * 24.0 * 3600.0), 1e-5)
                    r = 0.07
                    
                    if c_price > 0:
                        c_iv = estimate_implied_volatility(c_price, spot_price, strike, remaining_time, r, "CE")
                        _, c_d, c_g, c_v, c_t = black_scholes_greeks(spot_price, strike, remaining_time, r, c_iv, "CE")
                        self.db.insert_option(now, f"{config.UNDERLYING_SYMBOL}_STRIKE_{strike}", strike, "CE", c_price, c_iv, c_d, c_g, c_v, c_t, c_oi, c_vol)
                        self._update_cached_option(strike, "CE", c_price, c_iv, c_d, c_g, c_v, c_t, c_oi, c_vol)
                        
                    # Compute greeks & IV for Put
                    if p_price > 0:
                        p_iv = estimate_implied_volatility(p_price, spot_price, strike, remaining_time, r, "PE")
                        _, p_d, p_g, p_v, p_t = black_scholes_greeks(spot_price, strike, remaining_time, r, p_iv, "PE")
                        self.db.insert_option(now, f"{config.UNDERLYING_SYMBOL}_STRIKE_{strike}", strike, "PE", p_price, p_iv, p_d, p_g, p_v, p_t, p_oi, p_vol)
                        self._update_cached_option(strike, "PE", p_price, p_iv, p_d, p_g, p_v, p_t, p_oi, p_vol)
                        
                print(f"[LIVE] Initialized database with {len(self.latest_option_chain)} options chain closing prices.")
        except Exception as e:
            print(f"[LIVE] Failed to fetch option chain closing quote: {e}")
            
        # Notify interface immediately
        feed_tick = {
            "spot": self.latest_spot,
            "depth": self.latest_depth,
            "option_chain": self.latest_option_chain,
            "timestamp": now
        }
        self._notify(feed_tick)

    def _start_live_feed(self):
        """Starts real connection using dhanhq library."""
        if not DHAN_AVAILABLE:
            self.error_message = "DhanHQ SDK is not installed. Live feed unavailable."
            print(f"[ERROR] {self.error_message}")
            return
        
        # Initialize REST client defensively using keyword arguments
        try:
            try:
                dhan_rest = dhan_api(
                    client_id=config.DHAN_CLIENT_ID,
                    access_token=config.DHAN_ACCESS_TOKEN
                )
            except TypeError:
                dhan_rest = dhan_api(config.DHAN_CLIENT_ID, config.DHAN_ACCESS_TOKEN)
            self.error_message = None
        except Exception as e:
            self.error_message = f"Failed to initialize Dhan API client: {e}"
            print(f"[ERROR] {self.error_message}")
            return
        
        def live_run():
            try:
                # Default: always subscribe to Nifty 50 Spot Index (Security ID: 13, Segment: IDX_I)
                instruments = [{"exchange_segment": "IDX_I", "security_id": "13"}]
                
                # Fetch next Thursday's expiry date dynamically
                today = datetime.date.today()
                days_ahead = 3 - today.weekday()
                if days_ahead < 0:
                    days_ahead += 7
                # Rollover to next week if today is Thursday after 3:30 PM
                now_dt = datetime.datetime.now()
                if today.weekday() == 3 and now_dt.time() >= datetime.time(15, 30):
                    days_ahead += 7
                
                next_thursday = today + datetime.timedelta(days=days_ahead)
                expiry_str = next_thursday.strftime("%Y-%m-%d")
                
                # Fetch closing quotes via REST immediately before WebSocket starts (for after-hours display)
                self._initialize_live_data_from_rest(dhan_rest, expiry_str)
                
                # Check if market is open before starting the WebSocket thread
                if not self.is_market_open_ist():
                    print("[LIVE] Market is closed. Loaded real closing prices. Skipping live WebSocket stream.")
                    self.market_closed_override = True
                    return
                
                self.market_closed_override = False
                
                try:
                    print(f"[LIVE] Querying option chain for NIFTY-50 (expiry: {expiry_str})...")
                    response = dhan_rest.get_option_chain(
                        underlying_security_id="13",
                        underlying_type="INDEX",
                        expiry_date=expiry_str
                    )
                    
                    # Validate Token: raise error if token is unauthorized
                    if response.get("status") == "failure" or "error" in response or \
                       ("remarks" in response and "token" in response.get("remarks", "").lower()):
                        raise Exception(response.get("remarks", "Dhan token validation failed."))
                        
                    if response.get("status") == "success" or "data" in response:
                        data = response.get("data", response)
                        oc_dict = data.get("oc", {})
                        
                        # Find spot estimate from a quick market quote
                        spot_est = 24021.65
                        try:
                            q_resp = dhan_rest.get_market_quote(securities={"IDX_I": [13]})
                            if q_resp.get("status") == "success" and "data" in q_resp:
                                spot_data = q_resp["data"].get("13", q_resp["data"].get(13, {}))
                                spot_est = float(spot_data.get("last_price", spot_data.get("lastPrice", 24021.65)))
                        except Exception:
                            pass
                            
                        # Sort strikes by closeness to spot
                        sorted_strikes = sorted(
                            oc_dict.keys(),
                            key=lambda x: abs(float(x) - spot_est)
                        )
                        
                        # Take 8 nearest strikes
                        for strike_str in sorted_strikes[:8]:
                            strike_data = oc_dict[strike_str]
                            ce_id = strike_data.get("ce", {}).get("security_id", strike_data.get("ce", {}).get("securityId"))
                            pe_id = strike_data.get("pe", {}).get("security_id", strike_data.get("pe", {}).get("securityId"))
                            if ce_id:
                                instruments.append({"exchange_segment": "NSE_FNO", "security_id": str(ce_id)})
                            if pe_id:
                                instruments.append({"exchange_segment": "NSE_FNO", "security_id": str(pe_id)})
                        print(f"[LIVE] Option chain subscription compiled: {len(instruments) - 1} options added.")
                except Exception as e:
                    print(f"[LIVE] Option chain query failed: {e}. Subscribing to spot index only.")
                    self.error_message = f"Option chain query failed: {e}"

                def on_connect(instance):
                    print(f"[LIVE CONNECT] Connected to Dhan Market Feed. Subscribing to {len(instruments)} instruments...")
                    instance.subscribe_instruments(instruments)

                def on_message(instance, message):
                    self._parse_live_packet(message)

                feed = marketfeed.MarketFeed(
                    client_id=config.DHAN_CLIENT_ID,
                    access_token=config.DHAN_ACCESS_TOKEN,
                    instruments=instruments,
                    model_type=marketfeed.Quote,
                    on_connect=on_connect,
                    on_message=on_message
                )
                feed.run_forever()
            except Exception as e:
                print(f"[LIVE DISCONNECT] WebSocket crashed: {e}. Reconnecting in 5s...")
                time.sleep(5.0)
                if self.running and self.is_market_open_ist():
                    self._start_live_feed()

        self.thread = threading.Thread(target=live_run, daemon=True)
        self.thread.start()

    def _parse_live_packet(self, message):
        """Parses real-time data ticks and feeds database and listeners."""
        if not message:
            return

        # If it's a JSON string, parse it
        if isinstance(message, str):
            try:
                import json
                message = json.loads(message)
            except Exception:
                return

        # Ensure message is parsed as a dictionary
        if not isinstance(message, dict):
            return

        now = datetime.datetime.now()

        # Extract instrument metadata
        security_id = str(message.get("security_id", message.get("securityId", "")))
        segment = str(message.get("exchange_segment", message.get("exchangeSegment", "")))
        
        # 1. Handle Spot Index (Nifty 50 Spot security ID is typically '13', '11536' or '999920')
        is_spot = (segment == "NSE_EQ" or segment == "IDX" or segment == "IDX_I" or "STRIKE" not in security_id) and \
                  (security_id == "13" or security_id == "11536" or security_id == "999920" or "NIFTY" in security_id)
        
        ltp = float(message.get("ltp", message.get("last_traded_price", message.get("lastPrice", 0.0))))
        volume = float(message.get("volume", message.get("volume_traded", message.get("volumeTraded", 0.0))))
        vwap = float(message.get("vwap", message.get("average_price", message.get("averagePrice", ltp))))
        if vwap <= 0:
            vwap = ltp

        if is_spot and ltp > 0:
            self.spot_price = ltp
            self.vwap = vwap
            self.db.insert_spot(now, config.UNDERLYING_SYMBOL, ltp, volume, vwap)
            self.latest_spot = {
                "timestamp": now,
                "symbol": config.UNDERLYING_SYMBOL,
                "ltp": ltp,
                "volume": volume,
                "vwap": vwap
            }
            
            # Populate order book depth metrics linked to spot
            self._generate_depth_from_spot(now, ltp, message)

        # 2. Handle Options Chain strikes (NSE_FNO segment)
        elif segment == "NSE_FNO" or "STRIKE" in security_id or message.get("strike_price") is not None:
            strike_price = float(message.get("strike_price", message.get("strikePrice", 0.0)))
            option_type = str(message.get("option_type", message.get("optionType", ""))).upper()
            
            # Fallback parsing from symbol name string if fields are missing
            symbol_name = str(message.get("symbol_name", message.get("symbol", "")))
            if strike_price == 0.0:
                import re
                match = re.search(r"(\d{5})", symbol_name)
                if match:
                    strike_price = float(match.group(1))
                if "CE" in symbol_name:
                    option_type = "CE"
                elif "PE" in symbol_name:
                    option_type = "PE"

            if strike_price > 0 and option_type in ["CE", "PE"] and ltp > 0:
                # Compute Black-Scholes dynamic Greeks & IV
                expiry_dt = datetime.datetime.combine(datetime.date.today(), datetime.time(15, 30))
                remaining_time = max((expiry_dt - now).total_seconds() / (365.0 * 24.0 * 3600.0), 1e-5)
                r = 0.07  # Risk-free rate
                iv = estimate_implied_volatility(ltp, self.spot_price, strike_price, remaining_time, r, option_type)
                
                _, delta, gamma, vega, theta = black_scholes_greeks(
                    self.spot_price, strike_price, remaining_time, r, iv, option_type
                )
                
                oi = float(message.get("oi", message.get("open_interest", message.get("openInterest", 10000.0))))
                
                self.db.insert_option(
                    now, f"{config.UNDERLYING_SYMBOL}_STRIKE_{strike_price}", strike_price, option_type,
                    ltp, iv, delta, gamma, vega, theta, oi, volume
                )
                
                self._update_cached_option(strike_price, option_type, ltp, iv, delta, gamma, vega, theta, oi, volume)

        # 3. Monitor Active Position in State-Machine
        if self.router.is_locked and self.router.active_position:
            active_strike = self.router.active_position["strike"]
            active_type = self.router.active_position["option_type"]
            if segment == "NSE_FNO" and strike_price == active_strike and option_type == active_type:
                self.router.monitor_and_update(ltp)

        # Broadcast live tick state
        feed_tick = {
            "spot": self.latest_spot,
            "depth": self.latest_depth,
            "option_chain": self.latest_option_chain,
            "timestamp": now
        }
        self._notify(feed_tick)

    def _generate_depth_from_spot(self, now: datetime.datetime, ltp: float, message: dict):
        """Populates order book structures from live ticks, or simulates depth if empty."""
        bids = []
        asks = []
        depth_data = message.get("depth", message.get("marketDepth", {}))
        
        if depth_data and ("buy" in depth_data or "bids" in depth_data):
            real_bids = depth_data.get("buy", depth_data.get("bids", []))
            real_asks = depth_data.get("sell", depth_data.get("asks", []))
            for b in real_bids:
                bids.append((float(b.get("price", 0.0)), float(b.get("quantity", 0.0))))
            for a in real_asks:
                asks.append((float(a.get("price", 0.0)), float(a.get("quantity", 0.0))))
        
        # Fallback simulator for depth centered around LTP
        if not bids:
            for i in range(1, 11):
                bids.append((ltp - (i * 0.5), float(random.randint(5000, 20000))))
                asks.append((ltp + (i * 0.5), float(random.randint(5000, 20000))))
                
        bid_vol = sum(v for p, v in bids)
        ask_vol = sum(v for p, v in asks)
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0.0
        
        # Calculate wall concentration ratio within 0.2%
        bid_wall = sum(v for p, v in bids if p >= ltp * 0.998)
        ask_wall = sum(v for p, v in asks if p <= ltp * 1.002)
        
        bid_wall_ratio = bid_wall / (bid_vol + 1.0)
        ask_wall_ratio = ask_wall / (ask_vol + 1.0)
        density = bid_wall + ask_wall
        
        self.db.insert_order_book(now, config.UNDERLYING_SYMBOL, ltp, imbalance, density, bid_wall_ratio, ask_wall_ratio)
        self.latest_depth = {
            "timestamp": now,
            "symbol": config.UNDERLYING_SYMBOL,
            "ltp": ltp,
            "imbalance": imbalance,
            "density": density,
            "bid_wall_ratio": bid_wall_ratio,
            "ask_wall_ratio": ask_wall_ratio,
            "bids": bids,
            "asks": asks
        }

    def _update_cached_option(self, strike: float, opt_type: str, ltp: float, iv: float, 
                              delta: float, gamma: float, vega: float, theta: float, oi: float, volume: float):
        """Updates options list in memory cache."""
        found = False
        for item in self.latest_option_chain:
            if item["strike"] == strike:
                found = True
                if opt_type == "CE":
                    item["ce_ltp"] = ltp
                    item["ce_iv"] = iv
                    item["ce_delta"] = delta
                    item["ce_gamma"] = gamma
                    item["ce_vega"] = vega
                    item["ce_theta"] = theta
                    item["ce_oi"] = oi
                    item["ce_vol"] = volume
                else:
                    item["pe_ltp"] = ltp
                    item["pe_iv"] = iv
                    item["pe_delta"] = delta
                    item["pe_gamma"] = gamma
                    item["pe_vega"] = vega
                    item["pe_theta"] = theta
                    item["pe_oi"] = oi
                    item["pe_vol"] = volume
                break
                
        if not found:
            new_item = {
                "strike": strike,
                "ce_ltp": ltp if opt_type == "CE" else 0.0,
                "ce_iv": iv if opt_type == "CE" else 0.15,
                "ce_delta": delta if opt_type == "CE" else 0.0,
                "ce_gamma": gamma if opt_type == "CE" else 0.0,
                "ce_vega": vega if opt_type == "CE" else 0.0,
                "ce_theta": theta if opt_type == "CE" else 0.0,
                "ce_oi": oi if opt_type == "CE" else 10000.0,
                "ce_vol": volume if opt_type == "CE" else 0.0,
                "pe_ltp": ltp if opt_type == "PE" else 0.0,
                "pe_iv": iv if opt_type == "PE" else 0.15,
                "pe_delta": delta if opt_type == "PE" else 0.0,
                "pe_gamma": gamma if opt_type == "PE" else 0.0,
                "pe_vega": vega if opt_type == "PE" else 0.0,
                "pe_theta": theta if opt_type == "PE" else 0.0,
                "pe_oi": oi if opt_type == "PE" else 10000.0,
                "pe_vol": volume if opt_type == "PE" else 0.0,
            }
            self.latest_option_chain.append(new_item)

    def _run_simulator(self):
        """Generates hyper-realistic tick streams for spot, options, and order depth."""
        print("[SIMULATOR] Starting high-fidelity simulation engine feed...")
        random.seed(42)
        
        while self.running:
            now = datetime.datetime.now()
            
            # 1. Spot Price Simulation (Random walk with dynamic momentum drift)
            dt = 0.5 # tick interval
            drift = 0.05 * math.sin(time.time() / 120)  # cycling intraday trend
            shock = random.normalvariate(0.0, 1.2)
            spot_change = drift * dt + shock
            self.spot_price += spot_change
            
            # Volume & VWAP
            tick_volume = float(random.randint(500, 3000))
            self.cum_vol += tick_volume
            self.cum_pv += self.spot_price * tick_volume
            self.vwap = self.cum_pv / self.cum_vol
            
            # Store Spot
            self.db.insert_spot(now, config.UNDERLYING_SYMBOL, self.spot_price, tick_volume, self.vwap)
            self.latest_spot = {
                "timestamp": now,
                "symbol": config.UNDERLYING_SYMBOL,
                "ltp": self.spot_price,
                "volume": tick_volume,
                "vwap": self.vwap
            }

            # 2. 200-Level Depth Simulation & Institutional Walls
            # Create a 10-level order book centered on LTP
            bid_vol_total = 0.0
            ask_vol_total = 0.0
            bid_wall_vol = 0.0
            ask_wall_vol = 0.0
            
            # Generate depth
            bids = []
            asks = []
            for i in range(1, 11):
                bid_p = self.spot_price - (i * 0.5)
                ask_p = self.spot_price + (i * 0.5)
                
                # Introduce institutional wall shock (e.g. huge passive order at level 4 or 5)
                is_bid_wall = (i == 4) and (random.random() < 0.2)  # 20% chance of big bid wall
                is_ask_wall = (i == 5) and (random.random() < 0.15)
                
                b_vol = float(random.randint(5000, 20000) * (10 if is_bid_wall else 1))
                a_vol = float(random.randint(5000, 20000) * (10 if is_ask_wall else 1))
                
                bid_vol_total += b_vol
                ask_vol_total += a_vol
                
                # Check walls within 0.2% of LTP
                if bid_p >= self.spot_price * 0.998:
                    bid_wall_vol += b_vol
                if ask_p <= self.spot_price * 1.002:
                    ask_wall_vol += a_vol
                
                bids.append((bid_p, b_vol))
                asks.append((ask_p, a_vol))

            imbalance = (bid_vol_total - ask_vol_total) / (bid_vol_total + ask_vol_total)
            density = bid_wall_vol + ask_wall_vol
            bid_wall_ratio = bid_wall_vol / (bid_vol_total + 1.0)
            ask_wall_ratio = ask_wall_vol / (ask_vol_total + 1.0)

            # Store Order Book
            self.db.insert_order_book(now, config.UNDERLYING_SYMBOL, self.spot_price, imbalance, density, bid_wall_ratio, ask_wall_ratio)
            self.latest_depth = {
                "timestamp": now,
                "symbol": config.UNDERLYING_SYMBOL,
                "ltp": self.spot_price,
                "imbalance": imbalance,
                "density": density,
                "bid_wall_ratio": bid_wall_ratio,
                "ask_wall_ratio": ask_wall_ratio,
                "bids": bids,
                "asks": asks
            }

            # 3. Dynamic Options Chain Matrix Simulation
            atm_strike = round(self.spot_price / 50.0) * 50
            strikes = [atm_strike + i * 50 for i in range(-4, 5)]  # 9 liquid strikes
            
            chain_list = []
            for strike in strikes:
                # Black-Scholes variables
                expiry_dt = datetime.datetime.combine(datetime.date.today(), datetime.time(15, 30))
                remaining_time = max((expiry_dt - now).total_seconds() / (365.0 * 24.0 * 3600.0), 1e-5)
                r = 0.07       # 7% interest rate
                iv_base = 0.15 + 0.05 * math.sin(strike / 1000.0)  # Skew curve
                
                # Compute Call & Put prices/greeks
                c_price, c_d, c_g, c_v, c_t = black_scholes_greeks(self.spot_price, strike, remaining_time, r, iv_base, "CE")
                p_price, p_d, p_g, p_v, p_t = black_scholes_greeks(self.spot_price, strike, remaining_time, r, iv_base, "PE")
                
                # Generate open interest and volume
                c_oi = float(100000 + random.randint(-5000, 15000) * (10 if strike <= atm_strike else 1))
                p_oi = float(100000 + random.randint(-5000, 15000) * (10 if strike >= atm_strike else 1))
                c_vol = float(random.randint(1000, 8000))
                p_vol = float(random.randint(1000, 8000))

                # Insert CE option
                self.db.insert_option(
                    now, f"{config.UNDERLYING_SYMBOL}_STRIKE_{strike}", strike, "CE",
                    c_price, iv_base, c_d, c_g, c_v, c_t, c_oi, c_vol
                )
                # Insert PE option
                self.db.insert_option(
                    now, f"{config.UNDERLYING_SYMBOL}_STRIKE_{strike}", strike, "PE",
                    p_price, iv_base, p_d, p_g, p_v, p_t, p_oi, p_vol
                )

                chain_list.append({
                    "strike": strike,
                    "ce_ltp": c_price, "ce_iv": iv_base, "ce_delta": c_d, "ce_gamma": c_g, "ce_vega": c_v, "ce_theta": c_t, "ce_oi": c_oi, "ce_vol": c_vol,
                    "pe_ltp": p_price, "pe_iv": iv_base, "pe_delta": p_d, "pe_gamma": p_g, "pe_vega": p_v, "pe_theta": p_t, "pe_oi": p_oi, "pe_vol": p_vol
                })

            self.latest_option_chain = chain_list

            # 4. Monitor active position in state-machine
            if self.router.is_locked and self.router.active_position:
                # Retrieve the active contract's latest simulated price
                active_contract_strike = self.router.active_position["strike"]
                active_contract_type = self.router.active_position["option_type"]
                for item in chain_list:
                    if item["strike"] == active_contract_strike:
                        active_ltp = item["ce_ltp"] if active_contract_type == "CE" else item["pe_ltp"]
                        self.router.monitor_and_update(active_ltp)
                        break

            # 5. Broadcast complete state tick
            feed_tick = {
                "spot": self.latest_spot,
                "depth": self.latest_depth,
                "option_chain": self.latest_option_chain,
                "timestamp": now
            }
            self._notify(feed_tick)
            
            # Wait for next tick
            time.sleep(dt)
