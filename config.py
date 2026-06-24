import os

# ==============================================================================
# ENGINE OPERATIONAL MODE
# ==============================================================================
# Options: "SIMULATION" (uses realistic mock feeds) or "LIVE" (uses real DhanHQ API)
RUN_MODE = "SIMULATION"

# ==============================================================================
# DHANHQ API CREDENTIALS (Required if RUN_MODE == "LIVE")
# ==============================================================================
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "MOCK_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "MOCK_ACCESS_TOKEN")

# ==============================================================================
# TRADING SYSTEM CONFIGURATION
# ==============================================================================
UNDERLYING_SYMBOL = "NIFTY-50"
LOT_SIZE = 50                 # Contract lot size for NIFTY
TRADE_LOTS = 5                 # Fixed size: 5 lots = 250 contracts
RISK_REWARD_RATIO = 3.0        # 1:3 Risk to Reward target exit
STOP_LOSS_PCT = 0.10          # 10% stop loss on premium
TARGET_PCT = 0.30             # 30% take profit target (R:R 1:3)

# Strict daily time window enforcement (avoiding opening/closing volatility)
TRADING_WINDOW_START = "09:45:00"  # 30 minutes after open (09:15:00)
TRADING_WINDOW_END = "15:00:00"    # 30 minutes before close (15:30:00)

# ==============================================================================
# DATABASE & MACHINE LEARNING CONFIGURATIONS
# ==============================================================================
# Base directory for storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DUCKDB_PATH = os.path.join(BASE_DIR, "options_trading.db")
# Separate DB for the backtester so live feed data never pollutes backtest ticks
BACKTEST_DB_PATH = os.path.join(BASE_DIR, "backtest.db")
MODEL_PATH = os.path.join(BASE_DIR, "model_weights.pkl")

# ML Feature list matching the institutional edges
FEATURE_COLS = [
    "book_imbalance",
    "order_book_density",
    "bid_wall_ratio",
    "ask_wall_ratio",
    "iv_skew_slope",
    "iv_percentile",
    "pcr_divergence",
    "oi_velocity_ratio",
    "vwap_distance",
    "volume_velocity",
    "momentum_lag_1",
    "momentum_lag_5"
]

# Hyperparameters for the Gradient Boosting predictive core
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "random_state": 42
}
