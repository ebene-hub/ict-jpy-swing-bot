# config/config.py - UPDATED FOR DEMO TRADING

# MT5 Configuration for DEMO Account
MT5 = {
    "server": "Exness-MT5Trial9",  # Your demo server
    "login": 297034992,            # Your demo account number
    "password": "Testbot@1344",    # Your demo password
    "timeout": 10000,
    "portable": False
}

# DEMO Trading Parameters (More Conservative)
TRADING = {
    "initial_capital": 1000,       # Start with $1000 on demo
    "risk_per_trade": 0.005,       # 0.5% risk for demo
    "max_drawdown": 0.02,          # 2% max drawdown
    "commission": 0.0002,          # Broker commission
    "slippage": 0.0001,            # Execution slippage
    "max_positions": 2,            # Max 2 simultaneous trades
    "daily_loss_limit": 0.01       # 1% daily loss limit
}

# # LIVE Trading Parameters (MORE CONSERVATIVE)
# TRADING = {
#     "initial_capital": 500,      # Start SMALL with real money
#     "risk_per_trade": 0.002,     # 0.2% risk - VERY CONSERVATIVE
#     "max_drawdown": 0.01,        # 1% max drawdown
#     "commission": 0.0002,
#     "slippage": 0.0001,
#     "max_positions": 1,          # Only 1 trade at a time
#     "daily_loss_limit": 0.005    # 0.5% daily loss limit
# }


# Strategy Parameters (WINNING Simple Optimized)
STRATEGY = {
    "mss_lookback": 5,
    "liquidity_period": 20,
    "risk_reward_ratio": 2.0,
    "atr_period": 14,
    "atr_multiplier": 1.5,
    "min_volume_ratio": 1.2,
    "min_confluence": 4,           # Optimized for quality
    "use_kill_zones": True,
    "use_silver_bullet": False
}

# Pairs to Trade (DEMO - Start with fewer pairs)
PAIRS = []# Start with 3 major pairs


# Timeframes
TIMEFRAMES = {
    "swing": "H4",
    "analysis": "D1", 
    "entry": "H1"
}

# Risk Management
RISK_MANAGEMENT = {
    "daily_loss_limit": 0.01,      # 1% daily loss limit
    "weekly_loss_limit": 0.03,     # 3% weekly loss limit  
    "position_sizing": "kelly"     # kelly, fixed, or martingale
}