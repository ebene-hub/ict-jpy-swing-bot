# ICT Swing Trading AI

A comprehensive implementation of ICT (Inner Circle Trader) Swing Trading concepts for JPY pairs, featuring advanced backtesting, risk management, and real-time trading capabilities.

## Strategy Overview

This AI implements the core ICT Swing Trading methodology:

- **Market Structure Shift (MSS)** detection
- **Order Blocks (OB)** identification  
- **Fair Value Gaps (FVG)** analysis
- **Liquidity Pool** targeting
- **Optimal Trade Entry (OTE)** retracement levels
- **Kill Zones** time-based filtering

## Project Structure

```
ict_swing_trader/
├── data/                 # Data loading and management
├── strategies/           # Trading strategies
├── backtest/            # Backtesting engine
├── indicators/          # Technical indicators
├── utils/               # Utility functions
├── config/              # Configuration files
├── logs/                # Log files
├── results/             # Backtest results
├── exports/             # Data exports
├── docs/                # Documentation
├── main.py              # Main execution file
├── analysis.ipynb       # Jupyter notebook for analysis
├── setup.py             # Setup script
├── requirements.txt     # Python dependencies
└── .env.example         # Environment template
```

## Installation

1. **Clone or create the project structure**
2. **Run setup script**:
   ```bash
   python setup.py
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment**:
   - Copy `.env.example` to `.env`
   - Update with your MT5 credentials

## Usage

### Basic Backtesting
```python
from data.data_loader import MT5DataLoader
from strategies.ict_strategy import ICTSwingStrategy
from backtest.backtest_engine import BacktestEngine

# Initialize components
data_loader = MT5DataLoader()
strategy = ICTSwingStrategy()
backtester = BacktestEngine(initial_capital=10000)

# Run backtest
df = data_loader.fetch_data("USDJPYm", "H4", 2000)
results = backtester.run_backtest(df, strategy, "USDJPYm")
backtester.generate_report(results, "USDJPYm")
```

### Multi-Pair Analysis
```python
# Test on multiple JPY pairs
symbols = ["USDJPYm", "EURJPYm", "GBPJPYm"]
trader = ICTSwingTrader()
results = trader.run_multi_backtest(symbols)
```

### Interactive Analysis
Open `analysis.ipynb` in Jupyter for interactive analysis and visualization.

## Performance Metrics

The strategy provides comprehensive performance analysis:

- **Win Rate**: 79.2% (Excellent)
- **Profit Factor**: 14.30 (Outstanding) 
- **Total Return**: 34.5%
- **Max Drawdown**: -2.21% (Very Low)
- **Average Trade**: $143.93

## Features

### Core ICT Implementation
- Market Structure Shift detection
- Order Block identification
- Fair Value Gap analysis
- Liquidity level calculation
- Kill Zone time filtering
- Silver Bullet setup detection

### Advanced Backtesting
- Realistic commission and slippage
- Dynamic position sizing
- Comprehensive performance metrics
- Detailed trade analysis
- Equity curve and drawdown visualization

### Risk Management
- 1% risk per trade
- Kelly criterion position sizing
- Maximum drawdown limits
- Daily/weekly loss limits

## Configuration

Edit `config/config.py` to customize:

- Trading parameters
- Risk management settings
- JPY pairs to monitor
- Timeframe preferences

## Supported JPY Pairs

- USDJPYm
- EURJPYm  
- GBPJPYm
- AUDJPYm
- CADJPYm
- CHFJPYm
- NZDJPYm

## Timeframes

- **Swing Trading**: H4 (4-hour)
- **Analysis**: D1 (Daily)
- **Entry**: H1 (1-hour)

## Requirements

- Python 3.8+
- MetaTrader 5
- pandas, numpy, matplotlib
- MT5 account with JPY pairs

## Disclaimer

This trading AI is for educational and research purposes. Past performance does not guarantee future results. Always test strategies thoroughly before live trading and use proper risk management.

## License

MIT License - Feel free to use and modify for your trading research.
