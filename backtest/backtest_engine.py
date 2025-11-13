import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BacktestResult:
    """Backtest Result Data Class"""

    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.total_pnl = 0.0
        self.final_equity = 0.0
        self.profit_factor = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.avg_trade = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.calmar_ratio = 0.0
        self.long_short_ratio = 0.0
        self.avg_holding_period = timedelta(0)

class BacktestEngine:
    """Advanced Backtesting Engine for ICT Strategy"""

    def __init__(self, initial_capital: float = 10000, commission: float = 0.0002, slippage: float = 0.0001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results = BacktestResult()
        logger.info(f"Backtest engine initialized with ${initial_capital:,.2f} capital")

    def run_backtest(self, df: pd.DataFrame, strategy, symbol: str = "Unknown") -> Dict:
        """Run comprehensive backtest"""
        df_with_signals = strategy.generate_signals(df, symbol)

        # Initialize tracking
        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0
        entry_time = None
        trades = []
        equity_curve = []
        drawdown_curve = []

        for i, (timestamp, row) in enumerate(df_with_signals.iterrows()):
            current_price = row['close']

            # Apply slippage and commission for realistic execution
            execution_price = self._apply_slippage(current_price, row['signal'])

            # Check for exit conditions
            if position != 0:
                exit_signal, exit_price, exit_reason = self._check_exit_conditions(
                    position, entry_price, execution_price, row
                )

                if exit_signal:
                    pnl = self._calculate_pnl(position, entry_price, exit_price)
                    pnl_after_commission = pnl - abs(pnl) * self.commission

                    trades.append({
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'direction': 'LONG' if position > 0 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl_after_commission,
                        'exit_reason': exit_reason,
                        'holding_period': timestamp - entry_time,
                        'capital_before': capital,
                        'capital_after': capital + pnl_after_commission
                    })

                    capital += pnl_after_commission
                    position = 0.0
                    entry_time = None

            # Check for entry conditions
            if position == 0 and row['signal'] != 0 and not np.isnan(row['entry_price']):
                position_size = self._calculate_position_size(capital, row)

                if position_size > 0:
                    if row['signal'] == 1:  # LONG
                        position = position_size
                        entry_price = execution_price
                        entry_time = timestamp
                    elif row['signal'] == -1:  # SHORT
                        position = -position_size
                        entry_price = execution_price
                        entry_time = timestamp

            # Update equity curve
            unrealized_pnl = 0
            if position != 0:
                if position > 0:  # LONG
                    unrealized_pnl = (execution_price - entry_price) * position
                else:  # SHORT
                    unrealized_pnl = (entry_price - execution_price) * abs(position)

            total_equity = capital + unrealized_pnl
            equity_curve.append(total_equity)

        # Calculate performance metrics
        self._calculate_performance_metrics(trades, equity_curve, df_with_signals)

        return {
            'trades': pd.DataFrame(trades),
            'equity_curve': equity_curve,
            'drawdown_curve': drawdown_curve,
            'df': df_with_signals,
            'results': self.results
        }

    def _apply_slippage(self, price: float, signal: int) -> float:
        """Apply slippage to execution price"""
        if signal == 1:  # LONG - buy at slightly higher price
            return price * (1 + self.slippage)
        elif signal == -1:  # SHORT - sell at slightly lower price
            return price * (1 - self.slippage)
        else:
            return price

    def _check_exit_conditions(self, position: float, entry_price: float, current_price: float, 
                             row: pd.Series) -> Tuple[bool, float, str]:
        """Check if exit conditions are met"""
        if position > 0:  # LONG position
            if current_price <= row['stop_loss']:
                return True, row['stop_loss'], 'SL'
            elif current_price >= row['take_profit']:
                return True, row['take_profit'], 'TP'
        else:  # SHORT position
            if current_price >= row['stop_loss']:
                return True, row['stop_loss'], 'SL'
            elif current_price <= row['take_profit']:
                return True, row['take_profit'], 'TP'

        return False, 0.0, ''

    def _calculate_pnl(self, position: float, entry_price: float, exit_price: float) -> float:
        """Calculate P&L for a trade"""
        if position > 0:  # LONG
            return (exit_price - entry_price) * position
        else:  # SHORT
            return (entry_price - exit_price) * abs(position)

    def _calculate_position_size(self, capital: float, row: pd.Series) -> float:
        """Calculate position size based on risk management"""
        risk_per_trade = capital * 0.01  # 1% risk per trade

        if row['signal'] == 1:  # LONG
            risk_per_share = row['entry_price'] - row['stop_loss']
        else:  # SHORT
            risk_per_share = row['stop_loss'] - row['entry_price']

        if risk_per_share > 0:
            position_size = risk_per_trade / risk_per_share
            # Limit position size to 10% of capital
            max_position = (capital * 0.1) / row['entry_price']
            return min(position_size, max_position)

        return 0.0

    def _calculate_performance_metrics(self, trades: List[Dict], equity_curve: List[float], df: pd.DataFrame):
        """Calculate comprehensive performance metrics"""
        if not trades:
            logger.warning("No trades executed in backtest")
            return

        trades_df = pd.DataFrame(trades)
        self.results.total_trades = len(trades_df)
        self.results.winning_trades = len(trades_df[trades_df['pnl'] > 0])
        self.results.losing_trades = len(trades_df[trades_df['pnl'] < 0])

        self.results.win_rate = self.results.winning_trades / self.results.total_trades
        self.results.total_pnl = trades_df['pnl'].sum()
        self.results.final_equity = equity_curve[-1] if equity_curve else self.initial_capital

        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]

        self.results.avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        self.results.avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

        if self.results.avg_loss != 0:
            self.results.profit_factor = abs(self.results.avg_win * self.results.winning_trades) / \
                                       abs(self.results.avg_loss * self.results.losing_trades)
        else:
            self.results.profit_factor = float('inf')

        self.results.avg_trade = self.results.total_pnl / self.results.total_trades

        # Calculate drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        self.results.max_drawdown = drawdown.min() * 100

        # Calculate holding period
        if 'holding_period' in trades_df.columns:
            self.results.avg_holding_period = trades_df['holding_period'].mean()

        # Long/Short ratio
        long_trades = trades_df[trades_df['direction'] == 'LONG']
        short_trades = trades_df[trades_df['direction'] == 'SHORT']
        self.results.long_short_ratio = len(long_trades) / len(short_trades) if len(short_trades) > 0 else float('inf')

    def generate_report(self, results: Dict, symbol: str = "Unknown"):
        """Generate comprehensive backtest report"""
        trades_df = results['trades']
        df = results['df']

        print(f"\n{'='*60}")
        print(f"BACKTEST REPORT: {symbol}")
        print(f"{'='*60}")
        print(f"Total Trades: {self.results.total_trades}")
        print(f"Winning Trades: {self.results.winning_trades}")
        print(f"Losing Trades: {self.results.losing_trades}")
        print(f"Win Rate: {self.results.win_rate:.2%}")
        print(f"Total P&L: ${self.results.total_pnl:,.2f}")
        print(f"Final Equity: ${self.results.final_equity:,.2f}")
        print(f"Profit Factor: {self.results.profit_factor:.2f}")
        print(f"Average Win: ${self.results.avg_win:.2f}")
        print(f"Average Loss: ${self.results.avg_loss:.2f}")
        print(f"Average Trade: ${self.results.avg_trade:.2f}")
        print(f"Max Drawdown: {self.results.max_drawdown:.2f}%")
        print(f"Long/Short Ratio: {self.results.long_short_ratio:.2f}")

        if hasattr(self.results.avg_holding_period, 'total_seconds'):
            hours = self.results.avg_holding_period.total_seconds() / 3600
            print(f"Average Holding Period: {hours:.1f} hours")

    def plot_results(self, results: Dict, symbol: str = "Unknown"):
        """Plot comprehensive backtest results"""
        trades_df = results['trades']
        df = results['df']
        equity_curve = results['equity_curve']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'ICT Swing Trading Results - {symbol}', fontsize=16, fontweight='bold')

        # Price chart with signals
        axes[0, 0].plot(df.index, df['close'], label='Close', alpha=0.7, linewidth=1)
        axes[0, 0].plot(df.index, df['bsl'], label='BSL', alpha=0.5, linestyle='--', color='green')
        axes[0, 0].plot(df.index, df['ssl'], label='SSL', alpha=0.5, linestyle='--', color='red')

        long_entries = df[df['signal'] == 1]
        short_entries = df[df['signal'] == -1]

        if len(long_entries) > 0:
            axes[0, 0].scatter(long_entries.index, long_entries['close'], 
                             color='green', marker='^', s=50, label='Long', alpha=0.7)
        if len(short_entries) > 0:
            axes[0, 0].scatter(short_entries.index, short_entries['close'], 
                             color='red', marker='v', s=50, label='Short', alpha=0.7)

        axes[0, 0].set_title('Price with ICT Signals')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Equity curve
        axes[0, 1].plot(df.index, equity_curve, linewidth=2, color='blue', label='Equity')
        axes[0, 1].axhline(y=self.initial_capital, color='red', linestyle='--', label='Initial Capital')
        axes[0, 1].set_title('Equity Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100

        axes[1, 0].fill_between(df.index, drawdown, 0, alpha=0.3, color='red')
        axes[1, 0].plot(df.index, drawdown, color='red', linewidth=1)
        axes[1, 0].set_title('Drawdown (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # P&L distribution
        if len(trades_df) > 0:
            axes[1, 1].hist(trades_df['pnl'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[1, 1].set_title('P&L Distribution per Trade')
            axes[1, 1].set_xlabel('P&L ($)')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
