"""
Utility functions for ICT Swing Trader
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    """Risk Management utilities"""

    @staticmethod
    def calculate_kelly_position_size(win_rate: float, win_loss_ratio: float, capital: float) -> float:
        """Calculate position size using Kelly Criterion"""
        if win_loss_ratio <= 0:
            return 0.0

        kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
        # Use half-kelly for conservative approach
        return max(0.0, (kelly_fraction * 0.5) * capital)

    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return returns.quantile(1 - confidence)

    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        return drawdown.min() * 100

class PerformanceAnalyzer:
    """Performance analysis utilities"""

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    @staticmethod
    def calculate_calmar_ratio(total_return: float, max_drawdown: float, years: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return float('inf')
        annual_return = total_return / years
        return annual_return / abs(max_drawdown)

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()

        if downside_std == 0:
            return float('inf')

        return excess_returns.mean() / downside_std * np.sqrt(252)

    @staticmethod
    def generate_performance_report(trades_df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if trades_df.empty:
            return {}

        report = {}

        # Basic metrics
        report['total_trades'] = len(trades_df)
        report['winning_trades'] = len(trades_df[trades_df['pnl'] > 0])
        report['losing_trades'] = len(trades_df[trades_df['pnl'] < 0])
        report['win_rate'] = report['winning_trades'] / report['total_trades']
        report['total_pnl'] = trades_df['pnl'].sum()
        report['final_capital'] = initial_capital + report['total_pnl']

        # Advanced metrics
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]

        report['avg_win'] = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        report['avg_loss'] = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        report['largest_win'] = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0
        report['largest_loss'] = losing_trades['pnl'].min() if len(losing_trades) > 0 else 0

        if report['avg_loss'] != 0:
            report['profit_factor'] = abs(report['avg_win'] * report['winning_trades']) / \
                                    abs(report['avg_loss'] * report['losing_trades'])
        else:
            report['profit_factor'] = float('inf')

        return report

class DataValidator:
    """Data validation utilities"""

    @staticmethod
    def validate_price_data(df: pd.DataFrame) -> bool:
        """Validate price data integrity"""
        required_columns = ['open', 'high', 'low', 'close', 'tick_volume']

        if not all(col in df.columns for col in required_columns):
            logger.error("Missing required price columns")
            return False

        # Check for NaN values
        if df[required_columns].isnull().any().any():
            logger.warning("Price data contains NaN values")
            return False

        # Check for negative prices
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            logger.error("Price data contains non-positive values")
            return False

        # Check high >= low
        if (df['high'] < df['low']).any():
            logger.error("High prices less than low prices detected")
            return False

        # Check high >= open, close and low <= open, close
        if (df['high'] < df[['open', 'close']].max(axis=1)).any() or \
           (df['low'] > df[['open', 'close']].min(axis=1)).any():
            logger.error("Price consistency check failed")
            return False

        return True

    @staticmethod
    def detect_data_gaps(df: pd.DataFrame, timeframe: str) -> List[pd.Timestamp]:
        """Detect gaps in time series data"""
        if 'time' not in df.columns and df.index.name != 'time':
            logger.error("Time column not found")
            return []

        # Calculate expected frequency
        freq_map = {
            'H1': '1H',
            'H4': '4H', 
            'D1': '1D'
        }

        expected_freq = freq_map.get(timeframe, '1H')

        # Create complete time index
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
        missing_times = full_range.difference(df.index)

        return list(missing_times)

class Plotter:
    """Advanced plotting utilities"""

    @staticmethod
    def plot_ict_components(df: pd.DataFrame, symbol: str):
        """Plot all ICT components together"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'ICT Analysis - {symbol}', fontsize=16, fontweight='bold')

        # Price with MSS
        axes[0, 0].plot(df.index, df['close'], label='Close', linewidth=1)
        mss_bullish = df[df['mss'] == 1]
        mss_bearish = df[df['mss'] == -1]

        axes[0, 0].scatter(mss_bullish.index, mss_bullish['close'], 
                          color='green', marker='^', s=30, label='Bullish MSS')
        axes[0, 0].scatter(mss_bearish.index, mss_bearish['close'],
                          color='red', marker='v', s=30, label='Bearish MSS')
        axes[0, 0].set_title('Market Structure Shifts')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Order Blocks
        axes[0, 1].plot(df.index, df['close'], label='Close', linewidth=1, alpha=0.7)
        bullish_ob = df[df['bullish_ob']]
        bearish_ob = df[df['bearish_ob']]

        axes[0, 1].scatter(bullish_ob.index, bullish_ob['close'],
                          color='lime', marker='^', s=50, label='Bullish OB')
        axes[0, 1].scatter(bearish_ob.index, bearish_ob['close'],
                          color='orange', marker='v', s=50, label='Bearish OB')
        axes[0, 1].set_title('Order Blocks')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Fair Value Gaps
        axes[1, 0].plot(df.index, df['close'], label='Close', linewidth=1, alpha=0.7)
        bullish_fvg = df[df['bullish_fvg']]
        bearish_fvg = df[df['bearish_fvg']]

        axes[1, 0].scatter(bullish_fvg.index, bullish_fvg['close'],
                          color='cyan', marker='^', s=40, label='Bullish FVG')
        axes[1, 0].scatter(bearish_fvg.index, bearish_fvg['close'],
                          color='magenta', marker='v', s=40, label='Bearish FVG')
        axes[1, 0].set_title('Fair Value Gaps')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Liquidity Levels
        axes[1, 1].plot(df.index, df['close'], label='Close', linewidth=1)
        axes[1, 1].plot(df.index, df['bsl'], label='BSL', linestyle='--', color='green', alpha=0.7)
        axes[1, 1].plot(df.index, df['ssl'], label='SSL', linestyle='--', color='red', alpha=0.7)
        axes[1, 1].set_title('Liquidity Levels')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Confluence Score
        if 'confluence_score' in df.columns:
            signals = df[df['signal'] != 0]
            axes[2, 0].bar(signals.index, signals['confluence_score'], 
                          color=['green' if s > 0 else 'red' for s in signals['signal']],
                          alpha=0.7)
            axes[2, 0].set_title('Signal Confluence Score')
            axes[2, 0].grid(True, alpha=0.3)

        # Confidence
        if 'confidence' in df.columns:
            signals = df[df['signal'] != 0]
            axes[2, 1].bar(signals.index, signals['confidence'],
                          color=['green' if s > 0 else 'red' for s in signals['signal']],
                          alpha=0.7)
            axes[2, 1].set_title('Signal Confidence')
            axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
