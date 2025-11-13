import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ICTSignal:
    """ICT Signal Data Class"""
    timestamp: pd.Timestamp
    direction: int  # 1 for long, -1 for short
    entry_price: float
    stop_loss: float
    take_profit: float
    confluence_score: int
    confidence: float
    mss_strength: float
    liquidity_distance: float

class ICTIndicators:
    """ICT Indicators Calculator"""

    @staticmethod
    def detect_market_structure_shift(high: pd.Series, low: pd.Series, close: pd.Series, 
                                    lookback: int = 5) -> pd.Series:
        """
        Detect Market Structure Shift (MSS)
        Returns: 1 for bullish MSS, -1 for bearish MSS, 0 for no shift
        """
        mss = pd.Series(0, index=close.index)

        for i in range(lookback, len(close)):
            recent_highs = high.iloc[i-lookback:i]
            recent_lows = low.iloc[i-lookback:i]

            # Bullish MSS: Break above recent lower high after making a lower low
            if (low.iloc[i] <= recent_lows.min() and 
                high.iloc[i] > recent_highs.max() and 
                close.iloc[i] > close.iloc[i-1]):
                mss.iloc[i] = 1

            # Bearish MSS: Break below recent higher low after making a higher high
            elif (high.iloc[i] >= recent_highs.max() and 
                  low.iloc[i] < recent_lows.min() and 
                  close.iloc[i] < close.iloc[i-1]):
                mss.iloc[i] = -1

        return mss

    @staticmethod
    def find_order_blocks(high: pd.Series, low: pd.Series, close: pd.Series, 
                         volume: pd.Series, open_price: pd.Series, 
                         min_volume_ratio: float = 1.2) -> Tuple[pd.Series, pd.Series]:
        """Find Order Blocks based on consolidation and breakout"""
        bullish_ob = pd.Series(False, index=close.index)
        bearish_ob = pd.Series(False, index=close.index)

        for i in range(10, len(close)):
            # Look for consolidation period (last 3-5 bars)
            consolidation_start = i-5
            consolidation_end = i-1

            consolidation_high = high.iloc[consolidation_start:consolidation_end].max()
            consolidation_low = low.iloc[consolidation_start:consolidation_end].min()
            consolidation_range_pct = (consolidation_high - consolidation_low) / consolidation_low

            # Check for tight consolidation (less than 0.3%)
            if consolidation_range_pct < 0.003:
                current_volume = volume.iloc[i]
                avg_volume = volume.iloc[consolidation_start:consolidation_end].mean()

                # Bullish Order Block: Breakout above consolidation with volume
                if (close.iloc[i] > consolidation_high and 
                    current_volume > avg_volume * min_volume_ratio and 
                    close.iloc[i] > open_price.iloc[i]):
                    bullish_ob.iloc[i] = True

                # Bearish Order Block: Breakout below consolidation with volume
                elif (close.iloc[i] < consolidation_low and 
                      current_volume > avg_volume * min_volume_ratio and 
                      close.iloc[i] < open_price.iloc[i]):
                    bearish_ob.iloc[i] = True

        return bullish_ob, bearish_ob

    @staticmethod
    def find_fair_value_gaps(high: pd.Series, low: pd.Series, close: pd.Series, 
                            open_price: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Find Fair Value Gaps (3-candle imbalance)"""
        bullish_fvg = pd.Series(False, index=close.index)
        bearish_fvg = pd.Series(False, index=close.index)

        for i in range(2, len(close)):
            # Bullish FVG: Current low > previous high
            if low.iloc[i] > high.iloc[i-1]:
                bullish_fvg.iloc[i] = True

            # Bearish FVG: Current high < previous low
            elif high.iloc[i] < low.iloc[i-1]:
                bearish_fvg.iloc[i] = True

        return bullish_fvg, bearish_fvg

    @staticmethod
    def calculate_liquidity_levels(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Buy-side and Sell-side liquidity levels"""
        bsl = low.rolling(window=period).min()
        ssl = high.rolling(window=period).max()
        return bsl, ssl

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def detect_kill_zones(timestamps: pd.DatetimeIndex) -> pd.Series:
        """Detect ICT Kill Zones based on time"""
        kill_zone = pd.Series(False, index=timestamps)

        for i, ts in enumerate(timestamps):
            hour = ts.hour
            # London Kill Zone: 2:00-5:00 AM EST
            # New York Kill Zone: 8:00-11:00 AM EST
            if (2 <= hour <= 5) or (8 <= hour <= 11):
                kill_zone.iloc[i] = True

        return kill_zone

    @staticmethod
    def calculate_silver_bullet(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Silver Bullet setup (shallow 50% retracement)"""
        silver_bullet = pd.Series(False, index=close.index)

        for i in range(3, len(close)):
            # Look for strong move then shallow retracement
            move_size = abs(close.iloc[i-2] - close.iloc[i-1]) / close.iloc[i-2]
            retracement = abs(close.iloc[i] - close.iloc[i-1]) / move_size if move_size > 0 else 0

            if move_size > 0.002 and 0.4 <= retracement <= 0.6:  # 0.2% move with 40-60% retrace
                silver_bullet.iloc[i] = True

        return silver_bullet
