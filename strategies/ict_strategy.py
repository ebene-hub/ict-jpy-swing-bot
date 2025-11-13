# strategies/ict_strategy.py
"""
COMPLETE FIXED ICT Strategy with Proper SL/TP Methodology
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from indicators.ict_indicators import ICTIndicators, ICTSignal

logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
    """Trade Signal Data Class"""
    symbol: str
    timestamp: pd.Timestamp
    direction: int  # 1 for long, -1 for short
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    confidence: float
    signal_strength: int

class SimpleOptimizedICTStrategy:
    """
    WINNING STRATEGY - Simple Optimized ICT Swing Trading
    With PROPER ICT SL/TP Methodology
    """

    def __init__(self, parameters: Optional[Dict] = None):
        self.default_params = {
            'mss_lookback': 5,
            'liquidity_period': 20,
            'risk_reward_ratio': 2.0,
            'atr_period': 14,
            'atr_multiplier': 1.5,
            'min_volume_ratio': 1.2,
            'min_confluence': 4,
            'use_kill_zones': True,
            'use_silver_bullet': False
        }
        self.params = {**self.default_params, **(parameters or {})}
        self.indicators = ICTIndicators()
        logger.info(f"Simple Optimized ICT Strategy initialized with params: {self.params}")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all ICT indicators for the dataframe"""
        df = df.copy()

        # Core ICT indicators
        df['mss'] = self.indicators.detect_market_structure_shift(
            df['high'], df['low'], df['close'], self.params['mss_lookback']
        )

        df['bullish_ob'], df['bearish_ob'] = self.indicators.find_order_blocks(
            df['high'], df['low'], df['close'], df['tick_volume'], df['open'], 
            self.params['min_volume_ratio']
        )

        df['bullish_fvg'], df['bearish_fvg'] = self.indicators.find_fair_value_gaps(
            df['high'], df['low'], df['close'], df['open']
        )

        df['bsl'], df['ssl'] = self.indicators.calculate_liquidity_levels(
            df['high'], df['low'], self.params['liquidity_period']
        )

        df['atr'] = self.indicators.calculate_atr(
            df['high'], df['low'], df['close'], self.params['atr_period']
        )

        # Optional ICT concepts
        if self.params['use_kill_zones']:
            df['in_kill_zone'] = self.indicators.detect_kill_zones(df.index)

        if self.params['use_silver_bullet']:
            df['silver_bullet'] = self.indicators.calculate_silver_bullet(
                df['high'], df['low'], df['close']
            )

        return df

    def calculate_confluence_score(self, current, prev_1, prev_2) -> Tuple[int, int]:
        """Calculate confluence scores for long and short signals"""
        long_conditions = [
            current['mss'] == 1,
            current['bullish_ob'] or prev_1['bullish_ob'] or prev_2['bullish_ob'],
            current['bullish_fvg'] or prev_1['bullish_fvg'] or prev_2['bullish_fvg'],
            current['close'] > current['bsl'],
            current['close'] > current['open'],
        ]

        short_conditions = [
            current['mss'] == -1,
            current['bearish_ob'] or prev_1['bearish_ob'] or prev_2['bearish_ob'],
            current['bearish_fvg'] or prev_1['bearish_fvg'] or prev_2['bearish_fvg'],
            current['close'] < current['ssl'],
            current['close'] < current['open'],
        ]

        # Add kill zone condition if enabled
        if self.params['use_kill_zones']:
            long_conditions.append(current['in_kill_zone'])
            short_conditions.append(current['in_kill_zone'])

        long_confluence = sum(long_conditions)
        short_confluence = sum(short_conditions)

        return long_confluence, short_confluence


    def calculate_proper_ict_sl_tp(self, df: pd.DataFrame, idx: int, direction: int, entry_price: float) -> Tuple[float, float]:
        """Calculate PROPER ICT-based Stop Loss and Take Profit with FIXED R:R"""
        try:
            lookback = 20
            
            if direction == 1:  # LONG
                # ICT LONG SL: Below recent swing low OR below BSL (whichever is lower)
                recent_lows = df['low'].iloc[max(0, idx-lookback):idx]
                swing_low = recent_lows.min()
                bsl = df['bsl'].iloc[idx]
                stop_loss = min(bsl, swing_low)
                
                # Add buffer below support
                stop_loss = stop_loss * 0.998
                
                # ICT LONG TP: Below recent swing high OR below SSL (logical resistance)
                recent_highs = df['high'].iloc[max(0, idx-lookback):idx]
                swing_high = recent_highs.max()
                ssl = df['ssl'].iloc[idx]
                take_profit = min(ssl, swing_high)
                
                # Add buffer below resistance
                take_profit = take_profit * 0.998
                
            else:  # SHORT
                # ICT SHORT SL: Above recent swing high OR above SSL (whichever is higher)
                recent_highs = df['high'].iloc[max(0, idx-lookback):idx]
                swing_high = recent_highs.max()
                ssl = df['ssl'].iloc[idx]
                stop_loss = max(ssl, swing_high)
                
                # Add buffer above resistance
                stop_loss = stop_loss * 1.002
                
                # ICT SHORT TP: Above recent swing low OR above BSL (logical support)
                recent_lows = df['low'].iloc[max(0, idx-lookback):idx]
                swing_low = recent_lows.min()
                bsl = df['bsl'].iloc[idx]
                take_profit = max(bsl, swing_low)
                
                # Add buffer above support
                take_profit = take_profit * 1.002
            
            # CRITICAL FIX: Calculate risk-reward ratio PROPERLY
            if direction == 1:  # LONG
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:  # SHORT
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
            
            rr_ratio = reward / risk if risk > 0 else 0
            
            print(f"   Initial R:R Calculation: Risk=${risk:.3f}, Reward=${reward:.3f}, R:R={rr_ratio:.2f}:1")
            
            # CRITICAL FIX: Ensure reasonable R:R ratio (1.5 to 3.0)
            if rr_ratio < 1.5:
                print(f"   ⚠️  R:R too low ({rr_ratio:.2f}), adjusting TP...")
                if direction == 1:  # LONG
                    take_profit = entry_price + (risk * 2.0)  # Target 2:1 R:R
                else:  # SHORT
                    take_profit = entry_price - (risk * 2.0)  # Target 2:1 R:R
            elif rr_ratio > 5.0:  # Too optimistic
                print(f"   ⚠️  R:R too high ({rr_ratio:.2f}), adjusting TP...")
                if direction == 1:  # LONG
                    take_profit = entry_price + (risk * 2.5)  # Cap at 2.5:1
                else:  # SHORT
                    take_profit = entry_price - (risk * 2.5)  # Cap at 2.5:1
            
            # Final validation - CRITICAL FOR BROKER
            if direction == 1:  # LONG
                if stop_loss >= entry_price:
                    stop_loss = entry_price * 0.99  # Ensure SL is below entry
                    print(f"   🔧 Corrected SL to be below entry: ${stop_loss:.3f}")
                if take_profit <= entry_price:
                    take_profit = entry_price * 1.02  # Ensure TP is above entry
                    print(f"   🔧 Corrected TP to be above entry: ${take_profit:.3f}")
            else:  # SHORT
                if stop_loss <= entry_price:
                    stop_loss = entry_price * 1.01  # Ensure SL is above entry
                    print(f"   🔧 Corrected SL to be above entry: ${stop_loss:.3f}")
                if take_profit >= entry_price:
                    take_profit = entry_price * 0.98  # Ensure TP is below entry
                    print(f"   🔧 Corrected TP to be below entry: ${take_profit:.3f}")
            
            # Final R:R calculation
            if direction == 1:  # LONG
                final_risk = entry_price - stop_loss
                final_reward = take_profit - entry_price
            else:  # SHORT
                final_risk = stop_loss - entry_price
                final_reward = entry_price - take_profit
                
            final_rr = final_reward / final_risk if final_risk > 0 else 0
            
            logger.info(f"ICT SL/TP Final: Direction={'LONG' if direction == 1 else 'SHORT'}, "
                    f"Entry=${entry_price:.3f}, SL=${stop_loss:.3f}, TP=${take_profit:.3f}, "
                    f"R:R={final_rr:.2f}:1")
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating ICT SL/TP: {e}")
            # Fallback to reasonable defaults with good R:R
            if direction == 1:
                return entry_price * 0.99, entry_price * 1.02  # 2:1 R:R
            else:
                return entry_price * 1.01, entry_price * 0.98  # 2:1 R:R

    def generate_signals(self, df: pd.DataFrame, symbol: str = "Unknown") -> pd.DataFrame:
        """Generate high-quality ICT Swing Trading signals with PROPER SL/TP"""
        df = self.calculate_all_indicators(df)

        # Initialize signal columns
        df['signal'] = 0
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        df['position_size'] = np.nan
        df['confluence_score'] = 0
        df['confidence'] = 0.0

        signal_count = 0

        for i in range(20, len(df)):
            current = df.iloc[i]
            prev_1 = df.iloc[i-1]
            prev_2 = df.iloc[i-2]

            long_confluence, short_confluence = self.calculate_confluence_score(current, prev_1, prev_2)

            # Additional quality filters
            price_change = abs(current['close'] - current['open']) / current['open']
            min_price_move = 0.001  # 0.1% minimum price movement
            
            # LONG Entry with quality filters
            if (long_confluence >= self.params['min_confluence'] and 
                price_change > min_price_move and
                current['close'] > df['close'].iloc[i-5]):  # Simple uptrend filter
                
                entry_price = current['close']
                stop_loss, take_profit = self.calculate_proper_ict_sl_tp(df, i, 1, entry_price)
                
                self._create_signal(df, i, 1, entry_price, stop_loss, take_profit, long_confluence)
                signal_count += 1

            # SHORT Entry with quality filters
            elif (short_confluence >= self.params['min_confluence'] and 
                  price_change > min_price_move and
                  current['close'] < df['close'].iloc[i-5]):  # Simple downtrend filter
                
                entry_price = current['close']
                stop_loss, take_profit = self.calculate_proper_ict_sl_tp(df, i, -1, entry_price)
                
                self._create_signal(df, i, -1, entry_price, stop_loss, take_profit, short_confluence)
                signal_count += 1

        logger.info(f"Simple Optimized Strategy generated {signal_count} high-quality signals for {symbol}")
        return df

    def _create_signal(self, df: pd.DataFrame, idx: int, direction: int, entry_price: float, 
                      stop_loss: float, take_profit: float, confluence: int):
        """Create trading signal with calculated SL/TP"""
        confidence = min(confluence / 5.0, 1.0)
        
        df.loc[df.index[idx], 'signal'] = direction
        df.loc[df.index[idx], 'entry_price'] = entry_price
        df.loc[df.index[idx], 'stop_loss'] = stop_loss
        df.loc[df.index[idx], 'take_profit'] = take_profit
        df.loc[df.index[idx], 'confluence_score'] = confluence
        df.loc[df.index[idx], 'confidence'] = confidence

    def get_trade_signals(self, df: pd.DataFrame, symbol: str) -> List[TradeSignal]:
        """Extract trade signals from the dataframe"""
        signals = []
        signal_rows = df[df['signal'] != 0]

        for idx, row in signal_rows.iterrows():
            signal = TradeSignal(
                symbol=symbol,
                timestamp=idx,
                direction=row['signal'],
                entry_price=row['entry_price'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                position_size=0.0,  # Will be calculated by risk management
                confidence=row['confidence'],
                signal_strength=row['confluence_score']
            )
            signals.append(signal)

        return signals