# live/live_demo_bot.py
"""
COMPLETE FIXED LIVE DEMO BOT - BROKER AGNOSTIC
With PROPER ICT SL/TP Methodology and Auto-Detection
"""

import MetaTrader5 as mt5
import pandas as pd
import time
import logging
from datetime import datetime
import os
import sys

# Add project root to path
sys.path.append('.')

from config.config import MT5, TRADING, PAIRS, STRATEGY
from strategies.ict_strategy import SimpleOptimizedICTStrategy
from data.data_loader import MT5DataLoader
from utils.broker_helper import detect_broker_symbols, get_broker_conditions  # ✅ ADDED

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/ict_demo_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ICTDemoBot:
    """LIVE DEMO BOT - With PROPER ICT SL/TP Methodology & Broker Auto-Detection"""
    
    def __init__(self):
        self.strategy = SimpleOptimizedICTStrategy(STRATEGY)
        self.data_loader = MT5DataLoader()
        self.is_running = False
        self.trade_count = 0
        self.max_trades_per_day = 3
        self.available_pairs = []  # ✅ ADDED - Will auto-populate
        
    def initialize_mt5(self):
        """Initialize MT5 connection with broker auto-detection"""
        print("🔌 Connecting to MT5...")
        
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return False
            
        # Login to account
        if not mt5.login(MT5["login"], MT5["password"], MT5["server"]):
            print("❌ MT5 login failed")
            return False
            
        account_info = mt5.account_info()
        if account_info:
            print(f"✅ Connected to Account: {account_info.login}")
            print(f"💰 Balance: ${account_info.balance:.2f}")
            print(f"💼 Server: {account_info.server}")
            print(f"📈 Leverage: 1:{account_info.leverage}")
            
            # ✅ AUTO-DETECT AVAILABLE PAIRS FOR THIS BROKER
            print("🔍 Auto-detecting available JPY pairs...")
            self.available_pairs = detect_broker_symbols()
            
            if not self.available_pairs:
                print("❌ No JPY pairs detected! Using fallback pairs...")
                self.available_pairs = ['USDJPY', 'EURJPY', 'GBPJPY']  # Fallback
            else:
                print(f"🎯 Auto-detected {len(self.available_pairs)} JPY pairs:")
                for pair in self.available_pairs:
                    print(f"   ✅ {pair}")
            
            return True
        else:
            print("❌ Cannot get account info")
            return False

    def validate_broker_stops(self, symbol, entry_price, stop_loss, take_profit, direction):
        """Validate and adjust stops to meet broker requirements"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return stop_loss, take_profit, "Symbol info not available"
                
            # ✅ GET BROKER-SPECIFIC CONDITIONS
            broker_conditions = get_broker_conditions(symbol)
            if broker_conditions:
                min_stop_distance = broker_conditions['min_stop_distance']
                point = broker_conditions['point']
            else:
                point = symbol_info.point
                min_stop_distance = symbol_info.trade_stops_level * point
                
            if min_stop_distance == 0:
                min_stop_distance = 0.002  # Increased minimum for safety
                
            print(f"   Broker min stop distance: {min_stop_distance:.4f}")
            
            if direction == 'LONG':
                # For LONG: SL below entry, TP above entry
                sl_distance = entry_price - stop_loss
                tp_distance = take_profit - entry_price
                
                print(f"   Current SL distance: {sl_distance:.4f}")
                print(f"   Current TP distance: {tp_distance:.4f}")
                
                # Ensure SL is at proper distance
                if sl_distance < min_stop_distance:
                    stop_loss = entry_price - max(min_stop_distance, entry_price * 0.005)  # At least 0.5%
                    print(f"   🔧 Adjusted SL to proper distance: ${stop_loss:.3f}")
                    
                # Ensure TP is at proper distance and makes sense
                if tp_distance < min_stop_distance:
                    take_profit = entry_price + max(min_stop_distance, (entry_price - stop_loss) * 1.5)
                    print(f"   🔧 Adjusted TP to proper distance: ${take_profit:.3f}")
                    
            else:  # SHORT
                # For SHORT: SL above entry, TP below entry
                sl_distance = stop_loss - entry_price
                tp_distance = entry_price - take_profit
                
                print(f"   Current SL distance: {sl_distance:.4f}")
                print(f"   Current TP distance: {tp_distance:.4f}")
                
                # Ensure SL is at proper distance
                if sl_distance < min_stop_distance:
                    stop_loss = entry_price + max(min_stop_distance, entry_price * 0.005)  # At least 0.5%
                    print(f"   🔧 Adjusted SL to proper distance: ${stop_loss:.3f}")
                    
                # Ensure TP is at proper distance and makes sense
                if tp_distance < min_stop_distance:
                    take_profit = entry_price - max(min_stop_distance, (stop_loss - entry_price) * 1.5)
                    print(f"   🔧 Adjusted TP to proper distance: ${take_profit:.3f}")
            
            # Final sanity check
            if direction == 'LONG':
                if stop_loss >= entry_price:
                    stop_loss = entry_price * 0.99
                    print(f"   🛠️  Emergency SL correction: ${stop_loss:.3f}")
                if take_profit <= entry_price:
                    take_profit = entry_price * 1.02
                    print(f"   🛠️  Emergency TP correction: ${take_profit:.3f}")
            else:  # SHORT
                if stop_loss <= entry_price:
                    stop_loss = entry_price * 1.01
                    print(f"   🛠️  Emergency SL correction: ${stop_loss:.3f}")
                if take_profit >= entry_price:
                    take_profit = entry_price * 0.98
                    print(f"   🛠️  Emergency TP correction: ${take_profit:.3f}")
            
            return stop_loss, take_profit, "Broker stops validated and adjusted"
            
        except Exception as e:
            return stop_loss, take_profit, f"Broker validation error: {e}"
    
    def calculate_safe_position_size(self, symbol, entry_price, stop_loss, direction):
        """Calculate SAFE position size with broker-specific conditions"""
        try:
            account_info = mt5.account_info()
            
            # ✅ GET BROKER-SPECIFIC CONDITIONS
            broker_conditions = get_broker_conditions(symbol)
            if broker_conditions:
                min_lot = broker_conditions['min_lot']
                max_lot = broker_conditions['max_lot']
                lot_step = broker_conditions['lot_step']
            else:
                # Fallback to symbol info
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    min_lot = symbol_info.volume_min
                    max_lot = symbol_info.volume_max
                    lot_step = symbol_info.volume_step
                else:
                    min_lot = 0.01
                    max_lot = 1.00
                    lot_step = 0.01
            
            if not account_info:
                return min_lot
                
            equity = account_info.balance
            risk_amount = equity * 0.003  # 0.3% risk
            
            if direction == 'LONG':
                risk_per_unit = entry_price - stop_loss
            else:  # SHORT
                risk_per_unit = stop_loss - entry_price
                
            if risk_per_unit <= 0:
                return min_lot
                
            # Calculate lot size
            position_size = risk_amount / risk_per_unit
            
            # ✅ USE BROKER-SPECIFIC CONSTRAINTS
            position_size = max(min_lot, min(position_size, max_lot))
            position_size = round(position_size / lot_step) * lot_step
            
            # Conservative max limit
            position_size = min(position_size, 0.05)
            
            print(f"   📊 Broker conditions: Min={min_lot}, Max={max_lot}, Step={lot_step}")
            print(f"   💰 Position size: {position_size:.3f} lots")
            
            return round(position_size, 3)
            
        except Exception as e:
            print(f"❌ Position sizing error: {e}")
            return 0.01
    
    def execute_ict_trade(self, symbol, direction, entry_price, stop_loss, take_profit):
        """Execute trade with PROPER ICT methodology"""
        try:
            # Safety checks
            if self.trade_count >= self.max_trades_per_day:
                print("🛑 Daily trade limit reached")
                return False
                
            # Check if symbol is available
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Symbol {symbol} not available")
                return False
            
            # Validate broker requirements
            stop_loss, take_profit, validation_msg = self.validate_broker_stops(
                symbol, entry_price, stop_loss, take_profit, direction
            )
            print(f"   {validation_msg}")
            
            # Calculate safe position size
            lot_size = self.calculate_safe_position_size(symbol, entry_price, stop_loss, direction)
            
            # Prepare trade request
            if direction == 'LONG':
                order_type = mt5.ORDER_TYPE_BUY
            else:
                order_type = mt5.ORDER_TYPE_SELL
                
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": entry_price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": 123459,
                "comment": "ICT-Proper",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            print(f"🔄 EXECUTING ICT TRADE: {direction} {symbol} {lot_size} lots")
            print(f"   Entry: ${entry_price:.3f}")
            print(f"   SL: ${stop_loss:.3f} (ICT: {'below swing low/BSL' if direction == 'LONG' else 'above swing high/SSL'})")
            print(f"   TP: ${take_profit:.3f} (ICT: {'below swing high/SSL' if direction == 'LONG' else 'above swing low/BSL'})")
            
            # Calculate and display R:R
            if direction == 'LONG':
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
                
            rr_ratio = reward / risk if risk > 0 else 0
            print(f"   Risk: ${risk:.3f}, Reward: ${reward:.3f}, R:R: {rr_ratio:.1f}:1")
            
            # SEND THE TRADE!
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ ICT TRADE EXECUTED! Ticket: {result.order}")
                print(f"💰 {direction} {symbol} {lot_size} lots")
                self.trade_count += 1
                return True
            else:
                print(f"❌ Trade failed: {result.retcode} - {result.comment}")
                return False
                
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
            return False
    
    def get_current_price(self, symbol):
        """Get current bid/ask price"""
        try:
            if not mt5.symbol_select(symbol, True):
                print(f"❌ Cannot select symbol: {symbol}")
                return None
                
            tick = mt5.symbol_info_tick(symbol)
            if tick and tick.bid > 0 and tick.ask > 0:
                return (tick.bid + tick.ask) / 2
        except:
            pass
        return None
    
    def scan_and_trade_ict(self):
        """Scan for signals and execute PROPER ICT trades using auto-detected pairs"""
        print("🔍 Scanning for ICT signals...")
        
        # ✅ USE AUTO-DETECTED PAIRS INSTEAD OF HARDCODED
        if not self.available_pairs:
            print("❌ No pairs available for trading")
            return
            
        # Use first 2 detected pairs (or all if less than 2)
        trading_pairs = self.available_pairs[:min(2, len(self.available_pairs))]
        
        print(f"🎯 Trading pairs: {trading_pairs}")
        
        for symbol in trading_pairs:
            try:
                # Get current price
                current_price = self.get_current_price(symbol)
                if not current_price:
                    print(f"❌ Cannot get price for {symbol}")
                    continue
                    
                # Get historical data
                df = self.data_loader.fetch_data(symbol, "H4", bars=100)
                if df is None or len(df) < 50:
                    print(f"❌ Insufficient data for {symbol}")
                    continue
                    
                # Generate signals with PROPER ICT SL/TP
                df_with_signals = self.strategy.generate_signals(df, symbol)
                
                # Get latest signal
                latest_signals = df_with_signals[df_with_signals['signal'] != 0].tail(1)
                
                if not latest_signals.empty:
                    signal_data = latest_signals.iloc[-1]
                    
                    # Only trade high-confidence signals
                    if signal_data['confidence'] >= 0.7:
                        direction = 'LONG' if signal_data['signal'] == 1 else 'SHORT'
                        
                        print(f"🎯 ICT SIGNAL: {direction} {symbol}")
                        print(f"   Strategy Entry: ${signal_data['entry_price']:.3f}")
                        print(f"   ICT SL: ${signal_data['stop_loss']:.3f}")
                        print(f"   ICT TP: ${signal_data['take_profit']:.3f}")
                        print(f"   Confidence: {signal_data['confidence']:.1%}")
                        print(f"   Confluence: {signal_data['confluence_score']}/5")
                        
                        # Check for existing positions
                        positions = mt5.positions_get(symbol=symbol)
                        if positions:
                            print(f"⏸️  Position exists for {symbol}, skipping...")
                            continue
                            
                        # Use current market price
                        market_entry = current_price
                        
                        # EXECUTE PROPER ICT TRADE
                        print("🚀 EXECUTING PROPER ICT TRADE...")
                        if self.execute_ict_trade(
                            symbol,
                            direction, 
                            market_entry,
                            signal_data['stop_loss'],
                            signal_data['take_profit']
                        ):
                            print("💰 ICT TRADE SUCCESSFUL!")
                        else:
                            print("❌ ICT TRADE FAILED")
                            
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                continue
    
    def monitor_positions(self):
        """Monitor open positions"""
        try:
            positions = mt5.positions_get()
            if positions:
                print(f"\n📊 OPEN POSITIONS: {len(positions)}")
                total_pnl = 0
                for pos in positions:
                    pnl_color = "🟢" if pos.profit >= 0 else "🔴"
                    type_str = "LONG" if pos.type == 0 else "SHORT"
                    print(f"   {pnl_color} {pos.symbol} {type_str} - P&L: ${pos.profit:.2f}")
                    total_pnl += pos.profit
                    
                print(f"   Total P&L: ${total_pnl:.2f}")
            else:
                print("📊 No open positions")
                
        except Exception as e:
            print(f"❌ Error monitoring positions: {e}")
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        print(f"\n{'='*50}")
        print(f"🔄 ICT TRADING CYCLE - {datetime.now()}")
        print(f"{'='*50}")
        
        # 1. Monitor current positions
        self.monitor_positions()
        
        # 2. Scan and execute new ICT trades
        if self.trade_count < self.max_trades_per_day:
            self.scan_and_trade_ict()
        else:
            print("🛑 Daily trade limit reached")
            
        # 3. Show account summary
        account_info = mt5.account_info()
        if account_info:
            equity = account_info.equity
            balance = account_info.balance
            pnl = equity - balance
            pnl_pct = (pnl / balance) * 100 if balance > 0 else 0
            
            print(f"\n💼 ACCOUNT SUMMARY:")
            print(f"   Balance: ${balance:.2f}")
            print(f"   Equity: ${equity:.2f}")
            print(f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            print(f"   ICT Trades Today: {self.trade_count}/{self.max_trades_per_day}")
            print(f"   Available Pairs: {len(self.available_pairs)}")
    
    def run_bot(self):
        """Main bot execution loop"""
        print("🤖 STARTING ICT TRADING BOT - BROKER AGNOSTIC")
        print("="*60)
        print("🎯 PROPER ICT SL/TP METHODOLOGY:")
        print("   • LONG: SL below swing low/BSL, TP below swing high/SSL")
        print("   • SHORT: SL above swing high/SSL, TP above swing low/BSL")  
        print("   • Realistic R:R ratios (1.5-3.0)")
        print("🌍 BROKER AGNOSTIC FEATURES:")
        print("   • Auto-detects available JPY pairs")
        print("   • Adapts to broker-specific conditions")
        print("   • Works with any MT5 broker")
        print("💰 Using REAL money (demo or live)")
        print("⚡ Strategy: Simple Optimized ICT (60.9% win rate)")
        print("📈 Risk: 0.3% per trade, Max 3 trades/day")
        print("⏸️  Press Ctrl+C to STOP trading immediately")
        print("="*60)
        
        # Final confirmation
        response = input("❓ Start ICT trading? (yes/no): ")
        if response.lower() != 'yes':
            print("🛑 Trading cancelled by user")
            return
            
        if not self.initialize_mt5():
            return
            
        self.is_running = True
        cycle_count = 0
        
        try:
            while self.is_running and cycle_count < 24:
                cycle_count += 1
                self.run_trading_cycle()
                
                print(f"\n💤 Waiting 30 minutes... (Cycle {cycle_count}/24)")
                time.sleep(1800)
                
        except KeyboardInterrupt:
            print("\n🛑 EMERGENCY STOP - Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Bot error: {e}")
        finally:
            mt5.shutdown()
            print("🔴 ICT Trading bot shutdown complete")

def main():
    """Main function"""
    bot = ICTDemoBot()
    bot.run_bot()

if __name__ == "__main__":
    main()