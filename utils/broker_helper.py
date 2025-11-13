# utils/broker_helper.py
"""
Broker compatibility utilities
"""

import MetaTrader5 as mt5

def detect_broker_symbols():
    """Auto-detect available JPY pairs for any broker - FIXED VERSION"""
    try:
        symbols = mt5.symbols_get()
        jpy_pairs = []
        
        print(f"🔍 Scanning {len(symbols)} total symbols from broker...")
        
        # Look for JPY pairs with ANY suffix pattern
        for symbol in symbols:
            symbol_name = symbol.name
            # Check if it's a JPY pair (contains JPY in name)
            if 'JPY' in symbol_name:
                jpy_pairs.append(symbol_name)
        
        # Filter to only include pairs that end with 'm' (your broker's format)
        jpy_pairs_with_m = [pair for pair in jpy_pairs if pair.endswith('m')]
        
        if jpy_pairs_with_m:
            print(f"🎯 Filtered to {len(jpy_pairs_with_m)} JPY pairs with 'm' suffix")
            return jpy_pairs_with_m
        else:
            # Fallback: use detected JPY pairs even without 'm'
            print(f"⚠️  No 'm' suffix pairs found, using all {len(jpy_pairs)} JPY pairs")
            return jpy_pairs
        
    except Exception as e:
        print(f"❌ Symbol detection error: {e}")
        return ['USDJPYm', 'EURJPYm', 'GBPJPYm']  # Fallback to known pairs

def get_major_jpy_pairs():
    """Smart mapping of major JPY pairs to broker-specific symbols"""
    try:
        all_symbols = mt5.symbols_get()
        major_pairs_base = ['USDJPY', 'EURJPY', 'GBPJPY']  # Base names we want
        
        # Common suffix patterns across brokers
        suffix_patterns = ['', 'm', '.a', '.x', 'micro', 'mini', 'pro', 'cfd']
        
        major_pairs_found = []
        
        print("🔍 Mapping major JPY pairs to broker symbols...")
        
        for base_pair in major_pairs_base:
            found = False
            
            # Try different suffix combinations
            for suffix in suffix_patterns:
                test_symbol = base_pair + suffix
                
                # Check if this symbol exists
                for broker_symbol in all_symbols:
                    if broker_symbol.name == test_symbol:
                        major_pairs_found.append(test_symbol)
                        print(f"   ✅ {base_pair} → {test_symbol}")
                        found = True
                        break
                
                if found:
                    break
            
            if not found:
                print(f"   ❌ {base_pair} not found with any suffix")
        
        print(f"🎯 Mapped {len(major_pairs_found)} major JPY pairs")
        return major_pairs_found
        
    except Exception as e:
        print(f"❌ Error mapping major pairs: {e}")
        # Fallback to common patterns
        return ['USDJPYm', 'EURJPYm', 'GBPJPYm']

def get_broker_conditions(symbol):
    """Get broker-specific trading conditions"""
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            return {
                'min_lot': symbol_info.volume_min,
                'max_lot': symbol_info.volume_max,
                'lot_step': symbol_info.volume_step,
                'point': symbol_info.point,
                'min_stop_distance': symbol_info.trade_stops_level * symbol_info.point
            }
    except Exception as e:
        print(f"❌ Error getting broker conditions for {symbol}: {e}")
    
    return None