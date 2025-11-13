# utils/broker_helper.py
"""
Broker compatibility utilities
"""

def detect_broker_symbols():
    """Auto-detect available JPY pairs for any broker"""
    try:
        symbols = mt5.symbols_get()
        jpy_pairs = []
        
        # Common JPY pair patterns across brokers
        patterns = ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY', 'NZDJPY']
        
        for pattern in patterns:
            # Try different suffixes
            variations = [
                pattern,           # USDJPY
                pattern + 'm',     # USDJPYm
                pattern + '.a',    # USDJPY.a
                pattern + 'micro', # USDJPYmicro
            ]
            
            for variation in variations:
                if any(variation in s.name for s in symbols):
                    jpy_pairs.append(variation)
                    break  # Found one, move to next pattern
        
        print(f"🔍 Detected {len(jpy_pairs)} JPY pairs:")
        for pair in jpy_pairs:
            print(f"   ✅ {pair}")
            
        return jpy_pairs
        
    except Exception as e:
        print(f"❌ Symbol detection error: {e}")
        return ['USDJPYm', 'EURJPYm', 'GBPJPYm']  # Fallback

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
    except:
        pass
    return None