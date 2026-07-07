import numpy as np
import pandas as pd

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_hma(series: pd.Series, period: int) -> pd.Series:
    half_len = int(period / 2)
    sqrt_len = int(np.sqrt(period))
    wma_half = calculate_wma(series, half_len)
    wma_full = calculate_wma(series, period)
    raw_hma = 2 * wma_half - wma_full
    return calculate_wma(raw_hma, sqrt_len)

def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_supertrend_dir(df: pd.DataFrame, period: int, factor: float) -> pd.Series:
    atr = calculate_atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    
    basic_upper = hl2 + factor * atr
    basic_lower = hl2 - factor * atr
    
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    
    for i in range(1, len(df)):
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or df["close"].iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
            
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or df["close"].iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
            
    direction = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > final_upper.iloc[i]:
            direction.iloc[i] = 1 # bullish
        elif df["close"].iloc[i] < final_lower.iloc[i]:
            direction.iloc[i] = -1 # bearish
        else:
            direction.iloc[i] = direction.iloc[i-1]
            
    return direction

def build_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Base indicators
    df["base_ema"] = calculate_ema(df["close"], 12)
    df["atr_10"] = calculate_atr(df, 10)
    
    # DTM Bands
    df["upper_band_raw"] = df["base_ema"] + df["atr_10"] * 2.0
    df["lower_band_raw"] = df["base_ema"] - df["atr_10"] * 2.0
    
    # DTM Trend State
    dtm_trend = np.zeros(len(df))
    for i in range(1, len(df)):
        prev_trend = dtm_trend[i-1]
        close_curr = df["close"].iloc[i]
        
        raw_long_flip = close_curr > df["upper_band_raw"].iloc[i-1] and prev_trend != 1
        raw_short_flip = close_curr < df["lower_band_raw"].iloc[i-1] and prev_trend != -1
        
        if raw_long_flip:
            dtm_trend[i] = 1
        elif raw_short_flip:
            dtm_trend[i] = -1
        else:
            dtm_trend[i] = prev_trend
            
    df["dtm_trend"] = dtm_trend
    
    # Trend Master Supertrends
    dir1 = calculate_supertrend_dir(df, 1, 1.0)
    dir2 = calculate_supertrend_dir(df, 2, 2.0)
    dir3 = calculate_supertrend_dir(df, 15, 3.0)
    
    # HMA
    df["hma_50"] = calculate_hma(df["close"], 50)
    df["hma_slope_pct"] = (df["hma_50"] - df["hma_50"].shift(1)) / df["hma_50"].shift(1) * 100.0
    
    # Supertrend direction (BuySell Trend Master confluence)
    # 1 if all three are bullish, -1 if all three are bearish, 0 otherwise
    st_direction = np.zeros(len(df))
    for i in range(len(df)):
        if dir1.iloc[i] == 1 and dir2.iloc[i] == 1 and dir3.iloc[i] == 1:
            st_direction[i] = 1
        elif dir1.iloc[i] == -1 and dir2.iloc[i] == -1 and dir3.iloc[i] == -1:
            st_direction[i] = -1
    df["st_direction"] = st_direction
    
    # Candle Wick Ratios
    total_range = df["high"] - df["low"]
    body_top = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)
    
    upper_wick = df["high"] - body_top
    lower_wick = body_bottom - df["low"]
    
    df["upper_wick_pct"] = np.where(total_range > 0, (upper_wick / total_range) * 100.0, 0.0)
    df["lower_wick_pct"] = np.where(total_range > 0, (lower_wick / total_range) * 100.0, 0.0)
    
    # Squeeze
    df["dtm_band_width_pct"] = (df["upper_band_raw"] - df["lower_band_raw"]) / df["close"] * 100.0
    
    # Pivot Highs / Lows (18 bar lookback)
    df["last_pivot_high"] = df["high"].shift(1).rolling(18).max()
    df["last_pivot_low"] = df["low"].shift(1).rolling(18).min()
    
    df["dist_to_pivot_high_pct"] = (df["last_pivot_high"] - df["close"]) / df["close"] * 100.0
    df["dist_to_pivot_low_pct"] = (df["close"] - df["last_pivot_low"]) / df["close"] * 100.0
    
    # Base EMA Slope
    df["base_ema_slope_pct"] = (df["base_ema"] - df["base_ema"].shift(1)) / df["base_ema"].shift(1) * 100.0
    
    return df
