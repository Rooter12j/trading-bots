import asyncio
import ccxt
import pandas as pd
import numpy as np
import aiohttp
import os
import pickle
import datetime
from dotenv import load_dotenv

load_dotenv()
# Set event loop policy for Windows
import sys
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize the Binance API
exchange = ccxt.binance({
    'apiKey': os.getenv("API_KEY"),  # Replace with your actual API key
    'secret': os.getenv("SECRET"),  # Replace with your actual API secret
    'enableRateLimit': True,
})

# Define trading parameters
timeframes = ['1h', '4h']
limit = 100
risk_per_trade = 0.5
stop_loss_pct = 0.5
take_profit_pct = 0.20
CACHE_DIR = "cache"
CACHE_EXPIRY_SECONDS = 3600

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def sanitize_symbol(symbol):
    return symbol.replace('/', '_').replace(':', '_')

def cache_file_path(symbol, timeframe):
    sanitized_symbol = sanitize_symbol(symbol)
    return os.path.join(CACHE_DIR, f"cache_{sanitized_symbol}_{timeframe}.pkl")

def save_cache(symbol, timeframe, df):
    with open(cache_file_path(symbol, timeframe), 'wb') as f:
        pickle.dump({
            'timestamp': datetime.datetime.now(),
            'data': df
        }, f)

def load_cache(symbol, timeframe):
    cache_path = cache_file_path(symbol, timeframe)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
            cache_timestamp = cached['timestamp']
            df = cached['data']
            if (datetime.datetime.now() - cache_timestamp).total_seconds() < CACHE_EXPIRY_SECONDS:
                return df
            else:
                return None
    return None

async def fetch_data(symbol, timeframe):
    df = load_cache(symbol, timeframe)
    if df is None:
        print(f"Cache miss for {symbol} {timeframe}. Fetching fresh data...")
        async with aiohttp.ClientSession() as session:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol.replace('/', ''), 
                'interval': timeframe,
                'limit': limit
            }
            try:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    bars = await response.json()
                    if bars:
                        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        save_cache(symbol, timeframe, df)
                    else:
                        print(f"No data returned for {symbol} {timeframe}.")
                        df = pd.DataFrame()
            except aiohttp.ClientError as e:
                print(f"Client error while fetching data for {symbol} {timeframe}: {e}")
                df = pd.DataFrame()
            except Exception as e:
                print(f"Unexpected error while fetching data for {symbol} {timeframe}: {e}")
                df = pd.DataFrame()
    else:
        print(f"Loaded cached data for {symbol} {timeframe}.")
    return df

def calculate_atr(df, period=14):
    df['TR'] = df[['high', 'low', 'close']].apply(lambda row: max(row['high'] - row['low'], abs(row['high'] - row['close'].shift(1)), abs(row['low'] - row['close'].shift(1))), axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

def trailing_stop_loss(entry_price, atr_value, multiplier=1.5):
    return entry_price - (atr_value * multiplier)

def calculate_indicators(df):
    if df.empty:
        return df

    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + df['close'].pct_change().add(1).rolling(window=14).apply(lambda x: np.mean(x[x > 0]) / np.mean(np.abs(x)))))
    df = calculate_atr(df)
    df['OBV'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    return df

def enhanced_trading_signal(df):
    if df.empty or len(df) < 2:
        return 'hold'

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    buy_signal = (last_row['SMA_20'] > last_row['SMA_50']) and (last_row['MACD'] > last_row['MACD_Signal']) and (last_row['RSI'] < 30)
    sell_signal = (last_row['SMA_20'] < last_row['SMA_50']) and (last_row['MACD'] < last_row['MACD_Signal']) and (last_row['RSI'] > 70)

    if buy_signal:
        return 'buy'
    elif sell_signal:
        return 'sell'
    else:
        return 'hold'

def calculate_position_size(balance, risk_per_trade, stop_loss_pct, entry_price):
    risk_amount = balance * risk_per_trade
    position_size = risk_amount / (stop_loss_pct * entry_price)
    return position_size

def dynamic_risk_adjustment(balance, recent_performance):
    max_risk_per_trade = 0.5
    if recent_performance < 0.5:
        return max_risk_per_trade * 0.5
    return max_risk_per_trade

async def execute_trade(symbol, signal, entry_price, position_size):
    try:
        if signal == 'buy':
            print(f"Executing Buy Order for {symbol}: {position_size} units at {entry_price}")
            await asyncio.to_thread(exchange.create_market_buy_order, symbol, position_size)
            atr = calculate_atr((df).iloc[-1]['ATR']) # type: ignore
            stop_loss_price = trailing_stop_loss(entry_price, atr)
            take_profit_price = entry_price * (1 + take_profit_pct)
            print(f"Stop-Loss Set at {stop_loss_price}")
            print(f"Take-Profit Set at {take_profit_price}")
            await asyncio.to_thread(exchange.create_limit_sell_order, symbol, position_size, take_profit_price)
            await asyncio.to_thread(exchange.create_stop_loss_order, symbol, position_size, stop_loss_price)
        elif signal == 'sell':
            print(f"Executing Sell Order for {symbol}: {position_size} units at {entry_price}")
            await asyncio.to_thread(exchange.create_market_sell_order, symbol, position_size)
            atr = calculate_atr((df).iloc[-1]['ATR']) # type: ignore
            stop_loss_price = trailing_stop_loss(entry_price, atr)
            take_profit_price = entry_price * (1 - take_profit_pct)
            print(f"Stop-Loss Set at {stop_loss_price}")
            print(f"Take-Profit Set at {take_profit_price}")
            await asyncio.to_thread(exchange.create_limit_buy_order, symbol, position_size, take_profit_price)
            await asyncio.to_thread(exchange.create_stop_loss_order, symbol, position_size, stop_loss_price)
    except Exception as e:
        print(f"Error executing trade for {symbol}: {e}")

async def fetch_symbols():
    try:
        markets = await asyncio.to_thread(exchange.fetch_markets)
        symbols = [market['symbol'] for market in markets if market['active'] and market['quote'] == 'USDT']
        return symbols
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

async def multi_timeframe_signal(symbol):
    async def process_timeframe(timeframe):
        df = await fetch_data(symbol, timeframe)
        df = calculate_indicators(df)
        signal = enhanced_trading_signal(df)
        return timeframe, signal

    tasks = [process_timeframe(tf) for tf in timeframes]
    results = await asyncio.gather(*tasks)
    
    signals = dict(results)
    
    if all(signals[tf] == 'buy' for tf in timeframes):
        return 'buy'
    elif all(signals[tf] == 'sell' for tf in timeframes):
        return 'sell'
    else:
        return 'hold'

async def main():
    symbols = await fetch_symbols()
    print("Available symbols:")
    for idx, symbol in enumerate(symbols):
        print(f"{idx}: {symbol}")
    
    selected_symbols = symbols[267:269]  # Open up to 3 positions
    
    for selected_symbol in selected_symbols:
        try:
            signal = await multi_timeframe_signal(selected_symbol)
            print(f"Trading signal for {selected_symbol}: {signal}")
            
            if signal != 'hold':
                balance = await asyncio.to_thread(exchange.fetch_balance)
                usable_balance = balance['total']['USDT']
                position_size = calculate_position_size(usable_balance, risk_per_trade, stop_loss_pct, 1)  # Adjust entry_price as necessary
                
                await execute_trade(selected_symbol, signal, 1, position_size)  # Adjust entry_price as necessary
        except Exception as e:
            print(f"Error processing {selected_symbol}: {e}")

if __name__ == '__main__':
    asyncio.run(main())
