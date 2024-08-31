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
    'secret': os.getenv("SECRET", default=None),  # Replace with your actual API secret
    'enableRateLimit': True,
})

# Define trading parameters
timeframes = ['1h', '4h']  # Multiple timeframes for analysis
limit = 100  # Number of candles to fetch
risk_per_trade = 0.5 # Risk 0.5% of capital per trade
stop_loss_pct = 0.05  # Stop-loss percentage (5%)
take_profit_pct = 0.20  # Take-profit percentage (20%)
CACHE_DIR = "cache"  # Directory to store cached files
CACHE_EXPIRY_SECONDS = 3600  # Cache expiry time in seconds (e.g., 1 hour)
MAX_POSITIONS = 3  # Maximum number of open positions

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
            # Check if cache is expired
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
                'symbol': symbol.replace('/', ''),  # Adjust symbol format if necessary
                'interval': timeframe,
                'limit': limit
            }
            try:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()  # Raise HTTPError for bad responses
                    bars = await response.json()
                    if bars:  # Check if bars is not empty
                        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        save_cache(symbol, timeframe, df)  # Save the data to cache
                    else:
                        print(f"No data returned for {symbol} {timeframe}.")
                        df = pd.DataFrame()  # Return an empty DataFrame
            except aiohttp.ClientError as e:
                print(f"Client error while fetching data for {symbol} {timeframe}: {e}")
                df = pd.DataFrame()  # Return an empty DataFrame
            except Exception as e:
                print(f"Unexpected error while fetching data for {symbol} {timeframe}: {e}")
                df = pd.DataFrame()  # Return an empty DataFrame
    else:
        print(f"Loaded cached data for {symbol} {timeframe}.")
    return df

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
    
    # Volume Analysis
    df['OBV'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    return df

def trading_signal(df):
    if df.empty or len(df) < 2:
        return 'hold'  # Not enough data to make a decision

    last_row = df.iloc[-1]
    
    # Trend Analysis
    if last_row['SMA_20'] > last_row['SMA_50']:
        trend_signal = 'buy'
    elif last_row['SMA_20'] < last_row['SMA_50']:
        trend_signal = 'sell'
    else:
        trend_signal = 'hold'
    
    # MACD Signal
    if last_row['MACD'] > last_row['MACD_Signal']:
        macd_signal = 'buy'
    elif last_row['MACD'] < last_row['MACD_Signal']:
        macd_signal = 'sell'
    else:
        macd_signal = 'hold'
    
    # RSI Signal
    if last_row['RSI'] < 30:
        rsi_signal = 'buy'
    elif last_row['RSI'] > 70:
        rsi_signal = 'sell'
    else:
        rsi_signal = 'hold'
    
    # Volume Analysis
    if last_row['OBV'] > df['OBV'].iloc[-2]:
        obv_signal = 'buy'
    elif last_row['OBV'] < df['OBV'].iloc[-2]:
        obv_signal = 'sell'
    else:
        obv_signal = 'hold'
    
    # Combine Signals
    if trend_signal == macd_signal == rsi_signal == obv_signal:
        return trend_signal
    else:
        return 'hold'

def calculate_position_size(balance, risk_per_trade, stop_loss_pct, entry_price):
    risk_amount = balance * risk_per_trade
    position_size = risk_amount / (stop_loss_pct * entry_price)
    return position_size

async def execute_trade(symbol, signal, entry_price, position_size):
    try:
        if signal == 'buy':
            print(f"Executing Buy Order for {symbol}: {position_size} units at {entry_price}")
            await asyncio.to_thread(exchange.create_market_buy_order, symbol, position_size)
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)
            print(f"Stop-Loss Set at {stop_loss_price}")
            print(f"Take-Profit Set at {take_profit_price}")
            await asyncio.to_thread(exchange.create_limit_sell_order, symbol, position_size, take_profit_price)
            await asyncio.to_thread(exchange.create_stop_loss_order, symbol, position_size, stop_loss_price)
        elif signal == 'sell':
            print(f"Executing Sell Order for {symbol}: {position_size} units at {entry_price}")
            await asyncio.to_thread(exchange.create_market_sell_order, symbol, position_size)
            stop_loss_price = entry_price * (1 + stop_loss_pct)
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

async def trade_all_symbols():
    try:
        balance = await asyncio.to_thread(exchange.fetch_balance)
        usdt_balance = balance['total'].get('USDT', 0)
        if usdt_balance == 0:
            print("No USDT balance available for trading.")
            return

        symbols = await fetch_symbols()
        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                tasks.append(fetch_data(symbol, timeframe))

        all_data = await asyncio.gather(*tasks)

        open_positions = 0
        for i, df in enumerate(all_data):
            if df.empty:
                continue

            df = calculate_indicators(df)
            signal = trading_signal(df)
            symbol = symbols[i // len(timeframes)]

            if open_positions < MAX_POSITIONS and signal in ['buy', 'sell']:
                last_price = df['close'].iloc[-1]
                position_size = calculate_position_size(usdt_balance, risk_per_trade, stop_loss_pct, last_price)
                
                if position_size * last_price <= usdt_balance:  # Check if there's enough balance
                    await execute_trade(symbol, signal, last_price, position_size)
                    usdt_balance -= position_size * last_price  # Update balance after trade
                    open_positions += 1
                    if open_positions >= MAX_POSITIONS:
                        print("Maximum number of open positions reached.")
                        break

    except Exception as e:
        print(f"Error in trading all symbols: {e}")

if __name__ == "__main__":
    asyncio.run(trade_all_symbols())
