import asyncio
import aiohttp
import websockets
import orjson
import numpy as np
import pandas as pd
import time
import ccxt.async_support as ccxt
import uvloop
import redis.asyncio as redis
import cupy as cp
from typing import Dict, List, AsyncGenerator, Optional
from dataclasses import dataclass
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from lru import LRU

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

@dataclass
class MarketUpdate:
    symbol: str
    exchange: str
    price: float
    volume: float
    bid: float
    ask: float
    timestamp_ns: int
    orderbook: Dict
    trades: List
    momentum_1s: float
    momentum_5s: float
    momentum_1m: float
    momentum_5m: float
    volatility: float
    liquidity_score: float
    spread: float
    depth: float
    vwap: float
    rsi: float
    macd: float
    bollinger: float
    sentiment: float
    funding_rate: float
    basis: float
    options_flow: float
    whale_activity: float
    news_sentiment: float
    social_sentiment: float
    github_activity: float
    dev_activity: float
    holder_distribution: float
    liquidity_locked: bool
    audit_score: float
    team_score: float
    tokenomics_score: float
    market_cap: float
    volume_24h: float
    circulating_supply: float
    total_supply: float
    max_supply: float

class UltraFastMarketEngine:
    def __init__(self, config):
        self.config = config
        self.exchanges = {}
        self.redis_client = None
        self.ws_connections = {}
        self.data_cache = LRU(1000000)
        self.price_history = defaultdict(lambda: deque(maxlen=100000))
        self.volume_history = defaultdict(lambda: deque(maxlen=100000))
        self.orderbook_cache = defaultdict(lambda: LRU(10000))
        self.trade_cache = defaultdict(lambda: deque(maxlen=50000))
        self.thread_pool = ThreadPoolExecutor(max_workers=200)
        self.processing_queue = asyncio.Queue(maxsize=100000)
        
    async def initialize_ultra_fast_feeds(self):
        self.redis_client = redis.Redis.from_url("redis://localhost:6379", decode_responses=False)
        
        exchange_list = [
            'okx', 'binance', 'coinbase', 'kraken', 'huobi', 'bybit', 
            'gate', 'kucoin', 'mexc', 'bitget', 'phemex', 'deribit',
            'ftx', 'bitmex', 'bitfinex', 'poloniex', 'bitstamp'
        ]
        
        init_tasks = []
        for exchange_name in exchange_list:
            if exchange_name in self.config['exchanges']:
                task = self.init_exchange(exchange_name)
                init_tasks.append(task)
                
        await asyncio.gather(*init_tasks, return_exceptions=True)
        
        await self.start_ultra_fast_streams()
        
    async def init_exchange(self, name):
        try:
            exchange_class = getattr(ccxt, name)
            config = self.config['exchanges'][name]
            
            exchange = exchange_class({
                'apiKey': config['api_key'],
                'secret': config['secret'],
                'password': config.get('passphrase', ''),
                'sandbox': False,
                'enableRateLimit': True,
                'rateLimit': 25,
                'options': {'defaultType': 'spot'}
            })
            
            await exchange.load_markets()
            self.exchanges[name] = exchange
            
        except Exception:
            pass
            
    async def start_ultra_fast_streams(self):
        stream_tasks = []
        
        for exchange_name in self.exchanges:
            tasks = [
                asyncio.create_task(self.stream_tickers(exchange_name)),
                asyncio.create_task(self.stream_orderbooks(exchange_name)),
                asyncio.create_task(self.stream_trades(exchange_name)),
                asyncio.create_task(self.stream_funding(exchange_name)),
                asyncio.create_task(self.stream_futures(exchange_name))
            ]
            stream_tasks.extend(tasks)
            
        batch_processor_tasks = [
            asyncio.create_task(self.batch_processor()) for _ in range(50)
        ]
        
        stream_tasks.extend(batch_processor_tasks)
        await asyncio.gather(*stream_tasks, return_exceptions=True)
        
    async def stream_tickers(self, exchange_name):
        symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT',
            'MATIC/USDT', 'AVAX/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT',
            'NEAR/USDT', 'FTM/USDT', 'ALGO/USDT', 'ICP/USDT', 'VET/USDT',
            'SAND/USDT', 'MANA/USDT', 'APE/USDT', 'LRC/USDT', 'CRV/USDT',
            'AAVE/USDT', 'COMP/USDT', 'SNX/USDT', 'YFI/USDT', 'SUSHI/USDT',
            'BAL/USDT', 'REN/USDT', 'KNC/USDT', 'ZRX/USDT', 'BNT/USDT'
        ]
        
        while True:
            try:
                for symbol in symbols:
                    try:
                        ticker = await self.exchanges[exchange_name].fetch_ticker(symbol)
                        await self.process_ticker(ticker, exchange_name)
                    except:
                        continue
                        
                await asyncio.sleep(0.02)
                
            except Exception:
                await asyncio.sleep(1)
                
    async def stream_orderbooks(self, exchange_name):
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
        
        while True:
            try:
                for symbol in symbols:
                    try:
                        orderbook = await self.exchanges[exchange_name].fetch_order_book(symbol, 50)
                        await self.process_orderbook(orderbook, symbol, exchange_name)
                    except:
                        continue
                        
                await asyncio.sleep(0.01)
                
            except Exception:
                await asyncio.sleep(1)
                
    async def stream_trades(self, exchange_name):
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        while True:
            try:
                for symbol in symbols:
                    try:
                        trades = await self.exchanges[exchange_name].fetch_trades(symbol, 50)
                        await self.process_trades(trades, symbol, exchange_name)
                    except:
                        continue
                        
                await asyncio.sleep(0.05)
                
            except Exception:
                await asyncio.sleep(1)
                
    async def stream_funding(self, exchange_name):
        while True:
            try:
                if hasattr(self.exchanges[exchange_name], 'fetch_funding_rates'):
                    rates = await self.exchanges[exchange_name].fetch_funding_rates()
                    await self.process_funding(rates, exchange_name)
                    
                await asyncio.sleep(30)
                
            except Exception:
                await asyncio.sleep(30)
                
    async def stream_futures(self, exchange_name):
        while True:
            try:
                if hasattr(self.exchanges[exchange_name], 'fetch_tickers'):
                    tickers = await self.exchanges[exchange_name].fetch_tickers()
                    await self.process_futures_data(tickers, exchange_name)
                    
                await asyncio.sleep(10)
                
            except Exception:
                await asyncio.sleep(10)
                
    async def process_ticker(self, ticker, exchange_name):
        symbol = ticker['symbol'].replace('/', '-')
        timestamp = time.time_ns()
        
        price = float(ticker['last']) if ticker['last'] else 0
        volume = float(ticker['quoteVolume']) if ticker['quoteVolume'] else 0
        
        self.price_history[symbol].append((timestamp, price))
        self.volume_history[symbol].append((timestamp, volume))
        
        data = {
            'symbol': symbol,
            'exchange': exchange_name,
            'price': price,
            'volume': volume,
            'bid': float(ticker['bid']) if ticker['bid'] else price,
            'ask': float(ticker['ask']) if ticker['ask'] else price,
            'timestamp': timestamp,
            'type': 'ticker'
        }
        
        await self.processing_queue.put(data)
        
    async def process_orderbook(self, orderbook, symbol, exchange_name):
        symbol = symbol.replace('/', '-')
        
        self.orderbook_cache[symbol][exchange_name] = {
            'bids': orderbook['bids'][:20],
            'asks': orderbook['asks'][:20],
            'timestamp': time.time_ns()
        }
        
        data = {
            'symbol': symbol,
            'exchange': exchange_name,
            'orderbook': orderbook,
            'type': 'orderbook'
        }
        
        await self.processing_queue.put(data)
        
    async def process_trades(self, trades, symbol, exchange_name):
        symbol = symbol.replace('/', '-')
        
        for trade in trades:
            self.trade_cache[symbol].append({
                'price': float(trade['price']),
                'amount': float(trade['amount']),
                'side': trade['side'],
                'timestamp': trade['timestamp'] or time.time_ns(),
                'exchange': exchange_name
            })
            
    async def process_funding(self, rates, exchange_name):
        for symbol, rate in rates.items():
            data = {
                'symbol': symbol.replace('/', '-'),
                'exchange': exchange_name,
                'funding_rate': float(rate['fundingRate']) if rate.get('fundingRate') else 0,
                'type': 'funding'
            }
            await self.processing_queue.put(data)
            
    async def process_futures_data(self, tickers, exchange_name):
        for symbol, ticker in tickers.items():
            if 'PERP' in symbol or 'SWAP' in symbol:
                data = {
                    'symbol': symbol.replace('/', '-'),
                    'exchange': exchange_name,
                    'futures_price': float(ticker['last']) if ticker['last'] else 0,
                    'type': 'futures'
                }
                await self.processing_queue.put(data)
                
    async def batch_processor(self):
        while True:
            try:
                items = []
                
                for _ in range(1000):
                    try:
                        item = await asyncio.wait_for(self.processing_queue.get(), timeout=0.001)
                        items.append(item)
                    except asyncio.TimeoutError:
                        break
                        
                if items:
                    await self.process_batch(items)
                    
            except Exception:
                await asyncio.sleep(0.001)
                
    async def process_batch(self, items):
        grouped = defaultdict(list)
        for item in items:
            grouped[item['symbol']].append(item)
            
        for symbol, symbol_items in grouped.items():
            await self.create_market_update(symbol, symbol_items)
            
    async def create_market_update(self, symbol, items):
        try:
            latest_ticker = None
            latest_orderbook = None
            funding_rate = 0
            futures_price = 0
            
            for item in items:
                if item['type'] == 'ticker':
                    latest_ticker = item
                elif item['type'] == 'orderbook':
                    latest_orderbook = item
                elif item['type'] == 'funding':
                    funding_rate = item['funding_rate']
                elif item['type'] == 'futures':
                    futures_price = item['futures_price']
                    
            if not latest_ticker:
                return
                
            price = latest_ticker['price']
            volume = latest_ticker['volume']
            
            momentum_data = await self.calculate_momentum(symbol, price)
            volatility = await self.calculate_volatility(symbol)
            technical_indicators = await self.calculate_technical_indicators(symbol)
            liquidity_metrics = await self.calculate_liquidity_metrics(symbol, latest_orderbook)
            sentiment_data = await self.get_sentiment_data(symbol)
            fundamental_data = await self.get_fundamental_data(symbol)
            
            basis = 0
            if futures_price > 0 and price > 0:
                basis = (futures_price - price) / price * 100
                
            market_update = MarketUpdate(
                symbol=symbol,
                exchange=latest_ticker['exchange'],
                price=price,
                volume=volume,
                bid=latest_ticker['bid'],
                ask=latest_ticker['ask'],
                timestamp_ns=latest_ticker['timestamp'],
                orderbook=latest_orderbook['orderbook'] if latest_orderbook else {},
                trades=list(self.trade_cache[symbol])[-20:],
                momentum_1s=momentum_data.get('1s', 0),
                momentum_5s=momentum_data.get('5s', 0),
                momentum_1m=momentum_data.get('1m', 0),
                momentum_5m=momentum_data.get('5m', 0),
                volatility=volatility,
                liquidity_score=liquidity_metrics.get('score', 0),
                spread=liquidity_metrics.get('spread', 0),
                depth=liquidity_metrics.get('depth', 0),
                vwap=technical_indicators.get('vwap', price),
                rsi=technical_indicators.get('rsi', 50),
                macd=technical_indicators.get('macd', 0),
                bollinger=technical_indicators.get('bollinger', 0.5),
                sentiment=sentiment_data.get('overall', 0.5),
                funding_rate=funding_rate,
                basis=basis,
                options_flow=sentiment_data.get('options', 0),
                whale_activity=sentiment_data.get('whale', 0),
                news_sentiment=sentiment_data.get('news', 0.5),
                social_sentiment=sentiment_data.get('social', 0.5),
                github_activity=fundamental_data.get('github', 0),
                dev_activity=fundamental_data.get('dev_activity', 0),
                holder_distribution=fundamental_data.get('holders', 0),
                liquidity_locked=fundamental_data.get('locked', True),
                audit_score=fundamental_data.get('audit', 0.5),
                team_score=fundamental_data.get('team', 0.5),
                tokenomics_score=fundamental_data.get('tokenomics', 0.5),
                market_cap=fundamental_data.get('market_cap', 0),
                volume_24h=volume,
                circulating_supply=fundamental_data.get('circulating', 0),
                total_supply=fundamental_data.get('total_supply', 0),
                max_supply=fundamental_data.get('max_supply', 0)
            )
            
            cache_key = f"market_update:{symbol}:{int(time.time())}"
            await self.redis_client.set(cache_key, orjson.dumps(market_update.__dict__), ex=300)
            
        except Exception:
            pass
            
    async def calculate_momentum(self, symbol, current_price):
        if len(self.price_history[symbol]) < 10:
            return {'1s': 0, '5s': 0, '1m': 0, '5m': 0}
            
        prices = list(self.price_history[symbol])
        current_time = time.time_ns()
        
        timeframes = {
            '1s': 1_000_000_000,
            '5s': 5_000_000_000,
            '1m': 60_000_000_000,
            '5m': 300_000_000_000
        }
        
        momentum = {}
        for tf_name, tf_ns in timeframes.items():
            cutoff_time = current_time - tf_ns
            relevant_prices = [(ts, p) for ts, p in prices if ts >= cutoff_time]
            
            if len(relevant_prices) >= 2:
                start_price = relevant_prices[0][1]
                end_price = relevant_prices[-1][1]
                if start_price > 0:
                    momentum[tf_name] = ((end_price - start_price) / start_price) * 100
                else:
                    momentum[tf_name] = 0
            else:
                momentum[tf_name] = 0
                
        return momentum
        
    async def calculate_volatility(self, symbol):
        if len(self.price_history[symbol]) < 20:
            return 0.02
            
        prices = [p[1] for p in list(self.price_history[symbol])[-100:]]
        if len(prices) < 2:
            return 0.02
            
        returns = np.diff(np.log(np.array(prices) + 1e-8))
        return float(np.std(returns) * np.sqrt(86400))
        
    async def calculate_technical_indicators(self, symbol):
        if len(self.price_history[symbol]) < 50:
            return {'vwap': 0, 'rsi': 50, 'macd': 0, 'bollinger': 0.5}
            
        prices = [p[1] for p in list(self.price_history[symbol])[-100:]]
        volumes = [v[1] for v in list(self.volume_history[symbol])[-100:]]
        
        if len(prices) < 20 or len(volumes) < 20:
            return {'vwap': prices[-1] if prices else 0, 'rsi': 50, 'macd': 0, 'bollinger': 0.5}
            
        vwap = np.average(prices[-20:], weights=volumes[-20:]) if sum(volumes[-20:]) > 0 else prices[-1]
        rsi = self.calculate_rsi(prices)
        macd = self.calculate_macd(prices)
        bollinger = self.calculate_bollinger_position(prices)
        
        return {
            'vwap': float(vwap),
            'rsi': float(rsi),
            'macd': float(macd),
            'bollinger': float(bollinger)
        }
        
    def calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
        
    def calculate_macd(self, prices):
        if len(prices) < 26:
            return 0.0
            
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        return ema_12 - ema_26
        
    def calculate_ema(self, prices, period):
        if len(prices) < period:
            return np.mean(prices)
            
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
        
    def calculate_bollinger_position(self, prices):
        if len(prices) < 20:
            return 0.5
            
        ma = np.mean(prices[-20:])
        std = np.std(prices[-20:])
        
        if std == 0:
            return 0.5
            
        upper = ma + (2 * std)
        lower = ma - (2 * std)
        current = prices[-1]
        
        position = (current - lower) / (upper - lower)
        return np.clip(position, 0, 1)
        
    async def calculate_liquidity_metrics(self, symbol, orderbook_data):
        if not orderbook_data or 'orderbook' not in orderbook_data:
            return {'score': 0, 'spread': 999, 'depth': 0}
            
        orderbook = orderbook_data['orderbook']
        
        if not orderbook.get('bids') or not orderbook.get('asks'):
            return {'score': 0, 'spread': 999, 'depth': 0}
            
        bids = orderbook['bids'][:10]
        asks = orderbook['asks'][:10]
        
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        
        spread = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 999
        
        bid_depth = sum(float(bid[0]) * float(bid[1]) for bid in bids)
        ask_depth = sum(float(ask[0]) * float(ask[1]) for ask in asks)
        total_depth = bid_depth + ask_depth
        
        liquidity_score = min(100, total_depth / 100000)
        
        return {
            'score': liquidity_score,
            'spread': spread,
            'depth': total_depth
        }
        
    async def get_sentiment_data(self, symbol):
        cache_key = f"sentiment:{symbol}"
        
        try:
            cached = await self.redis_client.get(cache_key)
            if cached:
                return orjson.loads(cached)
        except:
            pass
            
        return {
            'overall': 0.5,
            'social': 0.5,
            'news': 0.5,
            'whale': 0,
            'options': 0
        }
        
    async def get_fundamental_data(self, symbol):
        cache_key = f"fundamental:{symbol}"
        
        try:
            cached = await self.redis_client.get(cache_key)
            if cached:
                return orjson.loads(cached)
        except:
            pass
            
        return {
            'github': 0,
            'dev_activity': 0,
            'holders': 0,
            'locked': True,
            'audit': 0.5,
            'team': 0.5,
            'tokenomics': 0.5,
            'market_cap': 0,
            'circulating': 0,
            'total_supply': 0,
            'max_supply': 0
        }
        
    async def stream_market_data(self) -> AsyncGenerator[List[MarketUpdate], None]:
        batch_size = 500
        last_yield = time.time()
        
        while True:
            try:
                pattern = "market_update:*"
                keys = await self.redis_client.keys(pattern)
                
                if len(keys) >= batch_size or (time.time() - last_yield) > 0.1:
                    batch = []
                    
                    for key in keys[:batch_size]:
                        try:
                            data = await self.redis_client.get(key)
                            if data:
                                update_dict = orjson.loads(data)
                                update = MarketUpdate(**update_dict)
                                batch.append(update)
                                await self.redis_client.delete(key)
                        except:
                            continue
                            
                    if batch:
                        yield batch
                        last_yield = time.time()
                        
                await asyncio.sleep(0.001)
                
            except Exception:
                await asyncio.sleep(0.01)