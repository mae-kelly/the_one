import asyncio
import aiohttp
import websockets
import orjson
import numpy as np
import pandas as pd
from typing import Dict, List, AsyncGenerator, Optional
from dataclasses import dataclass
from collections import deque, defaultdict
import time
import ccxt.async_support as ccxt
import uvloop
from concurrent.futures import ThreadPoolExecutor
import redis.asyncio as redis
from lru import LRU
import cupy as cp

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

@dataclass
class PerfectMarketUpdate:
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    timestamp_ns: int
    exchange: str
    orderbook_depth: Dict
    trade_flow: List
    liquidity_score: float
    momentum_1s: float
    momentum_5s: float
    momentum_1m: float
    momentum_5m: float
    volatility_realized: float
    volatility_implied: float
    microstructure_features: np.ndarray
    options_flow: Dict
    futures_basis: float
    funding_rate: float
    social_sentiment: float
    news_sentiment: float
    whale_activity: float
    cross_exchange_spread: float
    vwap: float
    twap: float
    rsi: float
    macd: float
    bollinger_position: float
    volume_profile: np.ndarray
    order_flow_imbalance: float
    effective_spread: float
    price_impact: float

class MarketDataEngine:
    def __init__(self, config):
        self.config = config
        self.exchanges = {}
        self.ws_connections = {}
        self.redis_client = None
        self.market_cache = LRU(100000)
        self.orderbook_cache = defaultdict(lambda: LRU(1000))
        self.trade_cache = defaultdict(lambda: deque(maxlen=10000))
        self.price_history = defaultdict(lambda: deque(maxlen=50000))
        self.volume_history = defaultdict(lambda: deque(maxlen=50000))
        self.momentum_calculator = MomentumCalculator()
        self.microstructure_analyzer = MicrostructureAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.correlation_engine = CorrelationEngine()
        self.thread_pool = ThreadPoolExecutor(max_workers=100)
        self.gpu_processor = GPUProcessor()
        
    async def initialize_all_exchanges(self):
        exchange_configs = {
            'okx': {'sandbox': False, 'rateLimit': 50},
            'binance': {'sandbox': False, 'rateLimit': 50},
            'coinbase': {'sandbox': False, 'rateLimit': 50},
            'kraken': {'sandbox': False, 'rateLimit': 50},
            'huobi': {'sandbox': False, 'rateLimit': 50},
            'bybit': {'sandbox': False, 'rateLimit': 50},
            'gate': {'sandbox': False, 'rateLimit': 50},
            'kucoin': {'sandbox': False, 'rateLimit': 50},
            'mexc': {'sandbox': False, 'rateLimit': 50},
            'bitget': {'sandbox': False, 'rateLimit': 50},
            'phemex': {'sandbox': False, 'rateLimit': 50},
            'deribit': {'sandbox': False, 'rateLimit': 50}
        }
        
        self.redis_client = redis.from_url(self.config['infrastructure']['redis_url'])
        
        init_tasks = []
        for exchange_name, config in exchange_configs.items():
            if exchange_name in self.config.get('api_keys', {}):
                task = self.initialize_single_exchange(exchange_name, config)
                init_tasks.append(task)
                
        await asyncio.gather(*init_tasks, return_exceptions=True)
        await self.start_all_websocket_streams()
        
    async def initialize_single_exchange(self, exchange_name: str, config: Dict):
        try:
            exchange_class = getattr(ccxt, exchange_name)
            
            exchange_config = {
                **config,
                'apiKey': self.config['api_keys'][exchange_name]['api_key'],
                'secret': self.config['api_keys'][exchange_name]['secret_key'],
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            }
            
            if 'passphrase' in self.config['api_keys'][exchange_name]:
                exchange_config['password'] = self.config['api_keys'][exchange_name]['passphrase']
                
            exchange = exchange_class(exchange_config)
            await exchange.load_markets()
            self.exchanges[exchange_name] = exchange
            
        except Exception as e:
            pass
            
    async def start_all_websocket_streams(self):
        ws_tasks = []
        
        for exchange_name, exchange in self.exchanges.items():
            task1 = asyncio.create_task(self.stream_exchange_tickers(exchange_name))
            task2 = asyncio.create_task(self.stream_exchange_orderbooks(exchange_name))
            task3 = asyncio.create_task(self.stream_exchange_trades(exchange_name))
            task4 = asyncio.create_task(self.stream_exchange_funding(exchange_name))
            ws_tasks.extend([task1, task2, task3, task4])
            
        await asyncio.gather(*ws_tasks, return_exceptions=True)
        
    async def stream_exchange_tickers(self, exchange_name: str):
        while True:
            try:
                symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 
                          'MATIC/USDT', 'AVAX/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT',
                          'NEAR/USDT', 'FTM/USDT', 'ALGO/USDT', 'ICP/USDT', 'VET/USDT',
                          'SAND/USDT', 'MANA/USDT', 'APE/USDT', 'LRC/USDT', 'CRV/USDT']
                          
                for symbol in symbols:
                    try:
                        ticker = await self.exchanges[exchange_name].fetch_ticker(symbol)
                        await self.process_ticker_update(ticker, exchange_name)
                    except:
                        continue
                        
                await asyncio.sleep(0.1)
                
            except Exception:
                await asyncio.sleep(5)
                
    async def stream_exchange_orderbooks(self, exchange_name: str):
        while True:
            try:
                symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
                
                for symbol in symbols:
                    try:
                        orderbook = await self.exchanges[exchange_name].fetch_order_book(symbol, 100)
                        await self.process_orderbook_update(orderbook, symbol, exchange_name)
                    except:
                        continue
                        
                await asyncio.sleep(0.05)
                
            except Exception:
                await asyncio.sleep(5)
                
    async def stream_exchange_trades(self, exchange_name: str):
        while True:
            try:
                symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
                
                for symbol in symbols:
                    try:
                        trades = await self.exchanges[exchange_name].fetch_trades(symbol, limit=100)
                        await self.process_trades_update(trades, symbol, exchange_name)
                    except:
                        continue
                        
                await asyncio.sleep(0.1)
                
            except Exception:
                await asyncio.sleep(5)
                
    async def stream_exchange_funding(self, exchange_name: str):
        while True:
            try:
                if hasattr(self.exchanges[exchange_name], 'fetch_funding_rates'):
                    funding_rates = await self.exchanges[exchange_name].fetch_funding_rates()
                    await self.process_funding_update(funding_rates, exchange_name)
                    
                await asyncio.sleep(60)
                
            except Exception:
                await asyncio.sleep(60)
                
    async def process_ticker_update(self, ticker, exchange_name):
        symbol = ticker['symbol'].replace('/', '-')
        price = float(ticker['last'])
        volume = float(ticker['quoteVolume']) if ticker['quoteVolume'] else 0
        
        self.price_history[symbol].append((time.time_ns(), price))
        self.volume_history[symbol].append((time.time_ns(), volume))
        
        cache_key = f"{symbol}_{exchange_name}_ticker"
        await self.redis_client.set(cache_key, orjson.dumps(ticker), ex=60)
        
    async def process_orderbook_update(self, orderbook, symbol, exchange_name):
        symbol = symbol.replace('/', '-')
        
        self.orderbook_cache[symbol][exchange_name] = {
            'bids': orderbook['bids'][:50],
            'asks': orderbook['asks'][:50],
            'timestamp': time.time_ns()
        }
        
        cache_key = f"{symbol}_{exchange_name}_orderbook"
        await self.redis_client.set(cache_key, orjson.dumps(orderbook), ex=10)
        
    async def process_trades_update(self, trades, symbol, exchange_name):
        symbol = symbol.replace('/', '-')
        
        for trade in trades:
            self.trade_cache[symbol].append({
                'price': float(trade['price']),
                'amount': float(trade['amount']),
                'side': trade['side'],
                'timestamp': trade['timestamp'],
                'exchange': exchange_name
            })
            
    async def process_funding_update(self, funding_rates, exchange_name):
        for symbol, rate in funding_rates.items():
            cache_key = f"{symbol}_funding_{exchange_name}"
            await self.redis_client.set(cache_key, str(rate), ex=3600)
            
    async def stream_ultra_fast_data(self) -> AsyncGenerator[List[PerfectMarketUpdate], None]:
        batch_size = 500
        batch_timeout = 0.001
        
        while True:
            batch = []
            start_time = time.time()
            
            for exchange_name in self.exchanges:
                updates = await self.get_exchange_updates(exchange_name)
                batch.extend(updates)
                
                if len(batch) >= batch_size:
                    break
                    
            if batch:
                processed_batch = await self.parallel_process_updates(batch)
                yield processed_batch
                
            await asyncio.sleep(0.0001)
            
    async def get_exchange_updates(self, exchange_name: str) -> List:
        updates = []
        
        try:
            symbols = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'ADA-USDT', 'DOT-USDT']
            
            for symbol in symbols:
                ticker_key = f"{symbol}_{exchange_name}_ticker"
                orderbook_key = f"{symbol}_{exchange_name}_orderbook"
                
                ticker_data = await self.redis_client.get(ticker_key)
                orderbook_data = await self.redis_client.get(orderbook_key)
                
                if ticker_data and orderbook_data:
                    ticker = orjson.loads(ticker_data)
                    orderbook = orjson.loads(orderbook_data)
                    
                    update = {
                        'symbol': symbol,
                        'price': float(ticker['last']),
                        'volume': float(ticker['quoteVolume']) if ticker['quoteVolume'] else 0,
                        'bid': float(ticker['bid']) if ticker['bid'] else 0,
                        'ask': float(ticker['ask']) if ticker['ask'] else 0,
                        'orderbook': orderbook,
                        'trades': list(self.trade_cache[symbol])[-20:],
                        'exchange': exchange_name,
                        'timestamp': time.time_ns()
                    }
                    
                    updates.append(update)
                    
        except Exception:
            pass
            
        return updates
        
    async def parallel_process_updates(self, raw_updates: List) -> List[PerfectMarketUpdate]:
        chunk_size = 50
        chunks = [raw_updates[i:i+chunk_size] for i in range(0, len(raw_updates), chunk_size)]
        
        processing_tasks = [self.process_update_chunk(chunk) for chunk in chunks]
        processed_chunks = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        processed_updates = []
        for chunk in processed_chunks:
            if not isinstance(chunk, Exception):
                processed_updates.extend(chunk)
                
        return processed_updates
        
    async def process_update_chunk(self, chunk: List) -> List[PerfectMarketUpdate]:
        processed = []
        
        for raw_update in chunk:
            try:
                symbol = raw_update['symbol']
                price = float(raw_update['price'])
                volume = float(raw_update['volume'])
                
                momentum_data = await self.momentum_calculator.calculate_all_timeframes(symbol, price, raw_update['timestamp'])
                volatility_data = await self.calculate_volatility_metrics(symbol, price)
                microstructure = await self.microstructure_analyzer.analyze(symbol, raw_update.get('orderbook', {}), raw_update.get('trades', []))
                liquidity_metrics = await self.liquidity_analyzer.analyze_liquidity(symbol, raw_update.get('orderbook', {}))
                technical_indicators = await self.calculate_technical_indicators(symbol, price)
                cross_exchange_data = await self.get_cross_exchange_data(symbol)
                sentiment_data = await self.get_sentiment_data(symbol)
                options_data = await self.get_options_data(symbol)
                
                perfect_update = PerfectMarketUpdate(
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    bid=float(raw_update.get('bid', price)),
                    ask=float(raw_update.get('ask', price)),
                    timestamp_ns=int(time.time_ns()),
                    exchange=raw_update.get('exchange', 'unknown'),
                    orderbook_depth=raw_update.get('orderbook', {}),
                    trade_flow=raw_update.get('trades', []),
                    liquidity_score=liquidity_metrics.get('score', 0),
                    momentum_1s=momentum_data.get('1s', 0),
                    momentum_5s=momentum_data.get('5s', 0),
                    momentum_1m=momentum_data.get('1m', 0),
                    momentum_5m=momentum_data.get('5m', 0),
                    volatility_realized=volatility_data.get('realized', 0),
                    volatility_implied=volatility_data.get('implied', 0),
                    microstructure_features=microstructure.get('features', np.zeros(50)),
                    options_flow=options_data,
                    futures_basis=cross_exchange_data.get('basis', 0),
                    funding_rate=cross_exchange_data.get('funding', 0),
                    social_sentiment=sentiment_data.get('social', 0.5),
                    news_sentiment=sentiment_data.get('news', 0.5),
                    whale_activity=sentiment_data.get('whale', 0),
                    cross_exchange_spread=cross_exchange_data.get('spread', 0),
                    vwap=technical_indicators.get('vwap', price),
                    twap=technical_indicators.get('twap', price),
                    rsi=technical_indicators.get('rsi', 50),
                    macd=technical_indicators.get('macd', 0),
                    bollinger_position=technical_indicators.get('bollinger_position', 0.5),
                    volume_profile=technical_indicators.get('volume_profile', np.zeros(10)),
                    order_flow_imbalance=microstructure.get('order_flow_imbalance', 0),
                    effective_spread=microstructure.get('effective_spread', 0),
                    price_impact=microstructure.get('price_impact', 0)
                )
                
                processed.append(perfect_update)
                await self.cache_update(perfect_update)
                
            except Exception:
                continue
                
        return processed
        
    async def calculate_volatility_metrics(self, symbol: str, price: float) -> Dict:
        if len(self.price_history[symbol]) < 20:
            return {'realized': 0.02, 'implied': 0.03}
            
        prices = [p[1] for p in list(self.price_history[symbol])[-100:]]
        returns = np.diff(np.log(prices))
        
        realized_vol = np.std(returns) * np.sqrt(86400)
        implied_vol = realized_vol * 1.2
        
        return {'realized': realized_vol, 'implied': implied_vol}
        
    async def calculate_technical_indicators(self, symbol: str, price: float) -> Dict:
        if len(self.price_history[symbol]) < 50:
            return {'vwap': price, 'twap': price, 'rsi': 50, 'macd': 0, 'bollinger_position': 0.5, 'volume_profile': np.zeros(10)}
            
        prices = [p[1] for p in list(self.price_history[symbol])[-100:]]
        volumes = [v[1] for v in list(self.volume_history[symbol])[-100:]]
        
        vwap = np.average(prices[-20:], weights=volumes[-20:]) if len(volumes) >= 20 else price
        twap = np.mean(prices[-20:])
        
        rsi = self.calculate_rsi(prices)
        macd = self.calculate_macd(prices)
        bollinger_position = self.calculate_bollinger_position(prices, price)
        volume_profile = self.calculate_volume_profile(prices, volumes)
        
        return {
            'vwap': vwap,
            'twap': twap,
            'rsi': rsi,
            'macd': macd,
            'bollinger_position': bollinger_position,
            'volume_profile': volume_profile
        }
        
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
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
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def calculate_macd(self, prices: List[float]) -> float:
        if len(prices) < 26:
            return 0.0
            
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        
        return ema_12 - ema_26
        
    def calculate_ema(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return np.mean(prices)
            
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
            
        return ema
        
    def calculate_bollinger_position(self, prices: List[float], current_price: float) -> float:
        if len(prices) < 20:
            return 0.5
            
        ma = np.mean(prices[-20:])
        std = np.std(prices[-20:])
        
        upper_band = ma + (2 * std)
        lower_band = ma - (2 * std)
        
        if upper_band == lower_band:
            return 0.5
            
        position = (current_price - lower_band) / (upper_band - lower_band)
        return np.clip(position, 0, 1)
        
    def calculate_volume_profile(self, prices: List[float], volumes: List[float]) -> np.ndarray:
        if len(prices) < 10:
            return np.zeros(10)
            
        price_min, price_max = min(prices[-50:]), max(prices[-50:])
        if price_max == price_min:
            return np.zeros(10)
            
        bins = np.linspace(price_min, price_max, 10)
        volume_profile = np.zeros(10)
        
        for i, (price, volume) in enumerate(zip(prices[-50:], volumes[-50:])):
            bin_idx = min(9, int((price - price_min) / (price_max - price_min) * 9))
            volume_profile[bin_idx] += volume
            
        return volume_profile
        
    async def get_cross_exchange_data(self, symbol: str) -> Dict:
        try:
            prices = []
            for exchange_name in self.exchanges:
                ticker_key = f"{symbol}_{exchange_name}_ticker"
                ticker_data = await self.redis_client.get(ticker_key)
                if ticker_data:
                    ticker = orjson.loads(ticker_data)
                    prices.append(float(ticker['last']))
                    
            if len(prices) >= 2:
                spread = (max(prices) - min(prices)) / min(prices) * 100
                return {'spread': spread, 'basis': 0, 'funding': 0}
                
        except Exception:
            pass
            
        return {'spread': 0, 'basis': 0, 'funding': 0}
        
    async def get_sentiment_data(self, symbol: str) -> Dict:
        try:
            sentiment_key = f"{symbol}_sentiment"
            sentiment_data = await self.redis_client.get(sentiment_key)
            
            if sentiment_data:
                return orjson.loads(sentiment_data)
                
        except Exception:
            pass
            
        return {'social': 0.5, 'news': 0.5, 'whale': 0}
        
    async def get_options_data(self, symbol: str) -> Dict:
        return {'put_call_ratio': 1.0, 'implied_volatility': 0.3, 'max_pain': 0}
        
    async def cache_update(self, update: PerfectMarketUpdate):
        cache_key = f"market_update_{update.symbol}_{update.timestamp_ns}"
        await self.redis_client.set(cache_key, orjson.dumps(update.__dict__), ex=300)

class MomentumCalculator:
    def __init__(self):
        self.price_history = defaultdict(lambda: deque(maxlen=10000))
        
    async def calculate_all_timeframes(self, symbol: str, price: float, timestamp: int) -> Dict:
        self.price_history[symbol].append((timestamp, price))
        
        if len(self.price_history[symbol]) < 10:
            return {'1s': 0, '5s': 0, '1m': 0, '5m': 0}
            
        prices = list(self.price_history[symbol])
        current_time = timestamp
        
        timeframes = {
            '1s': 1000000000,
            '5s': 5000000000,
            '1m': 60000000000,
            '5m': 300000000000
        }
        
        momentum = {}
        for tf_name, tf_ns in timeframes.items():
            cutoff_time = current_time - tf_ns
            relevant_prices = [(ts, p) for ts, p in prices if ts >= cutoff_time]
            
            if len(relevant_prices) >= 2:
                start_price = relevant_prices[0][1]
                end_price = relevant_prices[-1][1]
                momentum[tf_name] = ((end_price - start_price) / start_price) * 100
            else:
                momentum[tf_name] = 0
                
        return momentum

class MicrostructureAnalyzer:
    def __init__(self):
        self.trade_flow_history = defaultdict(lambda: deque(maxlen=1000))
        
    async def analyze(self, symbol: str, orderbook: Dict, trades: List) -> Dict:
        features = np.zeros(50)
        
        try:
            if 'bids' in orderbook and 'asks' in orderbook:
                bid_volume = sum(float(bid[1]) for bid in orderbook['bids'][:10])
                ask_volume = sum(float(ask[1]) for ask in orderbook['asks'][:10])
                
                if bid_volume + ask_volume > 0:
                    order_flow_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                    features[0] = order_flow_imbalance
                    
                best_bid = float(orderbook['bids'][0][0]) if orderbook['bids'] else 0
                best_ask = float(orderbook['asks'][0][0]) if orderbook['asks'] else 0
                
                if best_bid > 0:
                    effective_spread = (best_ask - best_bid) / best_bid
                    features[1] = effective_spread
                    
            if trades:
                trade_sizes = [float(trade.get('amount', 0)) for trade in trades]
                if trade_sizes:
                    features[2] = np.mean(trade_sizes)
                    features[3] = np.std(trade_sizes)
                    
                buy_volume = sum(float(trade.get('amount', 0)) for trade in trades if trade.get('side') == 'buy')
                sell_volume = sum(float(trade.get('amount', 0)) for trade in trades if trade.get('side') == 'sell')
                
                if buy_volume + sell_volume > 0:
                    features[4] = (buy_volume - sell_volume) / (buy_volume + sell_volume)
                    
                vwap = self.calculate_vwap(trades)
                last_price = float(trades[-1].get('price', 0)) if trades else 0
                if vwap > 0:
                    features[5] = (last_price - vwap) / vwap
                    
        except Exception:
            pass
            
        return {
            'features': features,
            'order_flow_imbalance': features[0],
            'effective_spread': features[1],
            'price_impact': features[5]
        }
        
    def calculate_vwap(self, trades: List) -> float:
        total_volume = 0
        total_value = 0
        
        for trade in trades:
            price = float(trade.get('price', 0))
            volume = float(trade.get('amount', 0))
            total_value += price * volume
            total_volume += volume
            
        return total_value / total_volume if total_volume > 0 else 0

class LiquidityAnalyzer:
    async def analyze_liquidity(self, symbol: str, orderbook: Dict) -> Dict:
        if not orderbook.get('bids') or not orderbook.get('asks'):
            return {'score': 0, 'depth': 0, 'spread': 999}
            
        bids = orderbook['bids'][:20]
        asks = orderbook['asks'][:20]
        
        bid_depth = sum(float(bid[0]) * float(bid[1]) for bid in bids)
        ask_depth = sum(float(ask[0]) * float(ask[1]) for ask in asks)
        total_depth = bid_depth + ask_depth
        
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        spread = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 999
        
        liquidity_score = min(100, total_depth / 10000)
        
        return {
            'score': liquidity_score,
            'depth': total_depth,
            'spread': spread,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth
        }

class CorrelationEngine:
    def __init__(self):
        self.correlation_matrix = np.eye(1000)
        self.symbol_index = {}
        self.price_matrix = np.zeros((1000, 1000))
        
    async def update_correlations(self, updates: List[PerfectMarketUpdate]):
        pass
        
    async def get_correlation_matrix(self) -> np.ndarray:
        return self.correlation_matrix.copy()

class GPUProcessor:
    def __init__(self):
        self.gpu_available = cp.cuda.is_available()
        
    async def process_gpu_calculations(self, data: np.ndarray) -> np.ndarray:
        if self.gpu_available:
            gpu_data = cp.asarray(data)
            result = cp.asnumpy(gpu_data)
            return result
        return data