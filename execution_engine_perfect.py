import asyncio
import aiohttp
import orjson
import time
import hmac
import hashlib
import base64
import numpy as np
import ccxt.async_support as ccxt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
from datetime import datetime
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

@dataclass
class ExecutionResult:
    success: bool
    order_id: str
    executed_price: float
    executed_size: float
    slippage: float
    fees: float
    execution_time_ms: int
    venue: str
    strategy: str
    error: Optional[str] = None

@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    entry_time: int
    fees_paid: float
    venue: str
    
    @property
    def profit_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100

@dataclass
class ExecutionPlan:
    symbol: str
    side: str
    total_size: float
    max_slippage: float
    time_limit: int
    strategy: str
    venues: List[str]
    chunks: List[Dict]
    urgency: str

class MultiVenueExecutionEngine:
    def __init__(self, config):
        self.config = config
        self.exchanges = {}
        self.positions = {}
        self.execution_queue = asyncio.Queue(maxsize=50000)
        self.order_cache = {}
        self.venue_latencies = {}
        self.venue_liquidity = {}
        self.execution_history = deque(maxlen=100000)
        self.session_pools = {}
        
    async def initialize_all_venues(self):
        venue_configs = {
            'okx': {'weight': 0.25, 'max_size': 100000},
            'binance': {'weight': 0.25, 'max_size': 100000},
            'coinbase': {'weight': 0.20, 'max_size': 80000},
            'kraken': {'weight': 0.15, 'max_size': 60000},
            'huobi': {'weight': 0.10, 'max_size': 50000},
            'bybit': {'weight': 0.05, 'max_size': 40000}
        }
        
        init_tasks = []
        for venue_name, venue_config in venue_configs.items():
            if venue_name in self.config['exchanges']:
                task = self.initialize_venue(venue_name, venue_config)
                init_tasks.append(task)
                
        await asyncio.gather(*init_tasks, return_exceptions=True)
        
        await self.start_latency_monitoring()
        await self.start_execution_workers()
        
    async def initialize_venue(self, venue_name, config):
        try:
            exchange_class = getattr(ccxt, venue_name)
            
            exchange_config = {
                'apiKey': self.config['exchanges'][venue_name]['api_key'],
                'secret': self.config['exchanges'][venue_name]['secret'],
                'password': self.config['exchanges'][venue_name].get('passphrase', ''),
                'sandbox': False,
                'enableRateLimit': True,
                'rateLimit': 10,
                'options': {'defaultType': 'spot'}
            }
            
            exchange = exchange_class(exchange_config)
            await exchange.load_markets()
            
            self.exchanges[venue_name] = {
                'exchange': exchange,
                'config': config,
                'active_orders': {},
                'last_order_time': 0,
                'error_count': 0
            }
            
            connector = aiohttp.TCPConnector(limit=200, keepalive_timeout=30)
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=5)
            )
            self.session_pools[venue_name] = session
            
        except Exception:
            pass
            
    async def start_latency_monitoring(self):
        for venue_name in self.exchanges:
            asyncio.create_task(self.monitor_venue_latency(venue_name))
            
    async def monitor_venue_latency(self, venue_name):
        while True:
            try:
                start_time = time.time()
                
                exchange = self.exchanges[venue_name]['exchange']
                await exchange.fetch_ticker('BTC/USDT')
                
                latency = (time.time() - start_time) * 1000
                self.venue_latencies[venue_name] = latency
                
                await asyncio.sleep(10)
                
            except Exception:
                self.venue_latencies[venue_name] = 9999
                await asyncio.sleep(10)
                
    async def start_execution_workers(self):
        for i in range(20):
            asyncio.create_task(self.execution_worker())
            
    async def execution_worker(self):
        while True:
            try:
                execution_plan = await self.execution_queue.get()
                await self.execute_plan(execution_plan)
            except Exception:
                await asyncio.sleep(0.001)
                
    async def create_optimal_plan(self, signal, position_size):
        symbol = signal.symbol
        side = 'buy'
        
        available_venues = self.get_available_venues(symbol)
        venue_scores = await self.calculate_venue_scores(symbol, position_size, available_venues)
        
        best_venues = sorted(venue_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        strategy = self.select_execution_strategy(position_size, signal.urgency if hasattr(signal, 'urgency') else 'normal')
        
        chunks = await self.calculate_optimal_chunks(position_size, best_venues, strategy)
        
        execution_plan = ExecutionPlan(
            symbol=symbol,
            side=side,
            total_size=position_size,
            max_slippage=self.config['execution_config']['max_slippage'],
            time_limit=self.config['execution_config']['timeout'],
            strategy=strategy,
            venues=[venue for venue, _ in best_venues],
            chunks=chunks,
            urgency='normal'
        )
        
        return execution_plan
        
    def get_available_venues(self, symbol):
        available = []
        
        for venue_name, venue_data in self.exchanges.items():
            if (venue_data['error_count'] < 5 and 
                venue_name in self.venue_latencies and 
                self.venue_latencies[venue_name] < 1000):
                available.append(venue_name)
                
        return available
        
    async def calculate_venue_scores(self, symbol, size, venues):
        scores = {}
        
        for venue in venues:
            try:
                latency_score = max(0, 1 - (self.venue_latencies.get(venue, 1000) / 1000))
                
                liquidity_score = await self.get_venue_liquidity_score(venue, symbol)
                
                fee_score = 1 - self.get_venue_fee_rate(venue)
                
                reliability_score = max(0, 1 - (self.exchanges[venue]['error_count'] / 10))
                
                capacity_score = min(1, self.exchanges[venue]['config']['max_size'] / size)
                
                composite_score = (
                    latency_score * 0.3 +
                    liquidity_score * 0.25 +
                    fee_score * 0.2 +
                    reliability_score * 0.15 +
                    capacity_score * 0.1
                )
                
                scores[venue] = composite_score
                
            except Exception:
                scores[venue] = 0.1
                
        return scores
        
    async def get_venue_liquidity_score(self, venue, symbol):
        try:
            exchange = self.exchanges[venue]['exchange']
            orderbook = await exchange.fetch_order_book(symbol, 20)
            
            bid_depth = sum(float(bid[1]) for bid in orderbook['bids'][:10])
            ask_depth = sum(float(ask[1]) for ask in orderbook['asks'][:10])
            total_depth = bid_depth + ask_depth
            
            return min(1.0, total_depth / 1000000)
            
        except Exception:
            return 0.5
            
    def get_venue_fee_rate(self, venue):
        fee_rates = {
            'okx': 0.001,
            'binance': 0.001,
            'coinbase': 0.005,
            'kraken': 0.0016,
            'huobi': 0.002,
            'bybit': 0.001
        }
        return fee_rates.get(venue, 0.002)
        
    def select_execution_strategy(self, size, urgency):
        if urgency == 'emergency':
            return 'aggressive_market'
        elif size > 50000:
            return 'iceberg_twap'
        elif size > 10000:
            return 'smart_limit'
        else:
            return 'market_sweep'
            
    async def calculate_optimal_chunks(self, total_size, venues, strategy):
        chunks = []
        
        if strategy == 'market_sweep':
            chunk_size = min(total_size, 5000)
            num_chunks = int(np.ceil(total_size / chunk_size))
            
            for i in range(num_chunks):
                size = min(chunk_size, total_size - i * chunk_size)
                venue = venues[i % len(venues)][0]
                
                chunks.append({
                    'venue': venue,
                    'size': size,
                    'order_type': 'market',
                    'delay': i * 0.1
                })
                
        elif strategy == 'iceberg_twap':
            chunk_size = min(total_size / 10, 8000)
            num_chunks = int(np.ceil(total_size / chunk_size))
            
            for i in range(num_chunks):
                size = min(chunk_size, total_size - i * chunk_size)
                venue = venues[i % len(venues)][0]
                
                chunks.append({
                    'venue': venue,
                    'size': size,
                    'order_type': 'limit',
                    'delay': i * 2.0
                })
                
        elif strategy == 'smart_limit':
            primary_venue = venues[0][0]
            secondary_venue = venues[1][0] if len(venues) > 1 else primary_venue
            
            chunks = [
                {
                    'venue': primary_venue,
                    'size': total_size * 0.7,
                    'order_type': 'limit',
                    'delay': 0
                },
                {
                    'venue': secondary_venue,
                    'size': total_size * 0.3,
                    'order_type': 'limit',
                    'delay': 0.5
                }
            ]
            
        elif strategy == 'aggressive_market':
            chunks = [{
                'venue': venues[0][0],
                'size': total_size,
                'order_type': 'market',
                'delay': 0
            }]
            
        return chunks
        
    async def execute_multi_venue(self, execution_plan):
        await self.execution_queue.put(execution_plan)
        
        plan_id = f"plan_{int(time.time_ns())}"
        self.order_cache[plan_id] = {
            'plan': execution_plan,
            'results': [],
            'status': 'pending'
        }
        
        return await self.wait_for_plan_completion(plan_id)
        
    async def execute_plan(self, execution_plan):
        plan_id = f"plan_{int(time.time_ns())}"
        results = []
        
        for chunk in execution_plan.chunks:
            if chunk['delay'] > 0:
                await asyncio.sleep(chunk['delay'])
                
            result = await self.execute_chunk(chunk, execution_plan.symbol, execution_plan.side)
            results.append(result)
            
            if not result.success and execution_plan.urgency == 'emergency':
                break
                
        self.order_cache[plan_id] = {
            'plan': execution_plan,
            'results': results,
            'status': 'completed'
        }
        
    async def execute_chunk(self, chunk, symbol, side):
        venue = chunk['venue']
        size = chunk['size']
        order_type = chunk['order_type']
        
        start_time = time.time()
        
        try:
            exchange = self.exchanges[venue]['exchange']
            
            if order_type == 'market':
                order = await exchange.create_market_order(symbol, side, size)
            else:
                price = await self.calculate_limit_price(venue, symbol, side)
                order = await exchange.create_limit_order(symbol, side, size, price)
                
            filled_order = await self.wait_for_fill(venue, order['id'], symbol, 30)
            
            if filled_order:
                execution_time = int((time.time() - start_time) * 1000)
                
                executed_price = float(filled_order.get('average', filled_order.get('price', 0)))
                executed_size = float(filled_order.get('filled', 0))
                fees = float(filled_order.get('fee', {}).get('cost', 0))
                
                slippage = self.calculate_slippage(symbol, side, executed_price)
                
                result = ExecutionResult(
                    success=True,
                    order_id=order['id'],
                    executed_price=executed_price,
                    executed_size=executed_size,
                    slippage=slippage,
                    fees=fees,
                    execution_time_ms=execution_time,
                    venue=venue,
                    strategy=order_type
                )
                
                await self.update_position(symbol, side, executed_size, executed_price, venue, fees)
                self.execution_history.append(result)
                
                return result
                
        except Exception as e:
            self.exchanges[venue]['error_count'] += 1
            
        return ExecutionResult(
            success=False,
            order_id='',
            executed_price=0,
            executed_size=0,
            slippage=0,
            fees=0,
            execution_time_ms=int((time.time() - start_time) * 1000),
            venue=venue,
            strategy=order_type,
            error=str(e) if 'e' in locals() else 'Unknown error'
        )
        
    async def calculate_limit_price(self, venue, symbol, side):
        try:
            exchange = self.exchanges[venue]['exchange']
            ticker = await exchange.fetch_ticker(symbol)
            
            if side == 'buy':
                return float(ticker['bid']) * 1.0002
            else:
                return float(ticker['ask']) * 0.9998
                
        except Exception:
            return 0
            
    async def wait_for_fill(self, venue, order_id, symbol, timeout):
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                exchange = self.exchanges[venue]['exchange']
                order = await exchange.fetch_order(order_id, symbol)
                
                if order['status'] in ['closed', 'filled']:
                    return order
                elif order['status'] == 'canceled':
                    break
                    
                await asyncio.sleep(0.1)
                
            except Exception:
                await asyncio.sleep(0.5)
                
        return None
        
    def calculate_slippage(self, symbol, side, executed_price):
        if symbol in self.venue_latencies:
            estimated_price = 50000
            return abs(executed_price - estimated_price) / estimated_price
        return 0.001
        
    async def update_position(self, symbol, side, size, price, venue, fees):
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                size=size if side == 'buy' else -size,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0,
                realized_pnl=0,
                entry_time=int(time.time()),
                fees_paid=fees,
                venue=venue
            )
        else:
            position = self.positions[symbol]
            
            if side == 'buy':
                if position.size >= 0:
                    total_cost = position.size * position.entry_price + size * price
                    total_size = position.size + size
                    position.entry_price = total_cost / total_size if total_size > 0 else price
                    position.size = total_size
                else:
                    if size >= abs(position.size):
                        realized_pnl = (price - position.entry_price) * abs(position.size)
                        position.realized_pnl += realized_pnl
                        position.size = size - abs(position.size)
                        position.entry_price = price
                    else:
                        realized_pnl = (price - position.entry_price) * size
                        position.realized_pnl += realized_pnl
                        position.size += size
            else:
                if position.size <= 0:
                    total_cost = abs(position.size) * position.entry_price + size * price
                    total_size = abs(position.size) + size
                    position.entry_price = total_cost / total_size if total_size > 0 else price
                    position.size = -total_size
                else:
                    if size >= position.size:
                        realized_pnl = (price - position.entry_price) * position.size
                        position.realized_pnl += realized_pnl
                        position.size = -(size - position.size)
                        position.entry_price = price
                    else:
                        realized_pnl = (price - position.entry_price) * size
                        position.realized_pnl += realized_pnl
                        position.size -= size
                        
            position.fees_paid += fees
            
    async def wait_for_plan_completion(self, plan_id, timeout=60):
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if plan_id in self.order_cache:
                cached_data = self.order_cache[plan_id]
                if cached_data['status'] == 'completed':
                    return self.aggregate_results(cached_data['results'])
                    
            await asyncio.sleep(0.1)
            
        return ExecutionResult(
            success=False,
            order_id='timeout',
            executed_price=0,
            executed_size=0,
            slippage=0,
            fees=0,
            execution_time_ms=60000,
            venue='timeout',
            strategy='timeout',
            error='Execution timeout'
        )
        
    def aggregate_results(self, results):
        if not results:
            return ExecutionResult(
                success=False,
                order_id='no_results',
                executed_price=0,
                executed_size=0,
                slippage=0,
                fees=0,
                execution_time_ms=0,
                venue='none',
                strategy='failed'
            )
            
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return results[0]
            
        total_size = sum(r.executed_size for r in successful_results)
        total_value = sum(r.executed_price * r.executed_size for r in successful_results)
        total_fees = sum(r.fees for r in successful_results)
        
        avg_price = total_value / total_size if total_size > 0 else 0
        avg_slippage = np.mean([r.slippage for r in successful_results])
        max_execution_time = max(r.execution_time_ms for r in successful_results)
        
        venues_used = ','.join(set(r.venue for r in successful_results))
        order_ids = ','.join(r.order_id for r in successful_results)
        
        return ExecutionResult(
            success=True,
            order_id=order_ids,
            executed_price=avg_price,
            executed_size=total_size,
            slippage=avg_slippage,
            fees=total_fees,
            execution_time_ms=max_execution_time,
            venue=venues_used,
            strategy='multi_venue'
        )
        
    async def execute_rebalancing(self, rebalancing_trade):
        execution_plan = await self.create_optimal_plan(rebalancing_trade, rebalancing_trade.size)
        return await self.execute_multi_venue(execution_plan)
        
    async def close_all_positions(self):
        close_tasks = []
        
        for symbol, position in self.positions.items():
            if abs(position.size) > 0:
                side = 'sell' if position.size > 0 else 'buy'
                size = abs(position.size)
                
                task = self.create_close_position_task(symbol, side, size)
                close_tasks.append(task)
                
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
            
    async def create_close_position_task(self, symbol, side, size):
        class CloseSignal:
            def __init__(self, symbol):
                self.symbol = symbol
                self.urgency = 'emergency'
                
        signal = CloseSignal(symbol)
        execution_plan = await self.create_optimal_plan(signal, size)
        execution_plan.side = side
        
        return await self.execute_multi_venue(execution_plan)
        
    async def emergency_exit(self, position):
        side = 'sell' if position.size > 0 else 'buy'
        size = abs(position.size)
        
        emergency_chunk = {
            'venue': list(self.exchanges.keys())[0],
            'size': size,
            'order_type': 'market',
            'delay': 0
        }
        
        return await self.execute_chunk(emergency_chunk, position.symbol, side)
        
    def get_positions(self):
        return list(self.positions.values())
        
    async def get_execution_queue(self):
        queue_items = []
        
        while not self.execution_queue.empty():
            try:
                item = self.execution_queue.get_nowait()
                queue_items.append(item)
            except:
                break
                
        return queue_items
        
    async def execute_optimal(self, trade):
        return await self.execute_multi_venue(trade)
        
    def get_execution_stats(self):
        if not self.execution_history:
            return {}
            
        successful_executions = [r for r in self.execution_history if r.success]
        
        if not successful_executions:
            return {'success_rate': 0, 'avg_slippage': 0, 'avg_execution_time': 0}
            
        return {
            'success_rate': len(successful_executions) / len(self.execution_history),
            'avg_slippage': np.mean([r.slippage for r in successful_executions]),
            'avg_execution_time': np.mean([r.execution_time_ms for r in successful_executions]),
            'total_trades': len(self.execution_history),
            'total_fees': sum(r.fees for r in successful_executions),
            'venue_distribution': self.get_venue_distribution()
        }
        
    def get_venue_distribution(self):
        venue_counts = defaultdict(int)
        
        for result in self.execution_history:
            if result.success:
                venue_counts[result.venue] += 1
                
        total_trades = sum(venue_counts.values())
        
        if total_trades == 0:
            return {}
            
        return {venue: count / total_trades for venue, count in venue_counts.items()}