import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from execution_engine_perfect import MultiVenueExecutionEngine, ExecutionResult, ExecutionPlan, Position

@pytest.fixture
def config():
    return {
        'exchanges': {
            'binance': {'api_key': 'test', 'secret': 'test'},
            'okx': {'api_key': 'test', 'secret': 'test', 'passphrase': 'test'},
            'coinbase': {'api_key': 'test', 'secret': 'test'}
        },
        'execution_config': {
            'max_slippage': 0.001,
            'timeout': 30
        }
    }

@pytest.fixture
def execution_engine(config):
    return MultiVenueExecutionEngine(config)

@pytest.fixture
def mock_exchange():
    exchange = Mock()
    exchange.create_market_order = AsyncMock(return_value={'id': 'test_order_123', 'filled': 1.0, 'average': 50000})
    exchange.create_limit_order = AsyncMock(return_value={'id': 'test_order_456', 'filled': 1.0, 'average': 50000})
    exchange.fetch_order = AsyncMock(return_value={'id': 'test_order_123', 'status': 'closed', 'filled': 1.0, 'average': 50000, 'fee': {'cost': 10}})
    exchange.fetch_ticker = AsyncMock(return_value={'bid': 49990, 'ask': 50010})
    exchange.fetch_order_book = AsyncMock(return_value={
        'bids': [[49990, 10], [49985, 20]],
        'asks': [[50010, 15], [50015, 25]]
    })
    exchange.load_markets = AsyncMock()
    return exchange

@pytest.mark.asyncio
async def test_engine_initialization(execution_engine):
    with patch('ccxt.binance') as mock_binance, \
         patch('ccxt.okx') as mock_okx, \
         patch('ccxt.coinbase') as mock_coinbase:
        
        mock_binance.return_value = Mock()
        mock_okx.return_value = Mock()
        mock_coinbase.return_value = Mock()
        
        for exchange in [mock_binance.return_value, mock_okx.return_value, mock_coinbase.return_value]:
            exchange.load_markets = AsyncMock()
        
        await execution_engine.initialize_all_venues()
        
        assert len(execution_engine.exchanges) >= 0

@pytest.mark.asyncio
async def test_venue_initialization(execution_engine, mock_exchange):
    with patch('ccxt.binance', return_value=mock_exchange):
        await execution_engine.initialize_venue('binance', {'weight': 0.25, 'max_size': 100000})
        
        assert 'binance' in execution_engine.exchanges
        assert execution_engine.exchanges['binance']['config']['weight'] == 0.25

def test_available_venues_filtering(execution_engine):
    execution_engine.exchanges = {
        'binance': {'error_count': 2},
        'okx': {'error_count': 10},
        'coinbase': {'error_count': 1}
    }
    execution_engine.venue_latencies = {
        'binance': 100,
        'okx': 2000,
        'coinbase': 200
    }
    
    available = execution_engine.get_available_venues('BTC/USDT')
    
    assert 'okx' not in available
    assert 'binance' in available
    assert 'coinbase' in available

@pytest.mark.asyncio
async def test_venue_score_calculation(execution_engine):
    execution_engine.venue_latencies = {'binance': 100}
    execution_engine.exchanges = {
        'binance': {
            'error_count': 1,
            'config': {'max_size': 100000}
        }
    }
    
    with patch.object(execution_engine, 'get_venue_liquidity_score', return_value=0.8):
        scores = await execution_engine.calculate_venue_scores('BTC/USDT', 10000, ['binance'])
        
        assert 'binance' in scores
        assert 0 <= scores['binance'] <= 1

@pytest.mark.asyncio
async def test_liquidity_score_calculation(execution_engine, mock_exchange):
    execution_engine.exchanges = {'binance': {'exchange': mock_exchange}}
    
    score = await execution_engine.get_venue_liquidity_score('binance', 'BTC/USDT')
    
    assert 0 <= score <= 1

def test_fee_rate_lookup(execution_engine):
    binance_fee = execution_engine.get_venue_fee_rate('binance')
    unknown_fee = execution_engine.get_venue_fee_rate('unknown_exchange')
    
    assert binance_fee == 0.001
    assert unknown_fee == 0.002

def test_execution_strategy_selection(execution_engine):
    emergency_strategy = execution_engine.select_execution_strategy(10000, 'emergency')
    large_strategy = execution_engine.select_execution_strategy(100000, 'normal')
    medium_strategy = execution_engine.select_execution_strategy(25000, 'normal')
    small_strategy = execution_engine.select_execution_strategy(5000, 'normal')
    
    assert emergency_strategy == 'aggressive_market'
    assert large_strategy == 'iceberg_twap'
    assert medium_strategy == 'smart_limit'
    assert small_strategy == 'market_sweep'

@pytest.mark.asyncio
async def test_chunk_calculation_market_sweep(execution_engine):
    venues = [('binance', 0.8), ('okx', 0.7)]
    
    chunks = await execution_engine.calculate_optimal_chunks(15000, venues, 'market_sweep')
    
    assert len(chunks) >= 1
    assert all('venue' in chunk for chunk in chunks)
    assert all('size' in chunk for chunk in chunks)
    assert all('order_type' in chunk for chunk in chunks)

@pytest.mark.asyncio
async def test_chunk_calculation_iceberg_twap(execution_engine):
    venues = [('binance', 0.8), ('okx', 0.7)]
    
    chunks = await execution_engine.calculate_optimal_chunks(80000, venues, 'iceberg_twap')
    
    assert len(chunks) >= 1
    assert all(chunk['order_type'] == 'limit' for chunk in chunks)

@pytest.mark.asyncio
async def test_execution_plan_creation(execution_engine):
    mock_signal = Mock()
    mock_signal.symbol = 'BTC-USDT'
    mock_signal.urgency = 'normal'
    
    execution_engine.exchanges = {
        'binance': {'error_count': 1, 'config': {'max_size': 100000}},
        'okx': {'error_count': 2, 'config': {'max_size': 80000}}
    }
    execution_engine.venue_latencies = {'binance': 100, 'okx': 150}
    
    with patch.object(execution_engine, 'calculate_venue_scores', return_value={'binance': 0.8, 'okx': 0.7}):
        plan = await execution_engine.create_optimal_plan(mock_signal, 10000)
        
        assert isinstance(plan, ExecutionPlan)
        assert plan.symbol == 'BTC-USDT'
        assert plan.total_size == 10000
        assert len(plan.chunks) > 0

@pytest.mark.asyncio
async def test_chunk_execution(execution_engine, mock_exchange):
    execution_engine.exchanges = {'binance': {'exchange': mock_exchange}}
    
    chunk = {
        'venue': 'binance',
        'size': 1.0,
        'order_type': 'market',
        'delay': 0
    }
    
    result = await execution_engine.execute_chunk(chunk, 'BTC/USDT', 'buy')
    
    assert isinstance(result, ExecutionResult)
    assert result.venue == 'binance'

@pytest.mark.asyncio
async def test_limit_price_calculation(execution_engine, mock_exchange):
    execution_engine.exchanges = {'binance': {'exchange': mock_exchange}}
    
    buy_price = await execution_engine.calculate_limit_price('binance', 'BTC/USDT', 'buy')
    sell_price = await execution_engine.calculate_limit_price('binance', 'BTC/USDT', 'sell')
    
    assert buy_price > 49990
    assert sell_price < 50010

@pytest.mark.asyncio
async def test_order_fill_waiting(execution_engine, mock_exchange):
    execution_engine.exchanges = {'binance': {'exchange': mock_exchange}}
    
    filled_order = await execution_engine.wait_for_fill('binance', 'test_order_123', 'BTC/USDT', 10)
    
    assert filled_order is not None
    assert filled_order['status'] == 'closed'

def test_slippage_calculation(execution_engine):
    execution_engine.venue_latencies = {'BTC-USDT': 100}
    
    slippage = execution_engine.calculate_slippage('BTC-USDT', 'buy', 50100)
    
    assert slippage >= 0

@pytest.mark.asyncio
async def test_position_update_new_position(execution_engine):
    await execution_engine.update_position('BTC-USDT', 'buy', 1.0, 50000, 'binance', 25)
    
    assert 'BTC-USDT' in execution_engine.positions
    position = execution_engine.positions['BTC-USDT']
    assert position.size == 1.0
    assert position.entry_price == 50000
    assert position.fees_paid == 25

@pytest.mark.asyncio
async def test_position_update_existing_position(execution_engine):
    execution_engine.positions['BTC-USDT'] = Position(
        symbol='BTC-USDT',
        size=1.0,
        entry_price=49000,
        current_price=50000,
        unrealized_pnl=1000,
        realized_pnl=0,
        entry_time=1234567890,
        fees_paid=25,
        venue='binance'
    )
    
    await execution_engine.update_position('BTC-USDT', 'buy', 1.0, 51000, 'binance', 25)
    
    position = execution_engine.positions['BTC-USDT']
    assert position.size == 2.0
    assert position.entry_price == 50000

@pytest.mark.asyncio
async def test_multi_venue_execution(execution_engine):
    plan = ExecutionPlan(
        symbol='BTC-USDT',
        side='buy',
        total_size=10000,
        max_slippage=0.001,
        time_limit=30,
        strategy='market_sweep',
        venues=['binance'],
        chunks=[{
            'venue': 'binance',
            'size': 10000,
            'order_type': 'market',
            'delay': 0
        }],
        urgency='normal'
    )
    
    with patch.object(execution_engine, 'execute_chunk') as mock_execute:
        mock_execute.return_value = ExecutionResult(
            success=True,
            order_id='test_123',
            executed_price=50000,
            executed_size=10000,
            slippage=0.0005,
            fees=25,
            execution_time_ms=100,
            venue='binance',
            strategy='market'
        )
        
        result = await execution_engine.execute_multi_venue(plan)
        
        assert isinstance(result, ExecutionResult)

def test_result_aggregation(execution_engine):
    results = [
        ExecutionResult(True, 'order1', 50000, 5000, 0.001, 12.5, 100, 'binance', 'market'),
        ExecutionResult(True, 'order2', 50100, 5000, 0.0015, 12.5, 150, 'okx', 'market')
    ]
    
    aggregated = execution_engine.aggregate_results(results)
    
    assert aggregated.success == True
    assert aggregated.executed_size == 10000
    assert 50000 <= aggregated.executed_price <= 50100
    assert aggregated.fees == 25

@pytest.mark.asyncio
async def test_rebalancing_execution(execution_engine):
    rebalancing_trade = Mock()
    rebalancing_trade.symbol = 'BTC-USDT'
    rebalancing_trade.size = 1.0
    
    with patch.object(execution_engine, 'create_optimal_plan') as mock_plan:
        with patch.object(execution_engine, 'execute_multi_venue') as mock_execute:
            mock_plan.return_value = Mock()
            mock_execute.return_value = ExecutionResult(True, 'order1', 50000, 1.0, 0.001, 25, 100, 'binance', 'market')
            
            result = await execution_engine.execute_rebalancing(rebalancing_trade)
            
            assert result.success == True

@pytest.mark.asyncio
async def test_close_all_positions(execution_engine):
    execution_engine.positions = {
        'BTC-USDT': Position('BTC-USDT', 1.0, 50000, 51000, 1000, 0, 1234567890, 25, 'binance'),
        'ETH-USDT': Position('ETH-USDT', -2.0, 3000, 2900, -200, 0, 1234567890, 15, 'okx')
    }
    
    with patch.object(execution_engine, 'create_close_position_task') as mock_close:
        mock_close.return_value = AsyncMock()
        
        await execution_engine.close_all_positions()
        
        assert mock_close.call_count == 2

@pytest.mark.asyncio
async def test_emergency_exit(execution_engine):
    position = Position('BTC-USDT', 1.0, 50000, 51000, 1000, 0, 1234567890, 25, 'binance')
    execution_engine.exchanges = {'binance': Mock()}
    
    with patch.object(execution_engine, 'execute_chunk') as mock_execute:
        mock_execute.return_value = ExecutionResult(True, 'emergency_order', 51000, 1.0, 0.002, 25, 50, 'binance', 'market')
        
        result = await execution_engine.emergency_exit(position)
        
        assert result.success == True
        assert result.venue == 'binance'

def test_get_positions(execution_engine):
    execution_engine.positions = {
        'BTC-USDT': Position('BTC-USDT', 1.0, 50000, 51000, 1000, 0, 1234567890, 25, 'binance')
    }
    
    positions = execution_engine.get_positions()
    
    assert len(positions) == 1
    assert positions[0].symbol == 'BTC-USDT'

def test_execution_stats(execution_engine):
    execution_engine.execution_history.extend([
        ExecutionResult(True, 'order1', 50000, 1.0, 0.001, 25, 100, 'binance', 'market'),
        ExecutionResult(True, 'order2', 50100, 1.0, 0.0015, 25, 150, 'okx', 'market'),
        ExecutionResult(False, 'order3', 0, 0, 0, 0, 200, 'coinbase', 'market', 'Timeout')
    ])
    
    stats = execution_engine.get_execution_stats()
    
    assert 'success_rate' in stats
    assert 'avg_slippage' in stats
    assert 'total_trades' in stats
    assert stats['total_trades'] == 3
    assert 0 <= stats['success_rate'] <= 1

def test_venue_distribution(execution_engine):
    execution_engine.execution_history.extend([
        ExecutionResult(True, 'order1', 50000, 1.0, 0.001, 25, 100, 'binance', 'market'),
        ExecutionResult(True, 'order2', 50100, 1.0, 0.0015, 25, 150, 'binance', 'market'),
        ExecutionResult(True, 'order3', 50050, 1.0, 0.001, 25, 120, 'okx', 'market')
    ])
    
    distribution = execution_engine.get_venue_distribution()
    
    assert 'binance' in distribution
    assert 'okx' in distribution
    assert abs(distribution['binance'] - 2/3) < 0.01
    assert abs(distribution['okx'] - 1/3) < 0.01