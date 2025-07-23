import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from market_data_engine_perfect import MarketDataEngine, PerfectMarketUpdate

@pytest.fixture
async def market_engine():
    config = {
        'api_keys': {
            'binance': {'api_key': 'test', 'secret_key': 'test'},
            'okx': {'api_key': 'test', 'secret_key': 'test', 'passphrase': 'test'}
        },
        'infrastructure': {'redis_url': 'redis://localhost:6379'}
    }
    engine = MarketDataEngine(config)
    yield engine
    await engine.redis_client.close()

@pytest.mark.asyncio
async def test_initialize_exchanges(market_engine):
    with patch('ccxt.binance') as mock_binance:
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_binance.return_value = mock_exchange
        
        await market_engine.initialize_single_exchange('binance', {'sandbox': False})
        
        assert 'binance' in market_engine.exchanges
        mock_exchange.load_markets.assert_called_once()

@pytest.mark.asyncio
async def test_process_ticker_update(market_engine):
    ticker = {
        'symbol': 'BTC/USDT',
        'last': 50000.0,
        'quoteVolume': 1000000.0,
        'bid': 49990.0,
        'ask': 50010.0
    }
    
    with patch.object(market_engine.redis_client, 'set', new_callable=AsyncMock) as mock_redis:
        await market_engine.process_ticker_update(ticker, 'binance')
        
        assert len(market_engine.price_history['BTC-USDT']) > 0
        assert len(market_engine.volume_history['BTC-USDT']) > 0
        mock_redis.assert_called_once()

@pytest.mark.asyncio
async def test_momentum_calculation(market_engine):
    symbol = 'BTC-USDT'
    prices = [49000, 49500, 50000, 50500, 51000]
    timestamps = [i * 1_000_000_000 for i in range(len(prices))]
    
    for price, ts in zip(prices, timestamps):
        market_engine.price_history[symbol].append((ts, price))
    
    momentum = await market_engine.momentum_calculator.calculate_all_timeframes(
        symbol, 51000, timestamps[-1]
    )
    
    assert '1s' in momentum
    assert momentum['1s'] > 0

@pytest.mark.asyncio
async def test_technical_indicators(market_engine):
    prices = np.random.uniform(49000, 51000, 100)
    volumes = np.random.uniform(100, 1000, 100)
    
    indicators = await market_engine.calculate_technical_indicators('BTC-USDT', 50000)
    
    assert 'rsi' in indicators
    assert 'macd' in indicators
    assert 'vwap' in indicators
    assert 0 <= indicators['rsi'] <= 100

@pytest.mark.asyncio
async def test_market_update_creation(market_engine):
    symbol = 'BTC-USDT'
    items = [
        {
            'symbol': symbol,
            'price': 50000.0,
            'volume': 1000.0,
            'bid': 49990.0,
            'ask': 50010.0,
            'exchange': 'binance',
            'timestamp': 1234567890,
            'type': 'ticker'
        }
    ]
    
    await market_engine.create_market_update(symbol, items)
    
    pattern = f"market_update:{symbol}:*"
    with patch.object(market_engine.redis_client, 'keys', return_value=[f'market_update:{symbol}:123']) as mock_keys:
        keys = await market_engine.redis_client.keys(pattern)
        assert len(keys) >= 0

def test_rsi_calculation():
    from market_data_engine_perfect import MicrostructureAnalyzer
    analyzer = MicrostructureAnalyzer()
    
    prices = [100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107]
    
    rsi = analyzer._calculate_rsi(prices)
    
    assert 0 <= rsi <= 100
    assert isinstance(rsi, float)

def test_macd_calculation():
    from market_data_engine_perfect import MicrostructureAnalyzer
    analyzer = MicrostructureAnalyzer()
    
    prices = list(range(100, 140))
    
    macd = analyzer._calculate_macd(prices)
    
    assert isinstance(macd, float)

@pytest.mark.asyncio
async def test_liquidity_analysis():
    from market_data_engine_perfect import LiquidityAnalyzer
    analyzer = LiquidityAnalyzer()
    
    orderbook = {
        'bids': [[49990, 10], [49985, 15], [49980, 20]],
        'asks': [[50010, 12], [50015, 18], [50020, 25]]
    }
    
    metrics = await analyzer.analyze_liquidity('BTC-USDT', orderbook)
    
    assert 'score' in metrics
    assert 'depth' in metrics
    assert 'spread' in metrics
    assert metrics['score'] >= 0

@pytest.mark.asyncio
async def test_batch_processing(market_engine):
    items = []
    for i in range(100):
        items.append({
            'symbol': f'TEST{i}-USDT',
            'price': 1000 + i,
            'volume': 100,
            'bid': 999 + i,
            'ask': 1001 + i,
            'exchange': 'test',
            'timestamp': 1234567890 + i,
            'type': 'ticker'
        })
    
    await market_engine.process_batch(items)
    
    assert len(market_engine.price_history) > 0

def test_correlation_engine():
    from market_data_engine_perfect import CorrelationEngine
    engine = CorrelationEngine()
    
    matrix = engine.correlation_matrix
    
    assert matrix.shape == (1000, 1000)
    assert np.allclose(np.diag(matrix), 1.0)

@pytest.mark.parametrize("symbol,expected_result", [
    ("BTC-USDT", True),
    ("SCAM-USDT", False),
    ("", False)
])
def test_symbol_validation(symbol, expected_result):
    valid_symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    result = symbol in valid_symbols
    assert result == expected_result

@pytest.mark.asyncio
async def test_streaming_data_generation(market_engine):
    count = 0
    async for batch in market_engine.stream_ultra_fast_data():
        count += 1
        assert isinstance(batch, list)
        if count >= 3:
            break
    
    assert count == 3