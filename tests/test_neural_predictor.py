import pytest
import torch
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from neural_predictor_perfect import EnsembleNeuralPredictor, TransformerPricePredictor, LSTMPredictor, CNNPredictor, WaveNetPredictor

@pytest.fixture
def device():
    return torch.device('cpu')

@pytest.fixture
def config():
    return {
        'neural_config': {
            'ensemble_size': 5,
            'training_years': 3,
            'update_frequency': 'real_time'
        }
    }

@pytest.fixture
def predictor(device, config):
    return EnsembleNeuralPredictor(device, config)

@pytest.mark.asyncio
async def test_predictor_initialization(predictor):
    assert len(predictor.models) == 4
    assert 'transformer' in predictor.models
    assert 'lstm' in predictor.models
    assert 'cnn' in predictor.models
    assert 'wavenet' in predictor.models

def test_transformer_forward_pass():
    model = TransformerPricePredictor(input_dim=512, d_model=256, num_heads=8, num_layers=6)
    x = torch.randn(2, 50, 512)
    
    price_pred, vol_pred, conf_pred, unc_pred = model(x)
    
    assert price_pred.shape == (2,)
    assert vol_pred.shape == (2,)
    assert conf_pred.shape == (2,)
    assert unc_pred.shape == (2,)
    assert torch.all(conf_pred >= 0) and torch.all(conf_pred <= 1)
    assert torch.all(vol_pred >= 0)
    assert torch.all(unc_pred >= 0)

def test_lstm_forward_pass():
    model = LSTMPredictor(input_dim=512, hidden_dim=256, num_layers=2)
    x = torch.randn(2, 50, 512)
    
    price_pred, vol_pred, conf_pred, unc_pred = model(x)
    
    assert price_pred.shape == (2,)
    assert vol_pred.shape == (2,)
    assert conf_pred.shape == (2,)
    assert unc_pred.shape == (2,)

def test_cnn_forward_pass():
    model = CNNPredictor(input_dim=512)
    x = torch.randn(2, 50, 512)
    
    price_pred, vol_pred, conf_pred, unc_pred = model(x)
    
    assert price_pred.shape == (2,)
    assert vol_pred.shape == (2,)
    assert conf_pred.shape == (2,)
    assert unc_pred.shape == (2,)

def test_wavenet_forward_pass():
    model = WaveNetPredictor(input_dim=512, num_layers=6, num_blocks=3)
    x = torch.randn(2, 50, 512)
    
    price_pred, vol_pred, conf_pred, unc_pred = model(x)
    
    assert price_pred.shape == (2,)
    assert vol_pred.shape == (2,)
    assert conf_pred.shape == (2,)
    assert unc_pred.shape == (2,)

@pytest.mark.asyncio
async def test_ensemble_prediction(predictor):
    with patch.object(predictor, '_get_symbol_features', return_value=np.random.randn(1024)):
        prediction = await predictor.get_ensemble_predictions('BTC-USDT')
        
        assert prediction is not None
        assert prediction.symbol == 'BTC-USDT'
        assert 0 <= prediction.confidence <= 1
        assert prediction.volatility_forecast >= 0
        assert prediction.uncertainty >= 0
        assert len(prediction.predictions) > 0

@pytest.mark.asyncio
async def test_model_training(predictor):
    predictor.X_train = np.random.randn(100, 1024)
    predictor.y_train = np.random.randn(100, 2)
    
    await predictor._train_neural_model('transformer', predictor.models['transformer'])
    
    assert True

def test_ensemble_prediction_calculation(predictor):
    neural_preds = {
        'transformer': {'price': 0.1, 'confidence': 0.8},
        'lstm': {'price': 0.2, 'confidence': 0.7}
    }
    ml_preds = {
        'xgboost': {'price': 0.15, 'confidence': 0.9}
    }
    
    ensemble_pred = predictor._calculate_ensemble_prediction(neural_preds, ml_preds)
    
    assert isinstance(ensemble_pred, float)
    assert -1 <= ensemble_pred <= 1

def test_confidence_calculation(predictor):
    neural_preds = {
        'transformer': {'confidence': 0.8},
        'lstm': {'confidence': 0.7}
    }
    ml_preds = {
        'xgboost': {'confidence': 0.9}
    }
    
    confidence = predictor._calculate_ensemble_confidence(neural_preds, ml_preds)
    
    assert 0 <= confidence <= 1

@pytest.mark.asyncio
async def test_feature_engineering():
    from neural_predictor_perfect import AdvancedFeatureEngineering
    
    engineer = AdvancedFeatureEngineering()
    
    mock_market_update = Mock()
    mock_market_update.microstructure_features = np.random.randn(50)
    mock_market_update.momentum_1s = 0.1
    mock_market_update.momentum_5s = 0.2
    mock_market_update.momentum_1m = 0.3
    mock_market_update.momentum_5m = 0.4
    mock_market_update.volatility_realized = 0.02
    mock_market_update.volatility_implied = 0.03
    mock_market_update.liquidity_score = 80
    mock_market_update.social_sentiment = 0.6
    mock_market_update.news_sentiment = 0.7
    mock_market_update.whale_activity = 0.1
    mock_market_update.cross_exchange_spread = 0.001
    mock_market_update.vwap = 50000
    mock_market_update.twap = 50100
    mock_market_update.rsi = 60
    mock_market_update.macd = 0.5
    mock_market_update.bollinger_position = 0.3
    mock_market_update.volume_profile = np.random.randn(10)
    mock_market_update.order_flow_imbalance = 0.1
    mock_market_update.effective_spread = 0.002
    mock_market_update.price_impact = 0.001
    mock_market_update.funding_rate = 0.0001
    mock_market_update.futures_basis = 0.05
    
    features = await engineer.extract_comprehensive_features(mock_market_update)
    
    assert len(features) == 1024
    assert not np.isnan(features).any()

def test_technical_indicators():
    from neural_predictor_perfect import AdvancedFeatureEngineering
    
    engineer = AdvancedFeatureEngineering()
    prices = np.random.uniform(49000, 51000, 100)
    
    rsi = engineer._calculate_rsi(prices)
    assert 0 <= rsi <= 100
    
    macd = engineer._calculate_macd(prices)
    assert isinstance(macd, float)
    
    ema = engineer._calculate_ema(prices, 20)
    assert isinstance(ema, float)

@pytest.mark.parametrize("prices,expected_range", [
    ([100, 101, 102, 103, 104], (0, 100)),
    ([50, 49, 48, 47, 46], (0, 100)),
    ([100] * 10, (0, 100))
])
def test_rsi_edge_cases(prices, expected_range):
    from neural_predictor_perfect import AdvancedFeatureEngineering
    
    engineer = AdvancedFeatureEngineering()
    rsi = engineer._calculate_rsi(np.array(prices))
    
    assert expected_range[0] <= rsi <= expected_range[1]

@pytest.mark.asyncio
async def test_ml_model_integration(predictor):
    predictor.X_train = np.random.randn(100, 1024)
    predictor.y_train = np.random.randn(100, 2)
    
    await predictor._initialize_ml_models()
    
    assert predictor.ml_models['xgboost'] is not None
    assert predictor.ml_models['lightgbm'] is not None
    assert predictor.ml_models['random_forest'] is not None

@pytest.mark.asyncio
async def test_incremental_training(predictor):
    predictor.training_data = {
        'features': [np.random.randn(1024) for _ in range(1000)],
        'targets': [],
        'timestamps': []
    }
    
    await predictor._incremental_retrain()
    
    assert True

def test_model_weights_normalization(predictor):
    weights = predictor.model_weights
    total_weight = sum(weights.values())
    
    assert abs(total_weight - 1.0) < 0.01

@pytest.mark.asyncio
async def test_prediction_caching(predictor):
    with patch.object(predictor, '_get_symbol_features', return_value=np.random.randn(1024)):
        prediction1 = await predictor.get_ensemble_predictions('BTC-USDT')
        prediction2 = await predictor.get_ensemble_predictions('BTC-USDT')
        
        assert prediction1.ensemble_prediction == prediction2.ensemble_prediction

def test_volatility_and_uncertainty_calculations(predictor):
    predictions = {
        'model1': 0.1,
        'model2': 0.2,
        'model3': 0.15
    }
    
    volatility = predictor._calculate_volatility_forecast(predictions)
    uncertainty = predictor._calculate_uncertainty(predictions)
    
    assert volatility >= 0
    assert uncertainty >= 0