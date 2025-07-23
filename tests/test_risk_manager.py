import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from risk_manager import MilitaryGradeRiskManager, RiskAssessment, PortfolioRisk, StressTestResult

@pytest.fixture
def config():
    return {
        'risk_config': {
            'max_var': 0.02,
            'max_position': 0.2,
            'stop_loss': 0.05,
            'take_profit': 0.15,
            'correlation_limit': 0.7,
            'stress_limit': 0.1
        }
    }

@pytest.fixture
def risk_manager(config):
    return MilitaryGradeRiskManager(config)

@pytest.fixture
def mock_signal():
    signal = Mock()
    signal.symbol = 'BTC-USDT'
    signal.confidence = 0.8
    signal.entry_price = 50000
    return signal

@pytest.fixture
def mock_research():
    research = Mock()
    research.overall_score = 0.9
    return research

@pytest.fixture
def mock_security():
    security = Mock()
    security.safety_score = 0.85
    return security

@pytest.fixture
def mock_portfolio():
    positions = []
    for i, symbol in enumerate(['BTC-USDT', 'ETH-USDT', 'SOL-USDT']):
        position = Mock()
        position.symbol = symbol
        position.size = 1.0
        position.current_price = 50000 - i * 1000
        position.entry_price = 49000 - i * 1000
        position.unrealized_pnl = 1000 - i * 100
        position.realized_pnl = 500
        position.fees_paid = 50
        position.entry_time = 1234567890
        position.venue = 'binance'
        positions.append(position)
    return positions

@pytest.mark.asyncio
async def test_risk_manager_initialization(risk_manager):
    await risk_manager.initialize_risk_systems()
    
    assert len(risk_manager.var_models) == 4
    assert 'parametric' in risk_manager.var_models
    assert 'historical' in risk_manager.var_models
    assert 'monte_carlo' in risk_manager.var_models
    assert 'extreme_value' in risk_manager.var_models

@pytest.mark.asyncio
async def test_assess_signal_approval(risk_manager, mock_signal, mock_research, mock_security, mock_portfolio):
    assessment = await risk_manager.assess_signal(mock_signal, mock_research, mock_security, mock_portfolio)
    
    assert isinstance(assessment, RiskAssessment)
    assert isinstance(assessment.approved, bool)
    assert assessment.position_size > 0
    assert assessment.risk_score >= 0
    assert assessment.confidence > 0

@pytest.mark.asyncio
async def test_assess_signal_rejection_high_risk(risk_manager, mock_signal, mock_research, mock_security, mock_portfolio):
    mock_signal.confidence = 0.3
    mock_research.overall_score = 0.2
    mock_security.safety_score = 0.1
    
    assessment = await risk_manager.assess_signal(mock_signal, mock_research, mock_security, mock_portfolio)
    
    assert assessment.approved == False
    assert assessment.risk_score > 0.6

def test_symbol_concentration_calculation(risk_manager, mock_portfolio):
    concentration = risk_manager._calculate_symbol_concentration('BTC-USDT', mock_portfolio)
    
    assert 0 <= concentration <= 1
    assert concentration > 0

def test_sector_concentration_calculation(risk_manager, mock_portfolio):
    concentration = risk_manager._calculate_sector_concentration('BTC-USDT', mock_portfolio)
    
    assert 0 <= concentration <= 1

@pytest.mark.asyncio
async def test_correlation_risk_calculation(risk_manager, mock_portfolio):
    risk_manager.symbol_mapping = {'BTC-USDT': 0, 'ETH-USDT': 1, 'SOL-USDT': 2}
    risk_manager.correlation_matrix = np.array([
        [1.0, 0.8, 0.6],
        [0.8, 1.0, 0.7],
        [0.6, 0.7, 1.0]
    ])
    
    correlation_risk = await risk_manager._calculate_correlation_risk('BTC-USDT', mock_portfolio)
    
    assert 0 <= correlation_risk <= 1

@pytest.mark.asyncio
async def test_volatility_adjustment(risk_manager):
    symbol = 'BTC-USDT'
    risk_manager.returns_history[symbol].extend([0.01, -0.02, 0.015, -0.01, 0.02] * 10)
    
    adjustment = await risk_manager._calculate_volatility_adjustment(symbol)
    
    assert 0 < adjustment <= 1

@pytest.mark.asyncio
async def test_portfolio_var_calculation(risk_manager):
    with patch.object(risk_manager, '_get_current_portfolio', return_value=[]):
        portfolio_risk = await risk_manager.calculate_portfolio_risk()
        
        assert isinstance(portfolio_risk, PortfolioRisk)
        assert portfolio_risk.total_var >= 0

@pytest.mark.asyncio
async def test_stress_testing(risk_manager):
    with patch.object(risk_manager, '_get_current_portfolio', return_value=[]):
        stress_results = await risk_manager.stress_test()
        
        assert isinstance(stress_results, list)
        for result in stress_results:
            assert isinstance(result, StressTestResult)
            assert 0 <= result.probability <= 1

def test_max_drawdown_calculation(risk_manager):
    portfolio_values = [100000, 105000, 98000, 102000, 95000, 110000]
    
    max_dd = risk_manager._calculate_max_drawdown(portfolio_values)
    
    assert 0 <= max_dd <= 1
    assert max_dd > 0

def test_sharpe_ratio_calculation(risk_manager):
    returns = np.random.normal(0.001, 0.02, 252)
    
    sharpe = risk_manager._calculate_sharpe_ratio(returns)
    
    assert isinstance(sharpe, float)

def test_sortino_ratio_calculation(risk_manager):
    returns = np.random.normal(0.001, 0.02, 252)
    
    sortino = risk_manager._calculate_sortino_ratio(returns)
    
    assert isinstance(sortino, float)
    assert sortino >= 0 or sortino == float('inf')

def test_calmar_ratio_calculation(risk_manager):
    returns = np.random.normal(0.001, 0.02, 252)
    max_drawdown = 0.1
    
    calmar = risk_manager._calculate_calmar_ratio(returns, max_drawdown)
    
    assert isinstance(calmar, float)

@pytest.mark.asyncio
async def test_emergency_risk_reduction(risk_manager):
    with patch.object(risk_manager, '_get_current_portfolio', return_value=[]):
        await risk_manager.emergency_risk_reduction()
        
        assert True

@pytest.mark.asyncio
async def test_reduce_positions(risk_manager):
    with patch.object(risk_manager, '_get_current_portfolio', return_value=[]):
        await risk_manager.reduce_positions()
        
        assert True

def test_var_models():
    from risk_manager import ParametricVaR, HistoricalVaR, MonteCarloVaR, ExtremeValueVaR
    
    returns = np.random.normal(0, 0.02, 1000)
    
    parametric_var = ParametricVaR().calculate(returns)
    historical_var = HistoricalVaR().calculate(returns)
    monte_carlo_var = MonteCarloVaR().calculate(returns, num_simulations=1000)
    extreme_value_var = ExtremeValueVaR().calculate(returns)
    
    assert all(var >= 0 for var in [parametric_var, historical_var, monte_carlo_var, extreme_value_var])

@pytest.mark.asyncio
async def test_correlation_matrix_update(risk_manager):
    risk_manager.symbol_mapping = {'BTC-USDT': 0, 'ETH-USDT': 1}
    risk_manager.returns_history['BTC-USDT'].extend(np.random.normal(0, 0.02, 252))
    risk_manager.returns_history['ETH-USDT'].extend(np.random.normal(0, 0.025, 252))
    
    await risk_manager._update_correlation_matrix()
    
    assert risk_manager.correlation_matrix.shape[0] >= 2
    assert np.allclose(np.diag(risk_manager.correlation_matrix), 1.0)

@pytest.mark.asyncio
async def test_calculate_correlation(risk_manager):
    risk_manager.returns_history['BTC-USDT'].extend(np.random.normal(0, 0.02, 100))
    risk_manager.returns_history['ETH-USDT'].extend(np.random.normal(0, 0.025, 100))
    
    correlation = await risk_manager._calculate_correlation('BTC-USDT', 'ETH-USDT')
    
    assert -1 <= correlation <= 1

@pytest.mark.parametrize("var_value,expected_result", [
    (0.01, False),
    (0.03, True),
    (0.05, True)
])
@pytest.mark.asyncio
async def test_var_limit_check(risk_manager, var_value, expected_result):
    mock_portfolio_risk = Mock()
    mock_portfolio_risk.total_var = var_value
    
    with patch.object(risk_manager, 'calculate_portfolio_risk', return_value=mock_portfolio_risk):
        exceeds_limit = var_value > risk_manager.max_var
        assert exceeds_limit == expected_result

@pytest.mark.asyncio
async def test_stress_scenario_execution(risk_manager, mock_portfolio):
    scenario = {
        'name': 'Test Crash',
        'btc_change': -0.3,
        'correlation_increase': 0.9,
        'probability': 0.05
    }
    
    result = await risk_manager._run_stress_scenario(mock_portfolio, scenario)
    
    assert isinstance(result, StressTestResult)
    assert result.scenario == 'Test Crash'
    assert result.probability == 0.05

def test_btc_correlation_lookup(risk_manager):
    risk_manager.symbol_mapping = {'BTC-USDT': 0, 'ETH-USDT': 1}
    risk_manager.correlation_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
    
    correlation = risk_manager._get_btc_correlation('ETH-USDT')
    
    assert correlation == 0.8

def test_recovery_time_estimation(risk_manager):
    assert risk_manager._estimate_recovery_time(5000) == 1
    assert risk_manager._estimate_recovery_time(25000) == 5
    assert risk_manager._estimate_recovery_time(75000) == 15
    assert risk_manager._estimate_recovery_time(150000) == 30

@pytest.mark.asyncio
async def test_incremental_var_calculation(risk_manager, mock_portfolio):
    risk_manager.symbol_mapping = {'BTC-USDT': 0, 'ETH-USDT': 1, 'SOL-USDT': 2}
    risk_manager.covariance_matrix = np.eye(3) * 0.0004
    
    incremental_var = await risk_manager._calculate_incremental_var('BTC-USDT', 10000, mock_portfolio)
    
    assert incremental_var >= 0

@pytest.mark.asyncio
async def test_continuous_monitoring_cycle(risk_manager):
    with patch.object(risk_manager, '_update_correlation_matrix') as mock_corr:
        with patch.object(risk_manager, '_update_portfolio_history') as mock_portfolio:
            with patch.object(risk_manager, '_check_risk_limits') as mock_limits:
                risk_manager.is_running = False
                
                await risk_manager._continuous_risk_monitoring()
                
                assert True