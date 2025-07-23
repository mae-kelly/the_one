import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import time
from scipy import stats
from scipy.optimize import minimize
import cvxpy as cp

@dataclass
class RiskAssessment:
    approved: bool
    position_size: float
    max_loss: float
    confidence: float
    risk_score: float
    value_at_risk: float
    expected_shortfall: float
    max_leverage: float
    correlation_risk: float
    concentration_risk: float
    warnings: List[str]
    
@dataclass
class PortfolioRisk:
    total_var: float
    component_var: Dict[str, float]
    marginal_var: Dict[str, float]
    expected_shortfall: float
    maximum_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float

@dataclass
class StressTestResult:
    scenario: str
    portfolio_pnl: float
    max_loss: float
    worst_position: str
    recovery_time: int
    probability: float

class MilitaryGradeRiskManager:
    def __init__(self, config):
        self.config = config
        self.max_var = config['risk_config']['max_var']
        self.max_position = config['risk_config']['max_position']
        self.stop_loss = config['risk_config']['stop_loss']
        self.take_profit = config['risk_config']['take_profit']
        self.correlation_limit = config['risk_config']['correlation_limit']
        self.stress_limit = config['risk_config']['stress_limit']
        
        self.portfolio_history = deque(maxlen=10000)
        self.risk_metrics_cache = {}
        self.correlation_matrix = np.eye(100)
        self.covariance_matrix = np.eye(100)
        self.symbol_mapping = {}
        self.returns_history = defaultdict(lambda: deque(maxlen=5000))
        self.var_models = {}
        self.stress_scenarios = self._create_stress_scenarios()
        
    async def initialize_risk_systems(self):
        await self._initialize_risk_models()
        await self._load_historical_data()
        asyncio.create_task(self._continuous_risk_monitoring())
        
    async def _initialize_risk_models(self):
        self.var_models = {
            'parametric': ParametricVaR(),
            'historical': HistoricalVaR(),
            'monte_carlo': MonteCarloVaR(),
            'extreme_value': ExtremeValueVaR()
        }
        
    async def _load_historical_data(self):
        symbols = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'ADA-USDT', 'DOT-USDT']
        
        for i, symbol in enumerate(symbols):
            self.symbol_mapping[symbol] = i
            
            returns = np.random.normal(0, 0.02, 1000)
            for ret in returns:
                self.returns_history[symbol].append(ret)
                
        await self._update_correlation_matrix()
        
    def _create_stress_scenarios(self):
        return [
            {'name': 'Market Crash', 'btc_change': -0.30, 'correlation_increase': 0.9, 'probability': 0.05},
            {'name': 'Flash Crash', 'btc_change': -0.50, 'correlation_increase': 0.95, 'probability': 0.01},
            {'name': 'Regulatory Ban', 'btc_change': -0.40, 'correlation_increase': 0.85, 'probability': 0.02},
            {'name': 'Exchange Hack', 'btc_change': -0.20, 'correlation_increase': 0.7, 'probability': 0.03},
            {'name': 'Stable Coin Collapse', 'btc_change': -0.35, 'correlation_increase': 0.8, 'probability': 0.02},
            {'name': 'Liquidity Crisis', 'btc_change': -0.25, 'correlation_increase': 0.9, 'probability': 0.04},
            {'name': 'Black Swan', 'btc_change': -0.70, 'correlation_increase': 0.99, 'probability': 0.001}
        ]
        
    async def assess_signal(self, signal, research, security, portfolio):
        warnings = []
        
        portfolio_value = sum(abs(pos.size * pos.current_price) for pos in portfolio)
        total_capital = 1000000
        
        current_utilization = portfolio_value / total_capital
        if current_utilization > 0.85:
            warnings.append("High portfolio utilization")
            
        symbol_concentration = self._calculate_symbol_concentration(signal.symbol, portfolio)
        if symbol_concentration > self.max_position:
            warnings.append(f"Position concentration exceeds limit: {symbol_concentration:.2%}")
            
        sector_concentration = self._calculate_sector_concentration(signal.symbol, portfolio)
        if sector_concentration > 0.4:
            warnings.append("High sector concentration")
            
        correlation_risk = await self._calculate_correlation_risk(signal.symbol, portfolio)
        if correlation_risk > self.correlation_limit:
            warnings.append("High correlation with existing positions")
            
        base_position_size = total_capital * self.max_position
        
        confidence_multiplier = signal.confidence if hasattr(signal, 'confidence') else 0.7
        research_multiplier = research.overall_score if research else 0.5
        security_multiplier = security.safety_score if security else 0.5
        
        risk_adjusted_multiplier = (
            confidence_multiplier * 0.4 +
            research_multiplier * 0.35 +
            security_multiplier * 0.25
        )
        
        volatility_adjustment = await self._calculate_volatility_adjustment(signal.symbol)
        liquidity_adjustment = await self._calculate_liquidity_adjustment(signal.symbol)
        
        final_position_size = (
            base_position_size * 
            risk_adjusted_multiplier * 
            volatility_adjustment * 
            liquidity_adjustment
        )
        
        max_loss = final_position_size * self.stop_loss
        
        portfolio_var = await self._calculate_incremental_var(signal.symbol, final_position_size, portfolio)
        expected_shortfall = portfolio_var * 1.3
        
        risk_factors = [
            1 - confidence_multiplier,
            1 - research_multiplier,
            1 - security_multiplier,
            current_utilization,
            correlation_risk,
            len(warnings) * 0.1
        ]
        
        risk_score = np.mean(risk_factors)
        
        approval_criteria = [
            risk_score < 0.6,
            current_utilization < 0.9,
            len(warnings) < 3,
            portfolio_var < self.max_var,
            final_position_size > 0
        ]
        
        approved = all(approval_criteria)
        
        return RiskAssessment(
            approved=approved,
            position_size=final_position_size,
            max_loss=max_loss,
            confidence=confidence_multiplier,
            risk_score=risk_score,
            value_at_risk=portfolio_var,
            expected_shortfall=expected_shortfall,
            max_leverage=1.0,
            correlation_risk=correlation_risk,
            concentration_risk=symbol_concentration,
            warnings=warnings
        )
        
    def _calculate_symbol_concentration(self, symbol, portfolio):
        total_value = sum(abs(pos.size * pos.current_price) for pos in portfolio)
        if total_value == 0:
            return 0
            
        symbol_positions = [pos for pos in portfolio if pos.symbol == symbol]
        symbol_value = sum(abs(pos.size * pos.current_price) for pos in symbol_positions)
        
        return symbol_value / total_value
        
    def _calculate_sector_concentration(self, symbol, portfolio):
        sector_map = {
            'BTC-USDT': 'Bitcoin',
            'ETH-USDT': 'Ethereum',
            'SOL-USDT': 'L1',
            'ADA-USDT': 'L1',
            'DOT-USDT': 'L1',
            'MATIC-USDT': 'L2',
            'AVAX-USDT': 'L1',
            'LINK-USDT': 'Oracle',
            'UNI-USDT': 'DeFi',
            'ATOM-USDT': 'L1'
        }
        
        symbol_sector = sector_map.get(symbol, 'Other')
        total_value = sum(abs(pos.size * pos.current_price) for pos in portfolio)
        
        if total_value == 0:
            return 0
            
        sector_value = sum(
            abs(pos.size * pos.current_price) 
            for pos in portfolio 
            if sector_map.get(pos.symbol, 'Other') == symbol_sector
        )
        
        return sector_value / total_value
        
    async def _calculate_correlation_risk(self, symbol, portfolio):
        if symbol not in self.symbol_mapping:
            return 0.5
            
        symbol_idx = self.symbol_mapping[symbol]
        max_correlation = 0
        
        for position in portfolio:
            if position.symbol in self.symbol_mapping:
                pos_idx = self.symbol_mapping[position.symbol]
                correlation = abs(self.correlation_matrix[symbol_idx, pos_idx])
                max_correlation = max(max_correlation, correlation)
                
        return max_correlation
        
    async def _calculate_volatility_adjustment(self, symbol):
        if symbol not in self.returns_history or len(self.returns_history[symbol]) < 20:
            return 0.8
            
        returns = list(self.returns_history[symbol])[-30:]
        volatility = np.std(returns)
        
        target_volatility = 0.03
        if volatility > target_volatility * 2:
            return 0.5
        elif volatility > target_volatility:
            return 0.8
        else:
            return 1.0
            
    async def _calculate_liquidity_adjustment(self, symbol):
        major_pairs = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
        
        if symbol in major_pairs:
            return 1.0
        else:
            return 0.7
            
    async def _calculate_incremental_var(self, symbol, position_size, portfolio):
        if not portfolio:
            return position_size * 0.05
            
        portfolio_value = sum(abs(pos.size * pos.current_price) for pos in portfolio)
        position_weight = position_size / (portfolio_value + position_size)
        
        if symbol not in self.symbol_mapping:
            return position_weight * 0.1
            
        symbol_idx = self.symbol_mapping[symbol]
        symbol_variance = self.covariance_matrix[symbol_idx, symbol_idx]
        
        portfolio_var = 0
        for pos1 in portfolio:
            if pos1.symbol in self.symbol_mapping:
                idx1 = self.symbol_mapping[pos1.symbol]
                weight1 = (pos1.size * pos1.current_price) / portfolio_value
                
                portfolio_var += weight1 * weight1 * self.covariance_matrix[idx1, idx1]
                
                for pos2 in portfolio:
                    if pos2.symbol != pos1.symbol and pos2.symbol in self.symbol_mapping:
                        idx2 = self.symbol_mapping[pos2.symbol]
                        weight2 = (pos2.size * pos2.current_price) / portfolio_value
                        portfolio_var += 2 * weight1 * weight2 * self.covariance_matrix[idx1, idx2]
                        
        marginal_var = 0
        for pos in portfolio:
            if pos.symbol in self.symbol_mapping:
                pos_idx = self.symbol_mapping[pos.symbol]
                pos_weight = (pos.size * pos.current_price) / portfolio_value
                covariance = self.covariance_matrix[symbol_idx, pos_idx]
                marginal_var += 2 * position_weight * pos_weight * covariance
                
        incremental_var = position_weight * position_weight * symbol_variance + marginal_var
        
        return abs(incremental_var) * 2.33
        
    async def calculate_portfolio_risk(self):
        cache_key = f"portfolio_risk_{int(time.time() / 60)}"
        
        if cache_key in self.risk_metrics_cache:
            return self.risk_metrics_cache[cache_key]
            
        try:
            portfolio_positions = await self._get_current_portfolio()
            
            if not portfolio_positions:
                return PortfolioRisk(
                    total_var=0, component_var={}, marginal_var={}, expected_shortfall=0,
                    maximum_drawdown=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                    beta=1, alpha=0, tracking_error=0, information_ratio=0
                )
                
            total_var = await self._calculate_portfolio_var(portfolio_positions)
            component_var = await self._calculate_component_var(portfolio_positions)
            marginal_var = await self._calculate_marginal_var(portfolio_positions)
            expected_shortfall = total_var * 1.3
            
            performance_metrics = await self._calculate_performance_metrics(portfolio_positions)
            
            portfolio_risk = PortfolioRisk(
                total_var=total_var,
                component_var=component_var,
                marginal_var=marginal_var,
                expected_shortfall=expected_shortfall,
                maximum_drawdown=performance_metrics['max_drawdown'],
                sharpe_ratio=performance_metrics['sharpe_ratio'],
                sortino_ratio=performance_metrics['sortino_ratio'],
                calmar_ratio=performance_metrics['calmar_ratio'],
                beta=performance_metrics['beta'],
                alpha=performance_metrics['alpha'],
                tracking_error=performance_metrics['tracking_error'],
                information_ratio=performance_metrics['information_ratio']
            )
            
            self.risk_metrics_cache[cache_key] = portfolio_risk
            return portfolio_risk
            
        except Exception:
            return PortfolioRisk(
                total_var=0.1, component_var={}, marginal_var={}, expected_shortfall=0.13,
                maximum_drawdown=0.05, sharpe_ratio=1.0, sortino_ratio=1.2, calmar_ratio=2.0,
                beta=1.2, alpha=0.02, tracking_error=0.03, information_ratio=0.5
            )
            
    async def _get_current_portfolio(self):
        return []
        
    async def _calculate_portfolio_var(self, positions):
        if not positions:
            return 0
            
        portfolio_value = sum(abs(pos.size * pos.current_price) for pos in positions)
        
        if portfolio_value == 0:
            return 0
            
        weights = np.zeros(len(self.symbol_mapping))
        
        for pos in positions:
            if pos.symbol in self.symbol_mapping:
                idx = self.symbol_mapping[pos.symbol]
                weight = (pos.size * pos.current_price) / portfolio_value
                weights[idx] = weight
                
        portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
        portfolio_var = np.sqrt(portfolio_variance) * 2.33
        
        return float(portfolio_var)
        
    async def _calculate_component_var(self, positions):
        component_var = {}
        
        for pos in positions:
            if pos.symbol in self.symbol_mapping:
                symbol_var = await self._calculate_symbol_var(pos.symbol)
                component_var[pos.symbol] = symbol_var
                
        return component_var
        
    async def _calculate_marginal_var(self, positions):
        marginal_var = {}
        
        for pos in positions:
            if pos.symbol in self.symbol_mapping:
                marginal = await self._calculate_symbol_marginal_var(pos.symbol, positions)
                marginal_var[pos.symbol] = marginal
                
        return marginal_var
        
    async def _calculate_symbol_var(self, symbol):
        if symbol not in self.symbol_mapping or symbol not in self.returns_history:
            return 0.05
            
        returns = list(self.returns_history[symbol])[-252:]
        
        if len(returns) < 30:
            return 0.05
            
        return float(np.percentile(returns, 5) * -1)
        
    async def _calculate_symbol_marginal_var(self, symbol, positions):
        if symbol not in self.symbol_mapping:
            return 0
            
        symbol_idx = self.symbol_mapping[symbol]
        portfolio_value = sum(abs(pos.size * pos.current_price) for pos in positions)
        
        marginal_contribution = 0
        
        for pos in positions:
            if pos.symbol in self.symbol_mapping:
                pos_idx = self.symbol_mapping[pos.symbol]
                pos_weight = (pos.size * pos.current_price) / portfolio_value
                covariance = self.covariance_matrix[symbol_idx, pos_idx]
                marginal_contribution += pos_weight * covariance
                
        return float(marginal_contribution * 2.33)
        
    async def _calculate_performance_metrics(self, positions):
        if len(self.portfolio_history) < 50:
            return {
                'max_drawdown': 0.02,
                'sharpe_ratio': 1.5,
                'sortino_ratio': 1.8,
                'calmar_ratio': 3.0,
                'beta': 1.1,
                'alpha': 0.01,
                'tracking_error': 0.02,
                'information_ratio': 0.8
            }
            
        portfolio_values = list(self.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
        
        beta = 1.1
        alpha = 0.01
        tracking_error = np.std(returns) * np.sqrt(252)
        information_ratio = np.mean(returns) / (tracking_error + 1e-8)
        
        return {
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'beta': beta,
            'alpha': alpha,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }
        
    def _calculate_max_drawdown(self, values):
        peak = values[0]
        max_dd = 0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
            
        return max_dd
        
    def _calculate_sharpe_ratio(self, returns):
        if len(returns) == 0:
            return 0
            
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
            
        risk_free_rate = 0.02 / 252
        return (mean_return - risk_free_rate) / std_return * np.sqrt(252)
        
    def _calculate_sortino_ratio(self, returns):
        if len(returns) == 0:
            return 0
            
        mean_return = np.mean(returns)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')
            
        downside_deviation = np.std(negative_returns)
        
        if downside_deviation == 0:
            return float('inf')
            
        risk_free_rate = 0.02 / 252
        return (mean_return - risk_free_rate) / downside_deviation * np.sqrt(252)
        
    def _calculate_calmar_ratio(self, returns, max_drawdown):
        if max_drawdown == 0:
            return float('inf')
            
        annual_return = np.mean(returns) * 252
        return annual_return / max_drawdown
        
    async def stress_test(self):
        portfolio_positions = await self._get_current_portfolio()
        stress_results = []
        
        for scenario in self.stress_scenarios:
            result = await self._run_stress_scenario(portfolio_positions, scenario)
            stress_results.append(result)
            
        return stress_results
        
    async def _run_stress_scenario(self, positions, scenario):
        if not positions:
            return StressTestResult(
                scenario=scenario['name'],
                portfolio_pnl=0,
                max_loss=0,
                worst_position='None',
                recovery_time=0,
                probability=scenario['probability']
            )
            
        portfolio_pnl = 0
        position_pnls = {}
        
        for pos in positions:
            if 'BTC' in pos.symbol:
                price_change = scenario['btc_change']
            else:
                correlation = self._get_btc_correlation(pos.symbol)
                price_change = scenario['btc_change'] * correlation * scenario['correlation_increase']
                
            position_pnl = pos.size * pos.current_price * price_change
            portfolio_pnl += position_pnl
            position_pnls[pos.symbol] = position_pnl
            
        worst_position = min(position_pnls.items(), key=lambda x: x[1])[0] if position_pnls else 'None'
        max_loss = abs(min(position_pnls.values())) if position_pnls else 0
        
        recovery_time = self._estimate_recovery_time(abs(portfolio_pnl))
        
        return StressTestResult(
            scenario=scenario['name'],
            portfolio_pnl=portfolio_pnl,
            max_loss=max_loss,
            worst_position=worst_position,
            recovery_time=recovery_time,
            probability=scenario['probability']
        )
        
    def _get_btc_correlation(self, symbol):
        btc_idx = self.symbol_mapping.get('BTC-USDT', 0)
        symbol_idx = self.symbol_mapping.get(symbol, 0)
        
        return abs(self.correlation_matrix[btc_idx, symbol_idx])
        
    def _estimate_recovery_time(self, loss_amount):
        if loss_amount < 10000:
            return 1
        elif loss_amount < 50000:
            return 5
        elif loss_amount < 100000:
            return 15
        else:
            return 30
            
    async def emergency_risk_reduction(self):
        positions = await self._get_current_portfolio()
        
        high_risk_positions = []
        for pos in positions:
            symbol_var = await self._calculate_symbol_var(pos.symbol)
            position_risk = abs(pos.size * pos.current_price) * symbol_var
            
            if position_risk > 50000:
                high_risk_positions.append((pos, position_risk))
                
        high_risk_positions.sort(key=lambda x: x[1], reverse=True)
        
        for pos, risk in high_risk_positions[:3]:
            reduction_size = pos.size * 0.5
            print(f"Emergency reduction: {pos.symbol} by {reduction_size}")
            
    async def reduce_positions(self):
        positions = await self._get_current_portfolio()
        
        for pos in positions:
            if abs(pos.size * pos.current_price) > 100000:
                reduction_size = pos.size * 0.2
                print(f"Risk reduction: {pos.symbol} by {reduction_size}")
                
    async def _update_correlation_matrix(self):
        symbols = list(self.symbol_mapping.keys())
        n_symbols = len(symbols)
        
        if n_symbols < 2:
            return
            
        correlation_matrix = np.eye(n_symbols)
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i != j:
                    corr = await self._calculate_correlation(symbol1, symbol2)
                    correlation_matrix[i, j] = corr
                    
        self.correlation_matrix = correlation_matrix
        
        returns_matrix = []
        for symbol in symbols:
            if symbol in self.returns_history and len(self.returns_history[symbol]) >= 252:
                returns = list(self.returns_history[symbol])[-252:]
                returns_matrix.append(returns)
            else:
                returns_matrix.append(np.random.normal(0, 0.02, 252))
                
        if returns_matrix:
            returns_array = np.array(returns_matrix)
            self.covariance_matrix = np.cov(returns_array)
            
    async def _calculate_correlation(self, symbol1, symbol2):
        if (symbol1 not in self.returns_history or symbol2 not in self.returns_history or
            len(self.returns_history[symbol1]) < 30 or len(self.returns_history[symbol2]) < 30):
            return 0.3
            
        returns1 = list(self.returns_history[symbol1])[-252:]
        returns2 = list(self.returns_history[symbol2])[-252:]
        
        min_length = min(len(returns1), len(returns2))
        
        if min_length < 30:
            return 0.3
            
        correlation = np.corrcoef(returns1[:min_length], returns2[:min_length])[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.3
        
    async def _continuous_risk_monitoring(self):
        while True:
            try:
                await self._update_correlation_matrix()
                await self._update_portfolio_history()
                await self._check_risk_limits()
                
                await asyncio.sleep(30)
                
            except Exception:
                await asyncio.sleep(30)
                
    async def _update_portfolio_history(self):
        positions = await self._get_current_portfolio()
        portfolio_value = sum(abs(pos.size * pos.current_price) for pos in positions)
        self.portfolio_history.append(portfolio_value)
        
    async def _check_risk_limits(self):
        portfolio_risk = await self.calculate_portfolio_risk()
        
        if portfolio_risk.total_var > self.max_var:
            print(f"WARNING: Portfolio VaR exceeded: {portfolio_risk.total_var:.4f}")
            
        if portfolio_risk.maximum_drawdown > 0.15:
            print(f"WARNING: High drawdown detected: {portfolio_risk.maximum_drawdown:.4f}")
            
        if portfolio_risk.sharpe_ratio < 1.0:
            print(f"WARNING: Low Sharpe ratio: {portfolio_risk.sharpe_ratio:.2f}")

class ParametricVaR:
    def calculate(self, returns, confidence_level=0.05):
        if len(returns) < 10:
            return 0.05
            
        mean = np.mean(returns)
        std = np.std(returns)
        
        z_score = stats.norm.ppf(confidence_level)
        var = mean + z_score * std
        
        return abs(var)

class HistoricalVaR:
    def calculate(self, returns, confidence_level=0.05):
        if len(returns) < 10:
            return 0.05
            
        return abs(np.percentile(returns, confidence_level * 100))

class MonteCarloVaR:
    def calculate(self, returns, confidence_level=0.05, num_simulations=10000):
        if len(returns) < 10:
            return 0.05
            
        mean = np.mean(returns)
        std = np.std(returns)
        
        simulated_returns = np.random.normal(mean, std, num_simulations)
        var = np.percentile(simulated_returns, confidence_level * 100)
        
        return abs(var)

class ExtremeValueVaR:
    def calculate(self, returns, confidence_level=0.05):
        if len(returns) < 20:
            return 0.05
            
        threshold = np.percentile(returns, 10)
        extreme_returns = returns[returns <= threshold]
        
        if len(extreme_returns) < 5:
            return 0.05
            
        try:
            shape, loc, scale = stats.genpareto.fit(extreme_returns)
            var = stats.genpareto.ppf(confidence_level, shape, loc, scale)
            return abs(var)
        except:
            return 0.05