import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

@dataclass
class BacktestResult:
    strategy_name: str
    start_date: datetime
    end_date: datetime
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    best_trade: float
    worst_trade: float
    volatility: float
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    underwater_periods: List[Tuple[datetime, datetime]]
    monthly_returns: pd.Series
    daily_returns: pd.Series
    equity_curve: pd.Series
    trade_log: pd.DataFrame

@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission: float
    slippage: float
    symbols: List[str]
    benchmark: str
    risk_free_rate: float
    rebalance_frequency: str
    position_sizing: str
    max_position_size: float
    stop_loss: Optional[float]
    take_profit: Optional[float]

class AdvancedBacktestingEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.historical_data = {}
        self.current_positions = {}
        self.cash = config.initial_capital
        self.equity_curve = []
        self.trade_log = []
        self.current_date = config.start_date
        
    async def load_historical_data(self, symbols: List[str]):
        data_tasks = []
        for symbol in symbols:
            task = self._fetch_symbol_data(symbol)
            data_tasks.append(task)
            
        results = await asyncio.gather(*data_tasks)
        
        for symbol, data in zip(symbols, results):
            if data is not None:
                self.historical_data[symbol] = data
                
    async def _fetch_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=self.config.start_date,
                end=self.config.end_date,
                interval='1d'
            )
            
            if len(data) > 0:
                data['returns'] = data['Close'].pct_change()
                data['volatility'] = data['returns'].rolling(20).std()
                data['sma_20'] = data['Close'].rolling(20).mean()
                data['sma_50'] = data['Close'].rolling(50).mean()
                data['rsi'] = self._calculate_rsi(data['Close'])
                data['bollinger_upper'], data['bollinger_lower'] = self._calculate_bollinger_bands(data['Close'])
                
                return data
                
        except Exception:
            pass
            
        return None
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
        
    async def run_backtest(self, strategy_func, strategy_name: str) -> BacktestResult:
        self.cash = self.config.initial_capital
        self.current_positions = {}
        self.equity_curve = []
        self.trade_log = []
        
        trading_days = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='D'
        )
        
        for date in trading_days:
            self.current_date = date
            
            current_data = {}
            for symbol, data in self.historical_data.items():
                if date in data.index:
                    current_data[symbol] = data.loc[date]
                    
            if current_data:
                signals = await strategy_func(current_data, self.current_positions)
                await self._process_signals(signals, current_data)
                
            portfolio_value = self._calculate_portfolio_value(current_data)
            self.equity_curve.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'positions_value': portfolio_value - self.cash
            })
            
        return self._generate_backtest_results(strategy_name)
        
    async def _process_signals(self, signals: Dict[str, float], current_data: Dict):
        for symbol, target_weight in signals.items():
            if symbol in current_data:
                current_price = current_data[symbol]['Close']
                current_weight = self._get_current_weight(symbol, current_price)
                
                weight_diff = target_weight - current_weight
                
                if abs(weight_diff) > 0.01:
                    portfolio_value = self._calculate_portfolio_value(current_data)
                    target_value = target_weight * portfolio_value
                    current_value = current_weight * portfolio_value
                    
                    trade_value = target_value - current_value
                    
                    if trade_value > 0:
                        await self._execute_buy(symbol, current_price, abs(trade_value))
                    else:
                        await self._execute_sell(symbol, current_price, abs(trade_value))
                        
    async def _execute_buy(self, symbol: str, price: float, value: float):
        adjusted_price = price * (1 + self.config.slippage)
        shares = value / adjusted_price
        commission = value * self.config.commission
        total_cost = value + commission
        
        if total_cost <= self.cash:
            if symbol in self.current_positions:
                self.current_positions[symbol] += shares
            else:
                self.current_positions[symbol] = shares
                
            self.cash -= total_cost
            
            self.trade_log.append({
                'date': self.current_date,
                'symbol': symbol,
                'action': 'BUY',
                'shares': shares,
                'price': adjusted_price,
                'value': value,
                'commission': commission
            })
            
    async def _execute_sell(self, symbol: str, price: float, value: float):
        if symbol in self.current_positions:
            adjusted_price = price * (1 - self.config.slippage)
            shares_to_sell = min(value / adjusted_price, self.current_positions[symbol])
            
            if shares_to_sell > 0:
                sale_value = shares_to_sell * adjusted_price
                commission = sale_value * self.config.commission
                net_proceeds = sale_value - commission
                
                self.current_positions[symbol] -= shares_to_sell
                if self.current_positions[symbol] <= 0:
                    del self.current_positions[symbol]
                    
                self.cash += net_proceeds
                
                self.trade_log.append({
                    'date': self.current_date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': adjusted_price,
                    'value': sale_value,
                    'commission': commission
                })
                
    def _get_current_weight(self, symbol: str, current_price: float) -> float:
        if symbol in self.current_positions:
            position_value = self.current_positions[symbol] * current_price
            portfolio_value = self._calculate_portfolio_value({symbol: {'Close': current_price}})
            return position_value / portfolio_value if portfolio_value > 0 else 0
        return 0
        
    def _calculate_portfolio_value(self, current_data: Dict) -> float:
        positions_value = 0
        for symbol, shares in self.current_positions.items():
            if symbol in current_data:
                positions_value += shares * current_data[symbol]['Close']
                
        return self.cash + positions_value
        
    def _generate_backtest_results(self, strategy_name: str) -> BacktestResult:
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        trades_df = pd.DataFrame(self.trade_log)
        
        returns = equity_df['portfolio_value'].pct_change().dropna()
        
        total_return = (equity_df['portfolio_value'].iloc[-1] / equity_df['portfolio_value'].iloc[0]) - 1
        
        days_in_backtest = (self.config.end_date - self.config.start_date).days
        annual_return = ((1 + total_return) ** (365.25 / days_in_backtest)) - 1
        
        max_drawdown = self._calculate_max_drawdown(equity_df['portfolio_value'])
        
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        win_rate, profit_factor = self._calculate_trade_statistics(trades_df)
        
        volatility = returns.std() * np.sqrt(252)
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        underwater_periods = self._find_underwater_periods(equity_df['portfolio_value'])
        
        monthly_returns = equity_df['portfolio_value'].resample('M').last().pct_change().dropna()
        
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades_df),
            avg_trade_duration=0,
            best_trade=self._calculate_best_trade(trades_df),
            worst_trade=self._calculate_worst_trade(trades_df),
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            underwater_periods=underwater_periods,
            monthly_returns=monthly_returns,
            daily_returns=returns,
            equity_curve=equity_df['portfolio_value'],
            trade_log=trades_df
        )
        
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())
        
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        excess_returns = returns - (self.config.risk_free_rate / 252)
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        excess_returns = returns - (self.config.risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        return excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
    def _calculate_trade_statistics(self, trades_df: pd.DataFrame) -> Tuple[float, float]:
        if len(trades_df) == 0:
            return 0, 0
            
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        if len(buy_trades) == 0 or len(sell_trades) == 0:
            return 0, 0
            
        winning_trades = 0
        total_profit = 0
        total_loss = 0
        
        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol].sort_values('date')
            
            position = 0
            avg_cost = 0
            
            for _, trade in symbol_trades.iterrows():
                if trade['action'] == 'BUY':
                    if position == 0:
                        avg_cost = trade['price']
                        position = trade['shares']
                    else:
                        total_cost = (position * avg_cost) + (trade['shares'] * trade['price'])
                        position += trade['shares']
                        avg_cost = total_cost / position
                        
                elif trade['action'] == 'SELL' and position > 0:
                    profit = (trade['price'] - avg_cost) * trade['shares']
                    
                    if profit > 0:
                        winning_trades += 1
                        total_profit += profit
                    else:
                        total_loss += abs(profit)
                        
                    position -= trade['shares']
                    
        total_trades = len(trades_df) // 2
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        return win_rate, profit_factor
        
    def _calculate_best_trade(self, trades_df: pd.DataFrame) -> float:
        if len(trades_df) == 0:
            return 0
            
        trade_returns = []
        
        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol].sort_values('date')
            
            buy_price = None
            for _, trade in symbol_trades.iterrows():
                if trade['action'] == 'BUY' and buy_price is None:
                    buy_price = trade['price']
                elif trade['action'] == 'SELL' and buy_price is not None:
                    trade_return = (trade['price'] - buy_price) / buy_price
                    trade_returns.append(trade_return)
                    buy_price = None
                    
        return max(trade_returns) if trade_returns else 0
        
    def _calculate_worst_trade(self, trades_df: pd.DataFrame) -> float:
        if len(trades_df) == 0:
            return 0
            
        trade_returns = []
        
        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol].sort_values('date')
            
            buy_price = None
            for _, trade in symbol_trades.iterrows():
                if trade['action'] == 'BUY' and buy_price is None:
                    buy_price = trade['price']
                elif trade['action'] == 'SELL' and buy_price is not None:
                    trade_return = (trade['price'] - buy_price) / buy_price
                    trade_returns.append(trade_return)
                    buy_price = None
                    
        return min(trade_returns) if trade_returns else 0
        
    def _find_underwater_periods(self, equity_curve: pd.Series) -> List[Tuple[datetime, datetime]]:
        peak = equity_curve.expanding().max()
        underwater = equity_curve < peak
        
        underwater_periods = []
        start_underwater = None
        
        for date, is_underwater in underwater.items():
            if is_underwater and start_underwater is None:
                start_underwater = date
            elif not is_underwater and start_underwater is not None:
                underwater_periods.append((start_underwater, date))
                start_underwater = None
                
        if start_underwater is not None:
            underwater_periods.append((start_underwater, equity_curve.index[-1]))
            
        return underwater_periods

class StrategyOptimizer:
    def __init__(self, backtest_engine: AdvancedBacktestingEngine):
        self.backtest_engine = backtest_engine
        
    async def optimize_parameters(self, strategy_func, parameter_ranges: Dict, optimization_metric: str = 'sharpe_ratio') -> Dict:
        import itertools
        
        parameter_names = list(parameter_ranges.keys())
        parameter_values = list(parameter_ranges.values())
        
        all_combinations = list(itertools.product(*parameter_values))
        
        optimization_tasks = []
        for combination in all_combinations:
            params = dict(zip(parameter_names, combination))
            task = self._run_single_optimization(strategy_func, params, optimization_metric)
            optimization_tasks.append(task)
            
        results = await asyncio.gather(*optimization_tasks)
        
        best_result = max(results, key=lambda x: x['metric_value'])
        
        return {
            'best_parameters': best_result['parameters'],
            'best_metric_value': best_result['metric_value'],
            'all_results': results
        }
        
    async def _run_single_optimization(self, strategy_func, parameters: Dict, metric: str) -> Dict:
        def parameterized_strategy(current_data, positions):
            return strategy_func(current_data, positions, **parameters)
            
        result = await self.backtest_engine.run_backtest(parameterized_strategy, f"Optimization_{parameters}")
        
        metric_value = getattr(result, metric, 0)
        
        return {
            'parameters': parameters,
            'metric_value': metric_value,
            'full_result': result
        }

class MonteCarloSimulator:
    def __init__(self, backtest_result: BacktestResult):
        self.backtest_result = backtest_result
        
    async def run_monte_carlo(self, num_simulations: int = 1000, simulation_days: int = 252) -> Dict:
        daily_returns = self.backtest_result.daily_returns
        
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        
        simulation_tasks = []
        for i in range(num_simulations):
            task = self._run_single_simulation(mean_return, std_return, simulation_days)
            simulation_tasks.append(task)
            
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            simulation_results = await asyncio.gather(*[
                asyncio.get_event_loop().run_in_executor(executor, self._run_simulation_sync, mean_return, std_return, simulation_days)
                for _ in range(num_simulations)
            ])
            
        final_values = [result['final_value'] for result in simulation_results]
        max_drawdowns = [result['max_drawdown'] for result in simulation_results]
        
        return {
            'final_value_percentiles': {
                '5%': np.percentile(final_values, 5),
                '25%': np.percentile(final_values, 25),
                '50%': np.percentile(final_values, 50),
                '75%': np.percentile(final_values, 75),
                '95%': np.percentile(final_values, 95)
            },
            'max_drawdown_percentiles': {
                '5%': np.percentile(max_drawdowns, 5),
                '25%': np.percentile(max_drawdowns, 25),
                '50%': np.percentile(max_drawdowns, 50),
                '75%': np.percentile(max_drawdowns, 75),
                '95%': np.percentile(max_drawdowns, 95)
            },
            'probability_of_loss': len([v for v in final_values if v < 1.0]) / len(final_values),
            'expected_final_value': np.mean(final_values),
            'simulation_results': simulation_results
        }
        
    def _run_simulation_sync(self, mean_return: float, std_return: float, simulation_days: int) -> Dict:
        returns = np.random.normal(mean_return, std_return, simulation_days)
        
        portfolio_values = [1.0]
        for daily_return in returns:
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(new_value)
            
        max_drawdown = 0
        peak = portfolio_values[0]
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return {
            'final_value': portfolio_values[-1],
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values
        }
        
    async def _run_single_simulation(self, mean_return: float, std_return: float, simulation_days: int) -> Dict:
        return self._run_simulation_sync(mean_return, std_return, simulation_days)

class WalkForwardAnalyzer:
    def __init__(self, backtest_engine: AdvancedBacktestingEngine):
        self.backtest_engine = backtest_engine
        
    async def run_walk_forward_analysis(self, strategy_func, training_period_days: int = 252, 
                                      testing_period_days: int = 63, step_size_days: int = 21) -> Dict:
        
        total_days = (self.backtest_engine.config.end_date - self.backtest_engine.config.start_date).days
        
        walk_forward_results = []
        
        current_start = self.backtest_engine.config.start_date
        
        while current_start + timedelta(days=training_period_days + testing_period_days) <= self.backtest_engine.config.end_date:
            training_end = current_start + timedelta(days=training_period_days)
            testing_start = training_end
            testing_end = testing_start + timedelta(days=testing_period_days)
            
            training_config = BacktestConfig(
                start_date=current_start,
                end_date=training_end,
                initial_capital=self.backtest_engine.config.initial_capital,
                commission=self.backtest_engine.config.commission,
                slippage=self.backtest_engine.config.slippage,
                symbols=self.backtest_engine.config.symbols,
                benchmark=self.backtest_engine.config.benchmark,
                risk_free_rate=self.backtest_engine.config.risk_free_rate,
                rebalance_frequency=self.backtest_engine.config.rebalance_frequency,
                position_sizing=self.backtest_engine.config.position_sizing,
                max_position_size=self.backtest_engine.config.max_position_size,
                stop_loss=self.backtest_engine.config.stop_loss,
                take_profit=self.backtest_engine.config.take_profit
            )
            
            testing_config = BacktestConfig(
                start_date=testing_start,
                end_date=testing_end,
                initial_capital=self.backtest_engine.config.initial_capital,
                commission=self.backtest_engine.config.commission,
                slippage=self.backtest_engine.config.slippage,
                symbols=self.backtest_engine.config.symbols,
                benchmark=self.backtest_engine.config.benchmark,
                risk_free_rate=self.backtest_engine.config.risk_free_rate,
                rebalance_frequency=self.backtest_engine.config.rebalance_frequency,
                position_sizing=self.backtest_engine.config.position_sizing,
                max_position_size=self.backtest_engine.config.max_position_size,
                stop_loss=self.backtest_engine.config.stop_loss,
                take_profit=self.backtest_engine.config.take_profit
            )
            
            training_engine = AdvancedBacktestingEngine(training_config)
            await training_engine.load_historical_data(self.backtest_engine.config.symbols)
            
            training_result = await training_engine.run_backtest(strategy_func, f"Training_{current_start}")
            
            testing_engine = AdvancedBacktestingEngine(testing_config)
            await testing_engine.load_historical_data(self.backtest_engine.config.symbols)
            
            testing_result = await testing_engine.run_backtest(strategy_func, f"Testing_{testing_start}")
            
            walk_forward_results.append({
                'training_period': (current_start, training_end),
                'testing_period': (testing_start, testing_end),
                'training_result': training_result,
                'testing_result': testing_result,
                'out_of_sample_sharpe': testing_result.sharpe_ratio,
                'out_of_sample_return': testing_result.total_return,
                'out_of_sample_drawdown': testing_result.max_drawdown
            })
            
            current_start += timedelta(days=step_size_days)
            
        return {
            'walk_forward_results': walk_forward_results,
            'average_out_of_sample_sharpe': np.mean([r['out_of_sample_sharpe'] for r in walk_forward_results]),
            'average_out_of_sample_return': np.mean([r['out_of_sample_return'] for r in walk_forward_results]),
            'average_out_of_sample_drawdown': np.mean([r['out_of_sample_drawdown'] for r in walk_forward_results]),
            'consistency_ratio': len([r for r in walk_forward_results if r['out_of_sample_return'] > 0]) / len(walk_forward_results)
        }

async def sample_momentum_strategy(current_data: Dict, positions: Dict, lookback_period: int = 20, 
                                 momentum_threshold: float = 0.05) -> Dict[str, float]:
    signals = {}
    
    for symbol, data in current_data.items():
        if hasattr(data, 'Close') and hasattr(data, 'sma_20'):
            current_price = data['Close']
            sma = data['sma_20']
            
            if pd.notna(sma) and sma > 0:
                momentum = (current_price - sma) / sma
                
                if momentum > momentum_threshold:
                    signals[symbol] = 0.5
                elif momentum < -momentum_threshold:
                    signals[symbol] = 0.0
                else:
                    signals[symbol] = 0.25
                    
    return signals

async def sample_mean_reversion_strategy(current_data: Dict, positions: Dict, 
                                       bollinger_threshold: float = 0.8) -> Dict[str, float]:
    signals = {}
    
    for symbol, data in current_data.items():
        if (hasattr(data, 'Close') and hasattr(data, 'bollinger_upper') and 
            hasattr(data, 'bollinger_lower')):
            
            current_price = data['Close']
            upper_band = data['bollinger_upper']
            lower_band = data['bollinger_lower']
            
            if pd.notna(upper_band) and pd.notna(lower_band) and upper_band != lower_band:
                position = (current_price - lower_band) / (upper_band - lower_band)
                
                if position > bollinger_threshold:
                    signals[symbol] = 0.0
                elif position < (1 - bollinger_threshold):
                    signals[symbol] = 0.5
                else:
                    signals[symbol] = 0.25
                    
    return signals

async def sample_rsi_strategy(current_data: Dict, positions: Dict, 
                            oversold_threshold: float = 30, overbought_threshold: float = 70) -> Dict[str, float]:
    signals = {}
    
    for symbol, data in current_data.items():
        if hasattr(data, 'rsi'):
            rsi = data['rsi']
            
            if pd.notna(rsi):
                if rsi < oversold_threshold:
                    signals[symbol] = 0.5
                elif rsi > overbought_threshold:
                    signals[symbol] = 0.0
                else:
                    signals[symbol] = 0.25
                    
    return signals