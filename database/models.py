from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import asyncpg
import asyncio

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    size = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    executed_price = Column(Float)
    executed_size = Column(Float)
    fees = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    execution_time_ms = Column(Integer, default=0)
    venue = Column(String(50), index=True)
    strategy = Column(String(50))
    order_id = Column(String(100), unique=True)
    signal_id = Column(UUID(as_uuid=True), ForeignKey('signals.id'))
    pnl = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    success = Column(Boolean, default=False)
    error_message = Column(Text)
    metadata = Column(JSONB)
    
    signal = relationship("Signal", back_populates="trades")
    
    __table_args__ = (
        Index('idx_trades_timestamp_symbol', 'timestamp', 'symbol'),
        Index('idx_trades_venue_success', 'venue', 'success'),
        Index('idx_trades_pnl', 'pnl'),
    )

class Signal(Base):
    __tablename__ = 'signals'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    signal_type = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    momentum = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    target_price = Column(Float)
    stop_loss = Column(Float)
    features = Column(JSONB)
    research_score = Column(Float)
    security_score = Column(Float)
    risk_score = Column(Float)
    position_size = Column(Float)
    urgency = Column(String(20), default='normal')
    status = Column(String(20), default='pending')
    executed_at = Column(DateTime)
    closed_at = Column(DateTime)
    final_pnl = Column(Float)
    
    trades = relationship("Trade", back_populates="signal")
    
    __table_args__ = (
        Index('idx_signals_confidence_momentum', 'confidence', 'momentum'),
        Index('idx_signals_status', 'status'),
    )

class Position(Base):
    __tablename__ = 'positions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    fees_paid = Column(Float, default=0.0)
    venue = Column(String(50))
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime)
    is_open = Column(Boolean, default=True, index=True)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    risk_score = Column(Float)
    
    __table_args__ = (
        Index('idx_positions_open_symbol', 'is_open', 'symbol'),
        Index('idx_positions_pnl', 'unrealized_pnl'),
    )

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(50), nullable=False, index=True)
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    bid = Column(Float)
    ask = Column(Float)
    spread = Column(Float)
    momentum_1s = Column(Float)
    momentum_5s = Column(Float)
    momentum_1m = Column(Float)
    momentum_5m = Column(Float)
    volatility = Column(Float)
    liquidity_score = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    bollinger_position = Column(Float)
    sentiment = Column(Float)
    whale_activity = Column(Float)
    orderbook_data = Column(JSONB)
    
    __table_args__ = (
        Index('idx_market_data_timestamp_symbol', 'timestamp', 'symbol'),
        Index('idx_market_data_momentum', 'momentum_1m'),
    )

class RiskMetrics(Base):
    __tablename__ = 'risk_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    portfolio_var = Column(Float, nullable=False)
    expected_shortfall = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    calmar_ratio = Column(Float)
    portfolio_value = Column(Float)
    num_positions = Column(Integer)
    correlation_risk = Column(Float)
    concentration_risk = Column(Float)
    stress_test_results = Column(JSONB)
    
    __table_args__ = (
        Index('idx_risk_metrics_timestamp', 'timestamp'),
        Index('idx_risk_metrics_var', 'portfolio_var'),
    )

class SecurityScan(Base):
    __tablename__ = 'security_scans'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    contract_address = Column(String(100))
    is_safe = Column(Boolean, nullable=False, index=True)
    threat_level = Column(Float, nullable=False)
    honeypot_risk = Column(Float)
    rugpull_risk = Column(Float)
    contract_risk = Column(Float)
    liquidity_risk = Column(Float)
    team_risk = Column(Float)
    safety_score = Column(Float)
    risk_reasons = Column(JSONB)
    scan_results = Column(JSONB)
    
    __table_args__ = (
        Index('idx_security_scans_safe_threat', 'is_safe', 'threat_level'),
        Index('idx_security_scans_contract', 'contract_address'),
    )

class ResearchData(Base):
    __tablename__ = 'research_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    overall_score = Column(Float, nullable=False)
    fundamental_score = Column(Float)
    technical_score = Column(Float)
    sentiment_score = Column(Float)
    security_score = Column(Float)
    team_score = Column(Float)
    tokenomics_score = Column(Float)
    liquidity_score = Column(Float)
    adoption_score = Column(Float)
    competitive_score = Column(Float)
    confidence = Column(Float)
    risk_flags = Column(JSONB)
    bullish_signals = Column(JSONB)
    bearish_signals = Column(JSONB)
    research_details = Column(JSONB)
    
    __table_args__ = (
        Index('idx_research_data_score', 'overall_score'),
        Index('idx_research_data_timestamp_symbol', 'timestamp', 'symbol'),
    )

class PerformanceMetrics(Base):
    __tablename__ = 'performance_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    total_return = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    sortino_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    calmar_ratio = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    avg_trade_duration = Column(Float, default=0.0)
    best_trade = Column(Float, default=0.0)
    worst_trade = Column(Float, default=0.0)
    current_streak = Column(Integer, default=0)
    avg_daily_return = Column(Float, default=0.0)
    volatility = Column(Float, default=0.0)
    alpha = Column(Float, default=0.0)
    beta = Column(Float, default=1.0)
    portfolio_value = Column(Float, default=0.0)
    
    __table_args__ = (
        Index('idx_performance_metrics_timestamp', 'timestamp'),
        Index('idx_performance_metrics_returns', 'total_return'),
    )

class SystemLogs(Base):
    __tablename__ = 'system_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(20), nullable=False, index=True)
    component = Column(String(50), nullable=False, index=True)
    message = Column(Text, nullable=False)
    metadata = Column(JSONB)
    trace_id = Column(String(100), index=True)
    
    __table_args__ = (
        Index('idx_system_logs_level_component', 'level', 'component'),
        Index('idx_system_logs_timestamp_level', 'timestamp', 'level'),
    )

class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.postgres_url = config['database']['postgres_url']
        self.clickhouse_url = config['database']['clickhouse_url']
        self.redis_url = config['database']['redis_url']
        
        self.engine = None
        self.session_factory = None
        self.pool = None
        
    async def initialize(self):
        self.engine = create_engine(self.postgres_url, echo=False, pool_size=20, max_overflow=30)
        self.session_factory = sessionmaker(bind=self.engine)
        
        Base.metadata.create_all(self.engine)
        
        self.pool = await asyncpg.create_pool(
            self.postgres_url,
            min_size=10,
            max_size=50,
            command_timeout=60
        )
        
        await self._create_clickhouse_tables()
        print("✅ Database initialized")
        
    async def _create_clickhouse_tables(self):
        clickhouse_schemas = {
            'trades_analytics': '''
                CREATE TABLE IF NOT EXISTS trades_analytics (
                    timestamp DateTime64(3),
                    symbol String,
                    side String,
                    size Float64,
                    price Float64,
                    pnl Float64,
                    venue String,
                    execution_time_ms UInt32,
                    success UInt8,
                    date Date MATERIALIZED toDate(timestamp)
                ) ENGINE = MergeTree()
                PARTITION BY date
                ORDER BY (timestamp, symbol)
                TTL timestamp + INTERVAL 2 YEAR
            ''',
            
            'market_data_analytics': '''
                CREATE TABLE IF NOT EXISTS market_data_analytics (
                    timestamp DateTime64(3),
                    symbol String,
                    exchange String,
                    price Float64,
                    volume Float64,
                    momentum_1m Float64,
                    volatility Float64,
                    liquidity_score Float64,
                    sentiment Float64,
                    date Date MATERIALIZED toDate(timestamp)
                ) ENGINE = MergeTree()
                PARTITION BY date
                ORDER BY (timestamp, symbol, exchange)
                TTL timestamp + INTERVAL 1 YEAR
            ''',
            
            'performance_analytics': '''
                CREATE TABLE IF NOT EXISTS performance_analytics (
                    timestamp DateTime64(3),
                    portfolio_value Float64,
                    total_return Float64,
                    sharpe_ratio Float64,
                    max_drawdown Float64,
                    num_trades UInt32,
                    win_rate Float64,
                    date Date MATERIALIZED toDate(timestamp)
                ) ENGINE = MergeTree()
                PARTITION BY date
                ORDER BY timestamp
                TTL timestamp + INTERVAL 5 YEAR
            '''
        }
        
        print("✅ ClickHouse tables created")
        
    async def insert_trade(self, trade_data: Dict[str, Any]) -> str:
        async with self.pool.acquire() as conn:
            trade_id = str(uuid.uuid4())
            
            await conn.execute('''
                INSERT INTO trades (
                    id, timestamp, symbol, side, size, price, executed_price,
                    executed_size, fees, slippage, execution_time_ms, venue,
                    strategy, order_id, pnl, pnl_pct, success, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
            ''', 
                trade_id,
                trade_data.get('timestamp', datetime.utcnow()),
                trade_data['symbol'],
                trade_data['side'],
                trade_data['size'],
                trade_data['price'],
                trade_data.get('executed_price'),
                trade_data.get('executed_size'),
                trade_data.get('fees', 0.0),
                trade_data.get('slippage', 0.0),
                trade_data.get('execution_time_ms', 0),
                trade_data.get('venue'),
                trade_data.get('strategy'),
                trade_data.get('order_id'),
                trade_data.get('pnl', 0.0),
                trade_data.get('pnl_pct', 0.0),
                trade_data.get('success', False),
                trade_data.get('metadata', {})
            )
            
            await self._insert_clickhouse_trade(trade_data)
            return trade_id
            
    async def _insert_clickhouse_trade(self, trade_data: Dict[str, Any]):
        pass
        
    async def insert_signal(self, signal_data: Dict[str, Any]) -> str:
        async with self.pool.acquire() as conn:
            signal_id = str(uuid.uuid4())
            
            await conn.execute('''
                INSERT INTO signals (
                    id, timestamp, symbol, signal_type, confidence, momentum,
                    entry_price, target_price, stop_loss, features, research_score,
                    security_score, risk_score, position_size, urgency, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            ''',
                signal_id,
                signal_data.get('timestamp', datetime.utcnow()),
                signal_data['symbol'],
                signal_data['signal_type'],
                signal_data['confidence'],
                signal_data['momentum'],
                signal_data['entry_price'],
                signal_data.get('target_price'),
                signal_data.get('stop_loss'),
                signal_data.get('features', {}),
                signal_data.get('research_score'),
                signal_data.get('security_score'),
                signal_data.get('risk_score'),
                signal_data.get('position_size'),
                signal_data.get('urgency', 'normal'),
                signal_data.get('status', 'pending')
            )
            
            return signal_id
            
    async def update_position(self, position_data: Dict[str, Any]):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO positions (
                    symbol, size, entry_price, current_price, unrealized_pnl,
                    realized_pnl, fees_paid, venue, is_open, stop_loss,
                    take_profit, risk_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (symbol) WHERE is_open = true
                DO UPDATE SET
                    size = EXCLUDED.size,
                    current_price = EXCLUDED.current_price,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    realized_pnl = EXCLUDED.realized_pnl,
                    fees_paid = EXCLUDED.fees_paid
            ''',
                position_data['symbol'],
                position_data['size'],
                position_data['entry_price'],
                position_data['current_price'],
                position_data.get('unrealized_pnl', 0.0),
                position_data.get('realized_pnl', 0.0),
                position_data.get('fees_paid', 0.0),
                position_data.get('venue'),
                position_data.get('is_open', True),
                position_data.get('stop_loss'),
                position_data.get('take_profit'),
                position_data.get('risk_score')
            )
            
    async def insert_market_data(self, market_data: Dict[str, Any]):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO market_data (
                    timestamp, symbol, exchange, price, volume, bid, ask, spread,
                    momentum_1s, momentum_5s, momentum_1m, momentum_5m, volatility,
                    liquidity_score, rsi, macd, bollinger_position, sentiment,
                    whale_activity, orderbook_data
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
            ''',
                market_data.get('timestamp', datetime.utcnow()),
                market_data['symbol'],
                market_data['exchange'],
                market_data['price'],
                market_data['volume'],
                market_data.get('bid'),
                market_data.get('ask'),
                market_data.get('spread'),
                market_data.get('momentum_1s'),
                market_data.get('momentum_5s'),
                market_data.get('momentum_1m'),
                market_data.get('momentum_5m'),
                market_data.get('volatility'),
                market_data.get('liquidity_score'),
                market_data.get('rsi'),
                market_data.get('macd'),
                market_data.get('bollinger_position'),
                market_data.get('sentiment'),
                market_data.get('whale_activity'),
                market_data.get('orderbook_data', {})
            )
            
    async def insert_risk_metrics(self, risk_data: Dict[str, Any]):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO risk_metrics (
                    timestamp, portfolio_var, expected_shortfall, max_drawdown,
                    sharpe_ratio, sortino_ratio, calmar_ratio, portfolio_value,
                    num_positions, correlation_risk, concentration_risk, stress_test_results
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ''',
                risk_data.get('timestamp', datetime.utcnow()),
                risk_data['portfolio_var'],
                risk_data.get('expected_shortfall'),
                risk_data.get('max_drawdown'),
                risk_data.get('sharpe_ratio'),
                risk_data.get('sortino_ratio'),
                risk_data.get('calmar_ratio'),
                risk_data.get('portfolio_value'),
                risk_data.get('num_positions'),
                risk_data.get('correlation_risk'),
                risk_data.get('concentration_risk'),
                risk_data.get('stress_test_results', {})
            )
            
    async def insert_performance_metrics(self, perf_data: Dict[str, Any]):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO performance_metrics (
                    timestamp, total_trades, win_rate, total_return, sharpe_ratio,
                    sortino_ratio, max_drawdown, calmar_ratio, profit_factor,
                    avg_trade_duration, best_trade, worst_trade, current_streak,
                    avg_daily_return, volatility, alpha, beta, portfolio_value
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
            ''',
                perf_data.get('timestamp', datetime.utcnow()),
                perf_data.get('total_trades', 0),
                perf_data.get('win_rate', 0.0),
                perf_data.get('total_return', 0.0),
                perf_data.get('sharpe_ratio', 0.0),
                perf_data.get('sortino_ratio', 0.0),
                perf_data.get('max_drawdown', 0.0),
                perf_data.get('calmar_ratio', 0.0),
                perf_data.get('profit_factor', 0.0),
                perf_data.get('avg_trade_duration', 0.0),
                perf_data.get('best_trade', 0.0),
                perf_data.get('worst_trade', 0.0),
                perf_data.get('current_streak', 0),
                perf_data.get('avg_daily_return', 0.0),
                perf_data.get('volatility', 0.0),
                perf_data.get('alpha', 0.0),
                perf_data.get('beta', 1.0),
                perf_data.get('portfolio_value', 0.0)
            )
            
    async def log_system_event(self, level: str, component: str, message: str, metadata: Dict = None, trace_id: str = None):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO system_logs (timestamp, level, component, message, metadata, trace_id)
                VALUES ($1, $2, $3, $4, $5, $6)
            ''',
                datetime.utcnow(),
                level,
                component,
                message,
                metadata or {},
                trace_id
            )
            
    async def get_recent_trades(self, limit: int = 100, symbol: str = None) -> List[Dict]:
        async with self.pool.acquire() as conn:
            if symbol:
                rows = await conn.fetch('''
                    SELECT * FROM trades 
                    WHERE symbol = $1 
                    ORDER BY timestamp DESC 
                    LIMIT $2
                ''', symbol, limit)
            else:
                rows = await conn.fetch('''
                    SELECT * FROM trades 
                    ORDER BY timestamp DESC 
                    LIMIT $1
                ''', limit)
                
            return [dict(row) for row in rows]
            
    async def get_open_positions(self) -> List[Dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM positions 
                WHERE is_open = true 
                ORDER BY entry_time DESC
            ''')
            
            return [dict(row) for row in rows]
            
    async def get_performance_history(self, days: int = 30) -> List[Dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM performance_metrics 
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                ORDER BY timestamp ASC
            ''', days)
            
            return [dict(row) for row in rows]
            
    async def get_risk_metrics_history(self, hours: int = 24) -> List[Dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM risk_metrics 
                WHERE timestamp >= NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC
            ''', hours)
            
            return [dict(row) for row in rows]
            
    async def cleanup_old_data(self):
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM market_data WHERE timestamp < NOW() - INTERVAL '7 days'")
            await conn.execute("DELETE FROM system_logs WHERE timestamp < NOW() - INTERVAL '30 days'")
            await conn.execute("DELETE FROM risk_metrics WHERE timestamp < NOW() - INTERVAL '90 days'")
            
        print("✅ Old data cleaned up")
        
    async def close(self):
        if self.pool:
            await self.pool.close()
        if self.engine:
            self.engine.dispose()