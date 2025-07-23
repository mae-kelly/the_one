-- Initial database setup
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Optimize PostgreSQL for trading workload
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

SELECT pg_reload_conf();

-- Create indexes for optimal query performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_timestamp_desc ON trades (timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_symbol_timestamp ON trades (symbol, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_venue_timestamp ON trades (venue, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_success_pnl ON trades (success, pnl DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data (symbol, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_momentum_liquidity ON market_data (momentum_1m, liquidity_score);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_confidence_desc ON signals (confidence DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_status_timestamp ON signals (status, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_symbol_open ON positions (symbol, is_open);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_unrealized_pnl ON positions (unrealized_pnl DESC);

-- Create materialized views for analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS trading_performance_summary AS
SELECT 
    DATE(timestamp) as trade_date,
    COUNT(*) as total_trades,
    COUNT(*) FILTER (WHERE success = true) as successful_trades,
    ROUND(AVG(CASE WHEN success THEN pnl ELSE 0 END)::numeric, 2) as avg_profit,
    ROUND(SUM(pnl)::numeric, 2) as daily_pnl,
    ROUND(AVG(execution_time_ms)::numeric, 2) as avg_execution_time,
    STRING_AGG(DISTINCT venue, ',') as venues_used
FROM trades 
WHERE timestamp >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY DATE(timestamp)
ORDER BY trade_date DESC;

CREATE UNIQUE INDEX ON trading_performance_summary (trade_date);

CREATE MATERIALIZED VIEW IF NOT EXISTS symbol_performance AS
SELECT 
    symbol,
    COUNT(*) as total_trades,
    ROUND(AVG(CASE WHEN success THEN pnl ELSE 0 END)::numeric, 2) as avg_profit,
    ROUND(SUM(pnl)::numeric, 2) as total_pnl,
    ROUND((COUNT(*) FILTER (WHERE success = true)::float / COUNT(*) * 100)::numeric, 1) as win_rate,
    MAX(timestamp) as last_trade
FROM trades 
WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY symbol
ORDER BY total_pnl DESC;

CREATE UNIQUE INDEX ON symbol_performance (symbol);

-- Refresh materialized views every hour
SELECT cron.schedule('refresh-trading-views', '0 * * * *', 'REFRESH MATERIALIZED VIEW CONCURRENTLY trading_performance_summary; REFRESH MATERIALIZED VIEW CONCURRENTLY symbol_performance;');

-- Create function for trade analytics
CREATE OR REPLACE FUNCTION get_trading_analytics(days_back INTEGER DEFAULT 7)
RETURNS TABLE (
    metric_name TEXT,
    metric_value NUMERIC,
    metric_unit TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH trade_stats AS (
        SELECT 
            COUNT(*) as total_trades,
            COUNT(*) FILTER (WHERE success = true) as winning_trades,
            SUM(pnl) as total_pnl,
            AVG(pnl) FILTER (WHERE success = true) as avg_winning_trade,
            AVG(pnl) FILTER (WHERE success = false) as avg_losing_trade,
            STDDEV(pnl) as pnl_std
        FROM trades 
        WHERE timestamp >= CURRENT_TIMESTAMP - (days_back || ' days')::INTERVAL
    )
    SELECT 'Total Trades'::TEXT, total_trades::NUMERIC, 'count'::TEXT FROM trade_stats
    UNION ALL
    SELECT 'Win Rate'::TEXT, ROUND((winning_trades::FLOAT / NULLIF(total_trades, 0) * 100)::NUMERIC, 2), 'percent'::TEXT FROM trade_stats
    UNION ALL
    SELECT 'Total PnL'::TEXT, ROUND(total_pnl::NUMERIC, 2), 'USD'::TEXT FROM trade_stats
    UNION ALL
    SELECT 'Avg Winning Trade'::TEXT, ROUND(avg_winning_trade::NUMERIC, 2), 'USD'::TEXT FROM trade_stats
    UNION ALL
    SELECT 'Avg Losing Trade'::TEXT, ROUND(avg_losing_trade::NUMERIC, 2), 'USD'::TEXT FROM trade_stats
    UNION ALL
    SELECT 'PnL Volatility'::TEXT, ROUND(pnl_std::NUMERIC, 2), 'USD'::TEXT FROM trade_stats;
END;
$$ LANGUAGE plpgsql;