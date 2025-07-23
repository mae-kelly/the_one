import asyncio
import psutil
import GPUtil
import redis
import asyncpg
import aiohttp
from typing import Dict, List
import time
import logging

class SystemHealthChecker:
    def __init__(self, config):
        self.config = config
        self.alert_manager = None
        self.thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'gpu_percent': 90,
            'disk_percent': 90,
            'response_time_ms': 5000
        }
        
    async def initialize(self, alert_manager):
        self.alert_manager = alert_manager
        asyncio.create_task(self._continuous_health_monitoring())
        
    async def _continuous_health_monitoring(self):
        while True:
            try:
                health_status = await self.comprehensive_health_check()
                await self._process_health_alerts(health_status)
                await asyncio.sleep(30)
            except Exception as e:
                logging.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)
                
    async def comprehensive_health_check(self) -> Dict:
        health_data = {}
        
        # System metrics
        health_data['system'] = self._get_system_metrics()
        
        # Database connectivity
        health_data['database'] = await self._check_database_health()
        
        # Exchange connectivity
        health_data['exchanges'] = await self._check_exchange_health()
        
        # Redis connectivity
        health_data['redis'] = await self._check_redis_health()
        
        # Application health
        health_data['application'] = await self._check_application_health()
        
        return health_data
        
    def _get_system_metrics(self) -> Dict:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        gpu_metrics = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_metrics.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                })
        except:
            gpu_metrics = []
            
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'gpu_metrics': gpu_metrics
        }
        
    async def _check_database_health(self) -> Dict:
        try:
            start_time = time.time()
            pool = await asyncpg.create_pool(
                self.config['database']['postgres_url'],
                min_size=1, max_size=2, command_timeout=5
            )
            
            async with pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
                
            await pool.close()
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'error': None
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'response_time_ms': None,
                'error': str(e)
            }
            
    async def _check_redis_health(self) -> Dict:
        try:
            start_time = time.time()
            r = redis.Redis.from_url(self.config['database']['redis_url'])
            r.ping()
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'memory_usage': r.info()['used_memory_human'],
                'connected_clients': r.info()['connected_clients']
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'response_time_ms': None,
                'error': str(e)
            }
            
    async def _check_exchange_health(self) -> Dict:
        exchange_health = {}
        
        exchanges_to_check = [
            {'name': 'binance', 'url': 'https://api.binance.com/api/v3/ping'},
            {'name': 'okx', 'url': 'https://www.okx.com/api/v5/public/time'},
            {'name': 'coinbase', 'url': 'https://api.exchange.coinbase.com/time'},
            {'name': 'kraken', 'url': 'https://api.kraken.com/0/public/SystemStatus'}
        ]
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for exchange in exchanges_to_check:
                try:
                    start_time = time.time()
                    async with session.get(exchange['url']) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            exchange_health[exchange['name']] = {
                                'status': 'healthy',
                                'response_time_ms': response_time
                            }
                        else:
                            exchange_health[exchange['name']] = {
                                'status': 'degraded',
                                'response_time_ms': response_time,
                                'status_code': response.status
                            }
                            
                except Exception as e:
                    exchange_health[exchange['name']] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                    
        return exchange_health
        
    async def _check_application_health(self) -> Dict:
        # Check if main application is responsive
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8080/health', timeout=5) as response:
                    if response.status == 200:
                        return {
                            'status': 'healthy',
                            'response_time_ms': 100  # placeholder
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'status_code': response.status
                        }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
            
    async def _process_health_alerts(self, health_data: Dict):
        # Check system thresholds
        system = health_data.get('system', {})
        
        if system.get('cpu_percent', 0) > self.thresholds['cpu_percent']:
            await self.alert_manager.send_alert(
                AlertSeverity.HIGH,
                'system',
                f"High CPU usage: {system['cpu_percent']:.1f}%",
                {'cpu_percent': system['cpu_percent']}
            )
            
        if system.get('memory_percent', 0) > self.thresholds['memory_percent']:
            await self.alert_manager.send_alert(
                AlertSeverity.HIGH,
                'system',
                f"High memory usage: {system['memory_percent']:.1f}%",
                {'memory_percent': system['memory_percent']}
            )
            
        # Check GPU usage
        for gpu in system.get('gpu_metrics', []):
            if gpu['load'] > self.thresholds['gpu_percent']:
                await self.alert_manager.send_alert(
                    AlertSeverity.MEDIUM,
                    'system',
                    f"High GPU usage: {gpu['load']:.1f}% on {gpu['name']}",
                    {'gpu_id': gpu['id'], 'gpu_load': gpu['load']}
                )
                
        # Check database health
        db_health = health_data.get('database', {})
        if db_health.get('status') == 'unhealthy':
            await self.alert_manager.send_alert(
                AlertSeverity.CRITICAL,
                'database',
                f"Database connection failed: {db_health.get('error', 'Unknown error')}",
                db_health
            )
            
        # Check exchange health
        exchange_health = health_data.get('exchanges', {})
        for exchange_name, health in exchange_health.items():
            if health.get('status') == 'unhealthy':
                await self.alert_manager.send_alert(
                    AlertSeverity.HIGH,
                    'exchange',
                    f"Exchange {exchange_name} is unhealthy: {health.get('error', 'Unknown error')}",
                    {'exchange': exchange_name, 'health': health}
                )
            elif health.get('response_time_ms', 0) > self.thresholds['response_time_ms']:
                await self.alert_manager.send_alert(
                    AlertSeverity.MEDIUM,
                    'exchange',
                    f"High latency to {exchange_name}: {health['response_time_ms']:.0f}ms",
                    {'exchange': exchange_name, 'latency': health['response_time_ms']}
                )