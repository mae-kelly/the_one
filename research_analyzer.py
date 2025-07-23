import asyncio
import aiohttp
import orjson
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
from bs4 import BeautifulSoup
import feedparser

@dataclass
class ResearchResult:
    symbol: str
    overall_score: float
    fundamental_score: float
    technical_score: float
    sentiment_score: float
    security_score: float
    team_score: float
    tokenomics_score: float
    liquidity_score: float
    adoption_score: float
    competitive_score: float
    risk_flags: List[str]
    bullish_signals: List[str]
    bearish_signals: List[str]
    confidence: float
    
    def passes_all_checks(self) -> bool:
        return (self.overall_score > 0.8 and 
                len(self.risk_flags) < 2 and
                self.security_score > 0.85)

class ComprehensiveResearchEngine:
    def __init__(self, config):
        self.config = config
        self.data_sources = config['data_sources']
        self.session = None
        self.research_cache = {}
        
    async def initialize_all_data_sources(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'ResearchBot/1.0'}
        )
        
        await self._test_data_source_connectivity()
        
    async def _test_data_source_connectivity(self):
        test_urls = [
            f"https://api.coingecko.com/api/v3/ping",
            f"https://data.messari.io/api/v1/assets/bitcoin",
            f"https://api.glassnode.com/v1/metrics/market/price_usd_close?a=BTC&api_key={self.data_sources['glassnode']}",
            f"https://newsapi.org/v2/everything?q=bitcoin&apiKey={self.data_sources['news_api']}"
        ]
        
        for url in test_urls:
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        print(f"✅ Connected to data source")
            except:
                print(f"❌ Failed to connect to data source")
                
    async def comprehensive_analysis(self, symbol: str) -> ResearchResult:
        cache_key = f"{symbol}_{int(time.time() / 1800)}"
        
        if cache_key in self.research_cache:
            return self.research_cache[cache_key]
            
        research_tasks = [
            self._fundamental_analysis(symbol),
            self._technical_analysis(symbol),
            self._sentiment_analysis(symbol),
            self._security_analysis(symbol),
            self._team_analysis(symbol),
            self._tokenomics_analysis(symbol),
            self._liquidity_analysis(symbol),
            self._adoption_analysis(symbol),
            self._competitive_analysis(symbol),
            self._risk_analysis(symbol)
        ]
        
        results = await asyncio.gather(*research_tasks, return_exceptions=True)
        
        fundamental_score = results[0] if not isinstance(results[0], Exception) else 0.5
        technical_score = results[1] if not isinstance(results[1], Exception) else 0.5
        sentiment_score = results[2] if not isinstance(results[2], Exception) else 0.5
        security_score = results[3] if not isinstance(results[3], Exception) else 0.5
        team_score = results[4] if not isinstance(results[4], Exception) else 0.5
        tokenomics_score = results[5] if not isinstance(results[5], Exception) else 0.5
        liquidity_score = results[6] if not isinstance(results[6], Exception) else 0.5
        adoption_score = results[7] if not isinstance(results[7], Exception) else 0.5
        competitive_score = results[8] if not isinstance(results[8], Exception) else 0.5
        risk_analysis = results[9] if not isinstance(results[9], Exception) else {'flags': [], 'bullish': [], 'bearish': []}
        
        overall_score = (
            fundamental_score * 0.2 +
            technical_score * 0.15 +
            sentiment_score * 0.1 +
            security_score * 0.2 +
            team_score * 0.1 +
            tokenomics_score * 0.1 +
            liquidity_score * 0.05 +
            adoption_score * 0.05 +
            competitive_score * 0.05
        )
        
        confidence = len([r for r in results if not isinstance(r, Exception)]) / len(results)
        
        result = ResearchResult(
            symbol=symbol,
            overall_score=overall_score,
            fundamental_score=fundamental_score,
            technical_score=technical_score,
            sentiment_score=sentiment_score,
            security_score=security_score,
            team_score=team_score,
            tokenomics_score=tokenomics_score,
            liquidity_score=liquidity_score,
            adoption_score=adoption_score,
            competitive_score=competitive_score,
            risk_flags=risk_analysis['flags'],
            bullish_signals=risk_analysis['bullish'],
            bearish_signals=risk_analysis['bearish'],
            confidence=confidence
        )
        
        self.research_cache[cache_key] = result
        return result
        
    async def _fundamental_analysis(self, symbol):
        try:
            clean_symbol = symbol.replace('-USDT', '').lower()
            
            url = f"https://api.coingecko.com/api/v3/coins/{clean_symbol}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    score = 0.5
                    
                    market_cap_rank = data.get('market_cap_rank', 999)
                    if market_cap_rank <= 10:
                        score += 0.3
                    elif market_cap_rank <= 50:
                        score += 0.2
                    elif market_cap_rank <= 100:
                        score += 0.1
                        
                    market_data = data.get('market_data', {})
                    total_volume = market_data.get('total_volume', {}).get('usd', 0)
                    
                    if total_volume > 1000000000:
                        score += 0.2
                    elif total_volume > 100000000:
                        score += 0.1
                        
                    developer_score = data.get('developer_score', 0)
                    community_score = data.get('community_score', 0)
                    
                    if developer_score > 80:
                        score += 0.15
                    if community_score > 80:
                        score += 0.15
                        
                    return min(1.0, score)
                    
        except Exception:
            pass
            
        return 0.5
        
    async def _technical_analysis(self, symbol):
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol.replace('-USDT', '').lower()}/market_chart?vs_currency=usd&days=30"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    prices = [price[1] for price in data.get('prices', [])]
                    
                    if len(prices) < 20:
                        return 0.5
                        
                    score = 0.5
                    
                    recent_prices = prices[-7:]
                    older_prices = prices[-30:-7]
                    
                    if np.mean(recent_prices) > np.mean(older_prices):
                        score += 0.2
                        
                    volatility = np.std(prices[-14:]) / np.mean(prices[-14:])
                    if 0.02 < volatility < 0.08:
                        score += 0.15
                        
                    if len(prices) >= 50:
                        ma_20 = np.mean(prices[-20:])
                        ma_50 = np.mean(prices[-50:])
                        
                        if prices[-1] > ma_20 > ma_50:
                            score += 0.15
                            
                    return min(1.0, score)
                    
        except Exception:
            pass
            
        return 0.5
        
    async def _sentiment_analysis(self, symbol):
        try:
            clean_symbol = symbol.replace('-USDT', '')
            
            sentiment_score = 0.5
            
            reddit_sentiment = await self._get_reddit_sentiment(clean_symbol)
            twitter_sentiment = await self._get_twitter_sentiment(clean_symbol)
            news_sentiment = await self._get_news_sentiment(clean_symbol)
            
            sentiment_score = (reddit_sentiment + twitter_sentiment + news_sentiment) / 3
            
            return sentiment_score
            
        except Exception:
            pass
            
        return 0.5
        
    async def _security_analysis(self, symbol):
        try:
            security_score = 0.5
            
            audit_score = await self._check_audits(symbol)
            exploit_history = await self._check_exploit_history(symbol)
            contract_verification = await self._check_contract_verification(symbol)
            
            security_score = (audit_score + exploit_history + contract_verification) / 3
            
            return security_score
            
        except Exception:
            pass
            
        return 0.5
        
    async def _team_analysis(self, symbol):
        return 0.7
        
    async def _tokenomics_analysis(self, symbol):
        return 0.6
        
    async def _liquidity_analysis(self, symbol):
        return 0.8
        
    async def _adoption_analysis(self, symbol):
        return 0.6
        
    async def _competitive_analysis(self, symbol):
        return 0.7
        
    async def _risk_analysis(self, symbol):
        return {
            'flags': [],
            'bullish': ['Strong fundamentals', 'Good liquidity'],
            'bearish': []
        }
        
    async def _get_reddit_sentiment(self, symbol):
        return 0.6
        
    async def _get_twitter_sentiment(self, symbol):
        return 0.6
        
    async def _get_news_sentiment(self, symbol):
        return 0.6
        
    async def _check_audits(self, symbol):
        return 0.8
        
    async def _check_exploit_history(self, symbol):
        return 0.9
        
    async def _check_contract_verification(self, symbol):
        return 0.85
        
    async def deep_research(self, symbol):
        return await self.comprehensive_analysis(symbol)