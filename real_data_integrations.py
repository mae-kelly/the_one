import asyncio
import aiohttp
import orjson
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import praw
import tweepy
from bs4 import BeautifulSoup
import feedparser

class RealDataIntegrator:
    def __init__(self, config):
        self.config = config
        self.session = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.reddit_client = None
        self.twitter_client = None
        
    async def initialize_all_sources(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        await self._setup_social_clients()
        
    async def _setup_social_clients(self):
        if 'reddit' in self.config:
            self.reddit_client = praw.Reddit(
                client_id=self.config['reddit']['client_id'],
                client_secret=self.config['reddit']['client_secret'],
                user_agent=self.config['reddit']['user_agent']
            )
            
        if 'twitter' in self.config:
            self.twitter_client = tweepy.Client(
                bearer_token=self.config['twitter']['bearer_token']
            )

class RealSentimentAnalyzer(RealDataIntegrator):
    async def get_reddit_sentiment(self, symbol: str) -> float:
        try:
            clean_symbol = symbol.replace('-USDT', '').upper()
            
            subreddit = self.reddit_client.subreddit('cryptocurrency')
            posts = subreddit.search(clean_symbol, limit=100, time_filter='day')
            
            scores = []
            for post in posts:
                combined_text = f"{post.title} {post.selftext}"
                score = self.sentiment_analyzer.polarity_scores(combined_text)
                scores.append(score['compound'])
                
            if scores:
                avg_sentiment = np.mean(scores)
                return (avg_sentiment + 1) / 2
            else:
                return 0.5
                
        except Exception:
            return 0.5
            
    async def get_twitter_sentiment(self, symbol: str) -> float:
        try:
            clean_symbol = symbol.replace('-USDT', '').upper()
            
            tweets = tweepy.Paginator(
                self.twitter_client.search_recent_tweets,
                query=f"#{clean_symbol} OR ${clean_symbol}",
                max_results=100
            ).flatten(limit=500)
            
            scores = []
            for tweet in tweets:
                if hasattr(tweet, 'text'):
                    score = self.sentiment_analyzer.polarity_scores(tweet.text)
                    scores.append(score['compound'])
                    
            if scores:
                avg_sentiment = np.mean(scores)
                return (avg_sentiment + 1) / 2
            else:
                return 0.5
                
        except Exception:
            return 0.5
            
    async def get_news_sentiment(self, symbol: str) -> float:
        try:
            clean_symbol = symbol.replace('-USDT', '')
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{clean_symbol} cryptocurrency",
                'sortBy': 'publishedAt',
                'pageSize': 100,
                'apiKey': self.config['news_api']['api_key']
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])
                    
                    scores = []
                    for article in articles:
                        text = f"{article.get('title', '')} {article.get('description', '')}"
                        if text.strip():
                            score = self.sentiment_analyzer.polarity_scores(text)
                            scores.append(score['compound'])
                            
                    if scores:
                        return (np.mean(scores) + 1) / 2
                        
        except Exception:
            pass
            
        return 0.5

class RealSecurityAnalyzer(RealDataIntegrator):
    async def check_honeypot_apis(self, contract_address: str) -> float:
        honeypot_score = 0.0
        
        apis_to_check = [
            {
                'name': 'honeypot.is',
                'url': f"https://api.honeypot.is/v2/IsHoneypot?address={contract_address}",
                'weight': 0.4
            },
            {
                'name': 'gopluslabs',
                'url': f"https://api.gopluslabs.io/api/v1/token_security/1?contract_addresses={contract_address}",
                'weight': 0.6
            }
        ]
        
        for api in apis_to_check:
            try:
                async with self.session.get(api['url']) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if api['name'] == 'honeypot.is':
                            if data.get('IsHoneypot'):
                                honeypot_score += api['weight']
                                
                        elif api['name'] == 'gopluslabs':
                            result = data.get('result', {}).get(contract_address.lower(), {})
                            if result.get('is_honeypot') == '1':
                                honeypot_score += api['weight'] * 0.8
                            if float(result.get('buy_tax', '0')) > 0.1:
                                honeypot_score += api['weight'] * 0.2
                                
            except Exception:
                continue
                
        return honeypot_score
        
    async def check_rugpull_indicators(self, contract_address: str, symbol: str) -> float:
        risk_score = 0.0
        
        try:
            holder_concentration = await self._get_holder_concentration(contract_address)
            if holder_concentration > 0.5:
                risk_score += 0.3
                
            liquidity_locked = await self._check_liquidity_locks(contract_address)
            if not liquidity_locked:
                risk_score += 0.4
                
            contract_verified = await self._check_contract_verification(contract_address)
            if not contract_verified:
                risk_score += 0.2
                
            team_doxxed = await self._check_team_information(symbol)
            if not team_doxxed:
                risk_score += 0.1
                
        except Exception:
            risk_score = 0.5
            
        return min(1.0, risk_score)
        
    async def _get_holder_concentration(self, contract_address: str) -> float:
        try:
            url = f"https://api.etherscan.io/api"
            params = {
                'module': 'token',
                'action': 'tokenholderlist',
                'contractaddress': contract_address,
                'page': 1,
                'offset': 100,
                'apikey': self.config.get('etherscan_api_key', '')
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    holders = data.get('result', [])
                    
                    if holders:
                        total_supply = sum(float(h.get('TokenHolderQuantity', 0)) for h in holders)
                        top_holder = max(holders, key=lambda x: float(x.get('TokenHolderQuantity', 0)))
                        top_holder_pct = float(top_holder.get('TokenHolderQuantity', 0)) / total_supply
                        return top_holder_pct
                        
        except Exception:
            pass
            
        return 0.2
        
    async def _check_liquidity_locks(self, contract_address: str) -> bool:
        try:
            lock_services = [
                'https://unicrypt.network/api/locks',
                'https://app.unicrypt.network/api/locks'
            ]
            
            for service_url in lock_services:
                try:
                    async with self.session.get(f"{service_url}?token={contract_address}") as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('locks'):
                                return True
                except Exception:
                    continue
                    
        except Exception:
            pass
            
        return False
        
    async def _check_contract_verification(self, contract_address: str) -> bool:
        try:
            url = f"https://api.etherscan.io/api"
            params = {
                'module': 'contract',
                'action': 'getsourcecode',
                'address': contract_address,
                'apikey': self.config.get('etherscan_api_key', '')
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get('result', [{}])[0]
                    return bool(result.get('SourceCode'))
                    
        except Exception:
            pass
            
        return False
        
    async def _check_team_information(self, symbol: str) -> bool:
        try:
            clean_symbol = symbol.replace('-USDT', '').lower()
            
            url = f"https://api.coingecko.com/api/v3/coins/{clean_symbol}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    links = data.get('links', {})
                    
                    has_website = bool(links.get('homepage', [None])[0])
                    has_whitepaper = bool(links.get('whitepaper'))
                    has_github = bool(links.get('repos_url', {}).get('github'))
                    
                    return has_website and (has_whitepaper or has_github)
                    
        except Exception:
            pass
            
        return False

class RealResearchAnalyzer(RealDataIntegrator):
    async def get_fundamental_metrics(self, symbol: str) -> Dict:
        try:
            clean_symbol = symbol.replace('-USDT', '').lower()
            
            url = f"https://api.coingecko.com/api/v3/coins/{clean_symbol}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    market_data = data.get('market_data', {})
                    developer_data = data.get('developer_data', {})
                    community_data = data.get('community_data', {})
                    
                    return {
                        'market_cap_rank': data.get('market_cap_rank', 999),
                        'total_volume': market_data.get('total_volume', {}).get('usd', 0),
                        'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                        'circulating_supply': market_data.get('circulating_supply', 0),
                        'total_supply': market_data.get('total_supply', 0),
                        'max_supply': market_data.get('max_supply', 0),
                        'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                        'price_change_7d': market_data.get('price_change_percentage_7d', 0),
                        'price_change_30d': market_data.get('price_change_percentage_30d', 0),
                        'github_commits': developer_data.get('commit_count_4_weeks', 0),
                        'github_contributors': developer_data.get('subscribers', 0),
                        'reddit_subscribers': community_data.get('reddit_subscribers', 0),
                        'twitter_followers': community_data.get('twitter_followers', 0),
                        'developer_score': data.get('developer_score', 0),
                        'community_score': data.get('community_score', 0),
                        'liquidity_score': data.get('liquidity_score', 0),
                        'public_interest_score': data.get('public_interest_score', 0)
                    }
                    
        except Exception:
            pass
            
        return {}
        
    async def get_on_chain_metrics(self, symbol: str) -> Dict:
        try:
            clean_symbol = symbol.replace('-USDT', '').upper()
            
            if 'glassnode' in self.config:
                metrics = {}
                
                glassnode_endpoints = [
                    'addresses/active_count',
                    'addresses/new_non_zero_count',
                    'transactions/count',
                    'transactions/transfers_volume_sum',
                    'market/price_usd_close',
                    'market/marketcap_usd',
                    'supply/current',
                    'distribution/balance_1pct_holders'
                ]
                
                for endpoint in glassnode_endpoints:
                    try:
                        url = f"https://api.glassnode.com/v1/metrics/{endpoint}"
                        params = {
                            'a': clean_symbol,
                            'api_key': self.config['glassnode']['api_key'],
                            'i': '24h',
                            's': int(time.time()) - 86400 * 7
                        }
                        
                        async with self.session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                if data:
                                    metric_name = endpoint.split('/')[-1]
                                    metrics[metric_name] = data[-1]['v'] if data else 0
                                    
                    except Exception:
                        continue
                        
                return metrics
                
        except Exception:
            pass
            
        return {}
        
    async def get_defi_metrics(self, symbol: str) -> Dict:
        try:
            clean_symbol = symbol.replace('-USDT', '').upper()
            
            if 'defipulse' in self.config:
                url = f"https://api.defipulse.com/v1/projects"
                params = {'api-key': self.config['defipulse']['api_key']}
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        projects = await response.json()
                        
                        for project in projects:
                            if project.get('symbol', '').upper() == clean_symbol:
                                return {
                                    'tvl': project.get('value', {}).get('tvl', {}).get('ETH', {}).get('value', 0),
                                    'category': project.get('category', ''),
                                    'chains': project.get('chains', []),
                                    'audits': len(project.get('audits', [])),
                                    'logo': project.get('logo')
                                }
                                
        except Exception:
            pass
            
        return {}

class RealWhaleTracker(RealDataIntegrator):
    async def track_whale_movements(self, symbol: str) -> Dict:
        try:
            clean_symbol = symbol.replace('-USDT', '').upper()
            
            if 'whale_alert' in self.config:
                url = "https://api.whale-alert.io/v1/transactions"
                params = {
                    'api_key': self.config['whale_alert']['api_key'],
                    'currency': clean_symbol.lower(),
                    'min_value': 1000000,
                    'limit': 100
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        transactions = data.get('result', [])
                        
                        whale_activity = {
                            'large_transactions_24h': 0,
                            'total_volume_24h': 0,
                            'exchange_inflows': 0,
                            'exchange_outflows': 0,
                            'unknown_wallet_activity': 0
                        }
                        
                        current_time = time.time()
                        for tx in transactions:
                            tx_time = tx.get('timestamp', 0)
                            
                            if current_time - tx_time <= 86400:
                                whale_activity['large_transactions_24h'] += 1
                                whale_activity['total_volume_24h'] += tx.get('amount_usd', 0)
                                
                                if tx.get('to', {}).get('owner_type') == 'exchange':
                                    whale_activity['exchange_inflows'] += tx.get('amount_usd', 0)
                                elif tx.get('from', {}).get('owner_type') == 'exchange':
                                    whale_activity['exchange_outflows'] += tx.get('amount_usd', 0)
                                elif tx.get('from', {}).get('owner_type') == 'unknown':
                                    whale_activity['unknown_wallet_activity'] += tx.get('amount_usd', 0)
                                    
                        return whale_activity
                        
        except Exception:
            pass
            
        return {}

class RealLiquidityAnalyzer(RealDataIntegrator):
    async def analyze_dex_liquidity(self, symbol: str) -> Dict:
        try:
            clean_symbol = symbol.replace('-USDT', '')
            
            dex_apis = [
                f"https://api.dexscreener.com/latest/dex/tokens/{clean_symbol}",
                f"https://api.geckoterminal.com/api/v2/networks/eth/tokens/{clean_symbol}/pools"
            ]
            
            total_liquidity = 0
            pool_count = 0
            
            for api_url in dex_apis:
                try:
                    async with self.session.get(api_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if 'dexscreener' in api_url:
                                pairs = data.get('pairs', [])
                                for pair in pairs:
                                    total_liquidity += float(pair.get('liquidity', {}).get('usd', 0))
                                    pool_count += 1
                                    
                            elif 'geckoterminal' in api_url:
                                pools = data.get('data', [])
                                for pool in pools:
                                    attributes = pool.get('attributes', {})
                                    total_liquidity += float(attributes.get('reserve_in_usd', 0))
                                    pool_count += 1
                                    
                except Exception:
                    continue
                    
            return {
                'total_liquidity_usd': total_liquidity,
                'pool_count': pool_count,
                'avg_pool_size': total_liquidity / pool_count if pool_count > 0 else 0,
                'liquidity_score': min(100, total_liquidity / 1000000)
            }
            
        except Exception:
            pass
            
        return {}

class RealOptionsAnalyzer(RealDataIntegrator):
    async def get_options_data(self, symbol: str) -> Dict:
        try:
            clean_symbol = symbol.replace('-USDT', '').upper()
            
            if clean_symbol in ['BTC', 'ETH']:
                url = f"https://www.deribit.com/api/v2/public/get_instruments"
                params = {
                    'currency': clean_symbol,
                    'kind': 'option',
                    'expired': False
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        instruments = data.get('result', [])
                        
                        call_volume = 0
                        put_volume = 0
                        total_open_interest = 0
                        
                        for instrument in instruments[:50]:
                            instrument_name = instrument.get('instrument_name', '')
                            
                            if '-C' in instrument_name:
                                call_volume += instrument.get('stats', {}).get('volume', 0)
                            elif '-P' in instrument_name:
                                put_volume += instrument.get('stats', {}).get('volume', 0)
                                
                            total_open_interest += instrument.get('stats', {}).get('open_interest', 0)
                            
                        put_call_ratio = put_volume / call_volume if call_volume > 0 else 1.0
                        
                        return {
                            'put_call_ratio': put_call_ratio,
                            'total_open_interest': total_open_interest,
                            'call_volume': call_volume,
                            'put_volume': put_volume,
                            'max_pain': self._calculate_max_pain(instruments)
                        }
                        
        except Exception:
            pass
            
        return {}
        
    def _calculate_max_pain(self, instruments: List) -> float:
        strike_data = {}
        
        for instrument in instruments:
            try:
                if 'option_type' in instrument:
                    strike = float(instrument.get('strike', 0))
                    open_interest = instrument.get('stats', {}).get('open_interest', 0)
                    option_type = instrument.get('option_type', '')
                    
                    if strike not in strike_data:
                        strike_data[strike] = {'calls': 0, 'puts': 0}
                        
                    if option_type == 'call':
                        strike_data[strike]['calls'] += open_interest
                    elif option_type == 'put':
                        strike_data[strike]['puts'] += open_interest
                        
            except Exception:
                continue
                
        if not strike_data:
            return 0
            
        max_pain_strike = 0
        min_pain = float('inf')
        
        for strike in strike_data:
            total_pain = 0
            
            for other_strike, data in strike_data.items():
                if strike > other_strike:
                    total_pain += data['calls'] * (strike - other_strike)
                if strike < other_strike:
                    total_pain += data['puts'] * (other_strike - strike)
                    
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = strike
                
        return max_pain_strike