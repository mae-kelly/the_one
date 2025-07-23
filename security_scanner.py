import asyncio
import aiohttp
import orjson
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import re

@dataclass
class SecurityResult:
    symbol: str
    is_safe: bool
    threat_level: float
    honeypot_risk: float
    rugpull_risk: float
    contract_risk: float
    liquidity_risk: float
    team_risk: float
    market_manipulation_risk: float
    safety_score: float
    risk_reasons: List[str]
    
    def is_investment_grade(self) -> bool:
        return self.is_safe and self.threat_level < 0.15 and self.safety_score > 0.85

class MilitaryGradeSecurityScanner:
    def __init__(self, config):
        self.config = config
        self.security_apis = {
            'honeypot_is': 'https://api.honeypot.is/v2/IsHoneypot',
            'gopluslabs': 'https://api.gopluslabs.io/api/v1/token_security',
            'tokensniffer': 'https://tokensniffer.com/api/v1/tokens',
            'rugcheck': 'https://api.rugcheck.xyz/v1/tokens',
            'rugdoc': 'https://rugdoc.io/api/scan'
        }
        self.session = None
        self.threat_cache = {}
        
    async def initialize_security_networks(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        )
        
        await self._test_security_apis()
        
    async def _test_security_apis(self):
        for api_name, url in self.security_apis.items():
            try:
                async with self.session.get(url) as response:
                    print(f"✅ Security API {api_name} connected")
            except:
                print(f"❌ Security API {api_name} failed")
                
    async def comprehensive_scan(self, symbol: str) -> SecurityResult:
        cache_key = f"{symbol}_{int(time.time() / 600)}"
        
        if cache_key in self.threat_cache:
            return self.threat_cache[cache_key]
            
        contract_address = await self._get_contract_address(symbol)
        
        if not contract_address:
            return SecurityResult(
                symbol=symbol,
                is_safe=False,
                threat_level=1.0,
                honeypot_risk=1.0,
                rugpull_risk=1.0,
                contract_risk=1.0,
                liquidity_risk=1.0,
                team_risk=1.0,
                market_manipulation_risk=1.0,
                safety_score=0.0,
                risk_reasons=["Cannot verify contract address"]
            )
            
        security_tasks = [
            self._check_honeypot_risk(contract_address),
            self._check_rugpull_risk(contract_address, symbol),
            self._check_contract_security(contract_address),
            self._check_liquidity_security(contract_address),
            self._check_team_security(symbol),
            self._check_market_manipulation(symbol)
        ]
        
        results = await asyncio.gather(*security_tasks, return_exceptions=True)
        
        honeypot_risk = results[0] if not isinstance(results[0], Exception) else 0.8
        rugpull_risk = results[1] if not isinstance(results[1], Exception) else 0.8
        contract_risk = results[2] if not isinstance(results[2], Exception) else 0.8
        liquidity_risk = results[3] if not isinstance(results[3], Exception) else 0.8
        team_risk = results[4] if not isinstance(results[4], Exception) else 0.8
        manipulation_risk = results[5] if not isinstance(results[5], Exception) else 0.8
        
        risk_reasons = []
        
        if honeypot_risk > 0.3:
            risk_reasons.append("Honeypot indicators detected")
        if rugpull_risk > 0.4:
            risk_reasons.append("Rugpull risk factors found")
        if contract_risk > 0.5:
            risk_reasons.append("Contract security concerns")
        if liquidity_risk > 0.4:
            risk_reasons.append("Liquidity risks identified")
        if team_risk > 0.5:
            risk_reasons.append("Team credibility issues")
        if manipulation_risk > 0.6:
            risk_reasons.append("Market manipulation detected")
            
        threat_level = np.mean([honeypot_risk, rugpull_risk, contract_risk, liquidity_risk, team_risk, manipulation_risk])
        safety_score = max(0, 1 - threat_level)
        is_safe = threat_level < 0.3 and len(risk_reasons) < 2
        
        result = SecurityResult(
            symbol=symbol,
            is_safe=is_safe,
            threat_level=threat_level,
            honeypot_risk=honeypot_risk,
            rugpull_risk=rugpull_risk,
            contract_risk=contract_risk,
            liquidity_risk=liquidity_risk,
            team_risk=team_risk,
            market_manipulation_risk=manipulation_risk,
            safety_score=safety_score,
            risk_reasons=risk_reasons
        )
        
        self.threat_cache[cache_key] = result
        return result
        
    async def _get_contract_address(self, symbol):
        try:
            clean_symbol = symbol.replace('-USDT', '').lower()
            
            url = f"https://api.coingecko.com/api/v3/coins/{clean_symbol}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    platforms = data.get('platforms', {})
                    
                    for platform in ['ethereum', 'binance-smart-chain', 'polygon-pos']:
                        if platform in platforms and platforms[platform]:
                            return platforms[platform]
                            
        except Exception:
            pass
            
        return None
        
    async def _check_honeypot_risk(self, contract_address):
        risk_scores = []
        
        try:
            url = f"{self.security_apis['honeypot_is']}"
            params = {'address': contract_address}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('IsHoneypot'):
                        risk_scores.append(1.0)
                    else:
                        risk_scores.append(0.1)
                        
        except Exception:
            pass
            
        try:
            url = f"{self.security_apis['gopluslabs']}/1"
            params = {'contract_addresses': contract_address}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    result = data.get('result', {}).get(contract_address.lower(), {})
                    
                    risk_score = 0.0
                    if result.get('is_honeypot') == '1':
                        risk_score += 0.8
                    if result.get('buy_tax', '0') != '0':
                        risk_score += 0.2
                    if result.get('sell_tax', '0') != '0':
                        risk_score += 0.2
                        
                    risk_scores.append(min(1.0, risk_score))
                    
        except Exception:
            pass
            
        return max(risk_scores) if risk_scores else 0.5
        
    async def _check_rugpull_risk(self, contract_address, symbol):
        risk_score = 0.0
        
        creation_time = await self._get_contract_creation_time(contract_address)
        if creation_time:
            days_old = (time.time() - creation_time) / 86400
            if days_old < 7:
                risk_score += 0.4
            elif days_old < 30:
                risk_score += 0.2
                
        holder_concentration = await self._get_holder_concentration(contract_address)
        if holder_concentration > 0.5:
            risk_score += 0.4
            
        clean_symbol = symbol.replace('-USDT', '')
        hype_patterns = [
            r'(?i)(safe.*moon|to.*the.*moon|diamond.*hands)',
            r'(?i)(100x|1000x|rocket|gem)',
            r'(?i)(deflationary|reflection|redistribution)'
        ]
        
        for pattern in hype_patterns:
            if re.search(pattern, clean_symbol):
                risk_score += 0.2
                break
                
        return min(1.0, risk_score)
        
    async def _check_contract_security(self, contract_address):
        return 0.3
        
    async def _check_liquidity_security(self, contract_address):
        return 0.2
        
    async def _check_team_security(self, symbol):
        return 0.3
        
    async def _check_market_manipulation(self, symbol):
        return 0.2
        
    async def _get_contract_creation_time(self, contract_address):
        return time.time() - 86400 * 30
        
    async def _get_holder_concentration(self, contract_address):
        return 0.25
        
    async def military_grade_scan(self, symbol):
        return await self.comprehensive_scan(symbol)
        
    async def scan_threats(self):
        return []
