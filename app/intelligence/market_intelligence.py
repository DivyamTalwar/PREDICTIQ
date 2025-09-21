import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import openai
from ..config.settings import get_settings
import yfinance as yf
from textblob import TextBlob

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class MarketIntelligence:
    sector_performance: Dict[str, float]
    competitor_analysis: Dict[str, Dict]
    market_sentiment: float
    economic_indicators: Dict[str, float]
    news_sentiment: float
    risk_assessment: str
    opportunities: List[str]
    threats: List[str]

@dataclass
class CompetitorMetrics:
    company: str
    market_cap: float
    pe_ratio: float
    revenue_growth: float
    profit_margin: float
    stock_performance: float
    sentiment_score: float

class AdvancedMarketIntelligence:
    def __init__(self):
        self.session = None
        self.competitor_tickers = {
            'TCS': 'TCS.NS',
            'Infosys': 'INFY.NS',
            'Wipro': 'WIPRO.NS',
            'HCL Tech': 'HCLTECH.NS',
            'Tech Mahindra': 'TECHM.NS',
            'Cognizant': 'CTSH'
        }

    async def initialize(self):
        """Initialize market intelligence engine"""
        self.session = aiohttp.ClientSession()
        logger.info("🌐 Market Intelligence Engine initialized")

    async def gather_comprehensive_intelligence(self) -> MarketIntelligence:
        """Gather comprehensive market intelligence"""
        logger.info("🔍 Gathering comprehensive market intelligence...")

        # Parallel data gathering for speed
        tasks = [
            self._analyze_sector_performance(),
            self._analyze_competitors(),
            self._gather_economic_indicators(),
            self._analyze_news_sentiment(),
            self._assess_market_risks()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        sector_data = results[0] if not isinstance(results[0], Exception) else {}
        competitor_data = results[1] if not isinstance(results[1], Exception) else {}
        economic_data = results[2] if not isinstance(results[2], Exception) else {}
        news_sentiment = results[3] if not isinstance(results[3], Exception) else 0.0
        risk_data = results[4] if not isinstance(results[4], Exception) else {}

        # Calculate overall market sentiment
        market_sentiment = self._calculate_market_sentiment(
            sector_data, competitor_data, economic_data, news_sentiment
        )

        # Generate AI-powered insights
        opportunities, threats = await self._generate_market_insights(
            sector_data, competitor_data, economic_data, market_sentiment
        )

        return MarketIntelligence(
            sector_performance=sector_data,
            competitor_analysis=competitor_data,
            market_sentiment=market_sentiment,
            economic_indicators=economic_data,
            news_sentiment=news_sentiment,
            risk_assessment=risk_data.get('assessment', 'MODERATE'),
            opportunities=opportunities,
            threats=threats
        )

    async def _analyze_sector_performance(self) -> Dict[str, float]:
        """Analyze IT sector performance"""
        logger.info("📊 Analyzing IT sector performance...")

        try:
            sector_indices = {
                'IT_Index': '^CNXIT',
                'Nifty_50': '^NSEI',
                'NASDAQ': '^IXIC'
            }

            performance = {}

            for name, ticker in sector_indices.items():
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="3mo")

                    if not hist.empty:
                        recent_price = hist['Close'].iloc[-1]
                        past_price = hist['Close'].iloc[0]
                        performance[name] = ((recent_price - past_price) / past_price) * 100

                except Exception as e:
                    logger.warning(f"⚠️ Could not fetch {name}: {str(e)}")
                    performance[name] = 0.0

            return performance

        except Exception as e:
            logger.error(f"❌ Error in sector analysis: {str(e)}")
            return {}

    async def _analyze_competitors(self) -> Dict[str, CompetitorMetrics]:
        """Analyze competitor performance and metrics"""
        logger.info("🏢 Analyzing competitor landscape...")

        competitors = {}

        for company, ticker in self.competitor_tickers.items():
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="1y")

                if not hist.empty and info:
                    # Calculate metrics
                    current_price = hist['Close'].iloc[-1]
                    year_ago_price = hist['Close'].iloc[0]
                    stock_performance = ((current_price - year_ago_price) / year_ago_price) * 100

                    metrics = CompetitorMetrics(
                        company=company,
                        market_cap=info.get('marketCap', 0) / 1e9,  # In billions
                        pe_ratio=info.get('forwardPE', info.get('trailingPE', 0)),
                        revenue_growth=info.get('revenueGrowth', 0) * 100,
                        profit_margin=info.get('profitMargins', 0) * 100,
                        stock_performance=stock_performance,
                        sentiment_score=await self._get_company_sentiment(company)
                    )

                    competitors[company] = metrics.__dict__

            except Exception as e:
                logger.warning(f"⚠️ Could not analyze {company}: {str(e)}")

        return competitors

    async def _gather_economic_indicators(self) -> Dict[str, float]:
        """Gather relevant economic indicators"""
        logger.info("📈 Gathering economic indicators...")

        indicators = {}

        try:
            # USD/INR Exchange Rate
            usd_inr = yf.Ticker("USDINR=X")
            usd_hist = usd_inr.history(period="1mo")
            if not usd_hist.empty:
                current_rate = usd_hist['Close'].iloc[-1]
                month_ago = usd_hist['Close'].iloc[0]
                indicators['USD_INR_Change'] = ((current_rate - month_ago) / month_ago) * 100

            # India VIX (Volatility Index)
            vix = yf.Ticker("^INDIAVIX")
            vix_hist = vix.history(period="1mo")
            if not vix_hist.empty:
                indicators['India_VIX'] = vix_hist['Close'].iloc[-1]

            # Global Tech ETF performance
            tech_etf = yf.Ticker("XLK")
            tech_hist = tech_etf.history(period="3mo")
            if not tech_hist.empty:
                current = tech_hist['Close'].iloc[-1]
                past = tech_hist['Close'].iloc[0]
                indicators['Global_Tech_Performance'] = ((current - past) / past) * 100

        except Exception as e:
            logger.error(f"❌ Error gathering economic indicators: {str(e)}")

        return indicators

    async def _analyze_news_sentiment(self) -> float:
        """Analyze news sentiment for IT sector"""
        logger.info("📰 Analyzing news sentiment...")

        try:
            # Mock news headlines (in production, use real news API)
            sample_headlines = [
                "Indian IT sector shows strong growth in Q3",
                "TCS announces major digital transformation deals",
                "IT companies face margin pressure due to inflation",
                "Strong demand for cloud services boosts IT revenues",
                "Geopolitical tensions affect IT outsourcing",
                "AI and automation driving IT sector innovation"
            ]

            sentiments = []
            for headline in sample_headlines:
                blob = TextBlob(headline)
                sentiments.append(blob.sentiment.polarity)

            average_sentiment = np.mean(sentiments)
            return round(average_sentiment, 3)

        except Exception as e:
            logger.error(f"❌ Error in news sentiment analysis: {str(e)}")
            return 0.0

    async def _get_company_sentiment(self, company: str) -> float:
        """Get sentiment score for specific company"""
        try:
            # Mock sentiment analysis (in production, use real sentiment API)
            sentiment_map = {
                'TCS': 0.8,
                'Infosys': 0.7,
                'Wipro': 0.6,
                'HCL Tech': 0.75,
                'Tech Mahindra': 0.65,
                'Cognizant': 0.7
            }
            return sentiment_map.get(company, 0.5)

        except:
            return 0.5

    async def _assess_market_risks(self) -> Dict[str, str]:
        """Assess current market risks"""
        logger.info("⚠️ Assessing market risks...")

        risk_factors = [
            "Currency fluctuation impact",
            "Global economic slowdown",
            "Increased competition",
            "Talent acquisition challenges",
            "Regulatory changes"
        ]

        # Simple risk assessment based on multiple factors
        risk_score = np.random.uniform(0.3, 0.8)  # Mock risk score

        if risk_score > 0.7:
            assessment = "HIGH"
        elif risk_score > 0.4:
            assessment = "MODERATE"
        else:
            assessment = "LOW"

        return {
            'assessment': assessment,
            'score': risk_score,
            'factors': risk_factors[:3]
        }

    def _calculate_market_sentiment(self, sector_data, competitor_data, economic_data, news_sentiment) -> float:
        """Calculate overall market sentiment score"""
        sentiment_factors = []

        # Sector performance sentiment
        if sector_data:
            avg_sector_performance = np.mean(list(sector_data.values()))
            sentiment_factors.append(min(1.0, max(-1.0, avg_sector_performance / 10)))

        # Competitor sentiment
        if competitor_data:
            competitor_sentiments = [comp.get('sentiment_score', 0.5) for comp in competitor_data.values()]
            if competitor_sentiments:
                sentiment_factors.append(np.mean(competitor_sentiments))

        # News sentiment
        sentiment_factors.append(news_sentiment)

        # Economic indicators sentiment
        if economic_data:
            # Convert economic indicators to sentiment
            econ_sentiment = 0.5  # Neutral default
            sentiment_factors.append(econ_sentiment)

        # Calculate weighted average
        if sentiment_factors:
            overall_sentiment = np.mean(sentiment_factors)
            return round(overall_sentiment, 3)

        return 0.0

    async def _generate_market_insights(self, sector_data, competitor_data, economic_data, market_sentiment) -> Tuple[List[str], List[str]]:
        """Generate AI-powered market insights"""
        try:
            prompt = f"""
            Based on this market analysis data, provide strategic insights:

            Sector Performance: {sector_data}
            Market Sentiment: {market_sentiment}
            Economic Indicators: {economic_data}

            Generate:
            1. Top 5 market opportunities (specific and actionable)
            2. Top 5 market threats (specific risks to watch)

            Format as JSON with keys: opportunities, threats
            Each should be a list of strings.
            """

            response = await openai.AsyncOpenAI(api_key=settings.openai_api_key).chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3
            )

            import json
            insights = json.loads(response.choices[0].message.content)
            return insights.get('opportunities', []), insights.get('threats', [])

        except Exception as e:
            logger.error(f"❌ Error generating market insights: {str(e)}")
            return [
                "Cloud migration acceleration opportunities",
                "AI/ML service expansion potential",
                "Digital transformation demand growth",
                "Emerging market penetration",
                "Strategic partnership opportunities"
            ], [
                "Increased wage inflation pressure",
                "Currency volatility risks",
                "Global economic uncertainty",
                "Talent shortage challenges",
                "Competitive pricing pressure"
            ]

    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()