import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy import stats
import openai
from ..config.settings import get_settings
from ..database.models import FinancialData, RiskAssessment
from ..database.connection import get_session
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class RiskMetric:
    risk_type: str
    current_level: str  
    probability: float
    impact_score: float
    risk_score: float
    mitigation_strategies: List[str]
    early_warning_indicators: List[str]
    trend: str  

@dataclass
class RiskPortfolio:
    overall_risk_score: float
    risk_category: str
    individual_risks: Dict[str, RiskMetric]
    top_risks: List[str]
    risk_trends: Dict[str, str]
    recommended_actions: List[str]
    confidence_level: float

class AdvancedRiskAnalyzer:
    def __init__(self):
        self.risk_weights = {
            'financial': 0.25,
            'operational': 0.20,
            'market': 0.20,
            'regulatory': 0.15,
            'technology': 0.10,
            'reputation': 0.10
        }
        self.historical_data = []

    async def perform_comprehensive_risk_analysis(self,financial_data: Dict,market_data: Dict,session: Session) -> RiskPortfolio:
        logger.info("Performing comprehensive risk analysis...")

        await self._load_historical_risk_data(session)

        risk_metrics = {}

        risk_metrics['financial'] = await self._analyze_financial_risks(financial_data)

        risk_metrics['operational'] = await self._analyze_operational_risks(financial_data)

        risk_metrics['market'] = await self._analyze_market_risks(market_data)

        risk_metrics['regulatory'] = await self._analyze_regulatory_risks()

        risk_metrics['technology'] = await self._analyze_technology_risks()

        risk_metrics['reputation'] = await self._analyze_reputation_risks(market_data)

        overall_score = self._calculate_overall_risk_score(risk_metrics)
        risk_category = self._categorize_risk_level(overall_score)

        top_risks = self._identify_top_risks(risk_metrics)

        risk_trends = self._analyze_risk_trends(risk_metrics)

        recommendations = await self._generate_risk_recommendations(risk_metrics, overall_score)

        confidence = self._calculate_confidence_level(risk_metrics)

        risk_portfolio = RiskPortfolio(
            overall_risk_score=overall_score,
            risk_category=risk_category,
            individual_risks=risk_metrics,
            top_risks=top_risks,
            risk_trends=risk_trends,
            recommended_actions=recommendations,
            confidence_level=confidence
        )

        # Save to database
        await self._save_risk_assessment(session, risk_portfolio)

        return risk_portfolio

    async def _analyze_financial_risks(self, financial_data: Dict) -> RiskMetric:
        logger.info("💰 Analyzing financial risks...")

        revenue = financial_data.get('revenue', 0)
        profit = financial_data.get('profit', 0)
        assets = financial_data.get('assets', 0)
        liabilities = financial_data.get('liabilities', 0)

        risk_indicators = []
        impact_factors = []

        # Liquidity Risk
        if assets > 0:
            debt_to_asset_ratio = liabilities / assets
            if debt_to_asset_ratio > 0.6:
                risk_indicators.append("High debt-to-asset ratio")
                impact_factors.append(0.8)

        # Profitability Risk
        if revenue > 0:
            profit_margin = (profit / revenue) * 100
            if profit_margin < 10:
                risk_indicators.append("Low profit margin")
                impact_factors.append(0.7)

        # Revenue Concentration Risk
        if revenue < 50000:  # Assuming millions
            risk_indicators.append("Revenue concentration risk")
            impact_factors.append(0.6)

        # Calculate risk metrics
        probability = min(0.9, len(risk_indicators) * 0.3)
        impact_score = np.mean(impact_factors) if impact_factors else 0.3
        risk_score = probability * impact_score

        level = self._determine_risk_level(risk_score)

        mitigation_strategies = [
            "Implement robust cash flow management",
            "Diversify revenue streams",
            "Establish credit facilities and backup funding",
            "Monitor financial ratios regularly",
            "Implement early warning systems"
        ]

        early_warnings = [
            "Cash flow below 30-day operating expenses",
            "Debt service coverage ratio < 1.5",
            "Working capital declining trend",
            "Customer payment delays increasing"
        ]

        return RiskMetric(
            risk_type="Financial",
            current_level=level,
            probability=probability,
            impact_score=impact_score,
            risk_score=risk_score,
            mitigation_strategies=mitigation_strategies[:3],
            early_warning_indicators=early_warnings[:3],
            trend="STABLE"
        )

    async def _analyze_operational_risks(self, financial_data: Dict) -> RiskMetric:
        logger.info("⚙️ Analyzing operational risks...")

        # Operational efficiency indicators
        revenue = financial_data.get('revenue', 0)
        expenses = financial_data.get('expenses', 0)

        risk_indicators = []
        impact_factors = []

        # Efficiency Risk
        if revenue > 0 and expenses > 0:
            expense_ratio = expenses / revenue
            if expense_ratio > 0.8:
                risk_indicators.append("High operational expense ratio")
                impact_factors.append(0.7)

        # Scale Risk
        if revenue < 100000:  # Assuming millions
            risk_indicators.append("Limited operational scale")
            impact_factors.append(0.5)

        # Process Risk (simulated)
        process_maturity = np.random.uniform(0.4, 0.9)  # Mock process maturity score
        if process_maturity < 0.6:
            risk_indicators.append("Low process maturity")
            impact_factors.append(0.6)

        probability = min(0.85, len(risk_indicators) * 0.25 + 0.1)
        impact_score = np.mean(impact_factors) if impact_factors else 0.4
        risk_score = probability * impact_score

        level = self._determine_risk_level(risk_score)

        mitigation_strategies = [
            "Implement process automation and standardization",
            "Establish business continuity and disaster recovery plans",
            "Invest in employee training and development",
            "Deploy monitoring and alerting systems",
            "Create redundancy in critical operations"
        ]

        early_warnings = [
            "Process failure rates increasing",
            "Employee turnover above industry average",
            "System downtime frequency rising",
            "Customer complaints about service quality"
        ]

        return RiskMetric(
            risk_type="Operational",
            current_level=level,
            probability=probability,
            impact_score=impact_score,
            risk_score=risk_score,
            mitigation_strategies=mitigation_strategies[:3],
            early_warning_indicators=early_warnings[:3],
            trend="STABLE"
        )

    async def _analyze_market_risks(self, market_data: Dict) -> RiskMetric:
        logger.info("📊 Analyzing market risks...")

        risk_indicators = []
        impact_factors = []

        # Market sentiment risk
        market_sentiment = market_data.get('market_sentiment', 0.5)
        if market_sentiment < 0.3:
            risk_indicators.append("Negative market sentiment")
            impact_factors.append(0.7)

        # Sector performance risk
        sector_performance = market_data.get('sector_performance', {})
        if sector_performance:
            avg_performance = np.mean(list(sector_performance.values()))
            if avg_performance < -5:
                risk_indicators.append("Poor sector performance")
                impact_factors.append(0.8)

        # Competition risk
        competitor_count = len(market_data.get('competitor_analysis', {}))
        if competitor_count > 5:
            risk_indicators.append("High competition intensity")
            impact_factors.append(0.6)

        probability = min(0.8, len(risk_indicators) * 0.3 + 0.2)
        impact_score = np.mean(impact_factors) if impact_factors else 0.5
        risk_score = probability * impact_score

        level = self._determine_risk_level(risk_score)

        mitigation_strategies = [
            "Develop competitive differentiation strategies",
            "Implement dynamic pricing models",
            "Diversify into emerging markets",
            "Strengthen customer relationships",
            "Monitor competitor activities closely"
        ]

        early_warnings = [
            "Market share declining",
            "Pricing pressure from competitors",
            "Customer acquisition cost increasing",
            "Industry growth rate slowing"
        ]

        return RiskMetric(
            risk_type="Market",
            current_level=level,
            probability=probability,
            impact_score=impact_score,
            risk_score=risk_score,
            mitigation_strategies=mitigation_strategies[:3],
            early_warning_indicators=early_warnings[:3],
            trend="INCREASING"
        )

    async def _analyze_regulatory_risks(self) -> RiskMetric:
        logger.info("📋 Analyzing regulatory risks...")

        # Simulate regulatory risk assessment
        risk_indicators = [
            "Data privacy regulation changes",
            "Tax policy modifications",
            "Industry-specific compliance requirements"
        ]

        probability = 0.4  # Moderate probability
        impact_score = 0.6  # Significant impact
        risk_score = probability * impact_score

        level = self._determine_risk_level(risk_score)

        mitigation_strategies = [
            "Establish regulatory monitoring system",
            "Implement compliance management framework",
            "Engage with industry associations",
            "Regular legal and compliance reviews",
            "Proactive stakeholder engagement"
        ]

        early_warnings = [
            "New regulations proposed in industry",
            "Compliance audit findings",
            "Regulatory body communications",
            "Industry penalty announcements"
        ]

        return RiskMetric(
            risk_type="Regulatory",
            current_level=level,
            probability=probability,
            impact_score=impact_score,
            risk_score=risk_score,
            mitigation_strategies=mitigation_strategies[:3],
            early_warning_indicators=early_warnings[:3],
            trend="STABLE"
        )

    async def _analyze_technology_risks(self) -> RiskMetric:
        logger.info("💻 Analyzing technology risks...")

        # Technology risk factors
        risk_indicators = [
            "Legacy system dependencies",
            "Cybersecurity threats",
            "Technology obsolescence"
        ]

        probability = 0.5  # Moderate to high probability
        impact_score = 0.7  # High impact
        risk_score = probability * impact_score

        level = self._determine_risk_level(risk_score)

        mitigation_strategies = [
            "Implement robust cybersecurity measures",
            "Regular technology infrastructure updates",
            "Develop disaster recovery capabilities",
            "Conduct security audits and penetration testing",
            "Employee cybersecurity training programs"
        ]

        early_warnings = [
            "Increased cybersecurity incidents",
            "System performance degradation",
            "Technology vendor end-of-life notices",
            "Security vulnerability disclosures"
        ]

        return RiskMetric(
            risk_type="Technology",
            current_level=level,
            probability=probability,
            impact_score=impact_score,
            risk_score=risk_score,
            mitigation_strategies=mitigation_strategies[:3],
            early_warning_indicators=early_warnings[:3],
            trend="INCREASING"
        )

    async def _analyze_reputation_risks(self, market_data: Dict) -> RiskMetric:
        logger.info("🏢 Analyzing reputation risks...")

        news_sentiment = market_data.get('news_sentiment', 0.5)

        risk_indicators = []
        impact_factors = []

        if news_sentiment < 0.2:
            risk_indicators.append("Negative media coverage")
            impact_factors.append(0.8)

        if news_sentiment < 0.4:
            risk_indicators.append("Public perception concerns")
            impact_factors.append(0.6)

        probability = min(0.6, len(risk_indicators) * 0.3 + 0.1)
        impact_score = np.mean(impact_factors) if impact_factors else 0.3
        risk_score = probability * impact_score

        level = self._determine_risk_level(risk_score)

        mitigation_strategies = [
            "Implement proactive PR and communication strategy",
            "Monitor social media and news sentiment",
            "Establish crisis communication protocols",
            "Build strong stakeholder relationships",
            "Maintain transparency and ethical practices"
        ]

        early_warnings = [
            "Negative social media mentions increasing",
            "Media coverage tone deteriorating",
            "Customer complaints rising",
            "Employee satisfaction declining"
        ]

        return RiskMetric(
            risk_type="Reputation",
            current_level=level,
            probability=probability,
            impact_score=impact_score,
            risk_score=risk_score,
            mitigation_strategies=mitigation_strategies[:3],
            early_warning_indicators=early_warnings[:3],
            trend="STABLE"
        )

    def _determine_risk_level(self, risk_score: float) -> str:
        if risk_score >= 0.7:
            return "CRITICAL"
        elif risk_score >= 0.5:
            return "HIGH"
        elif risk_score >= 0.3:
            return "MODERATE"
        else:
            return "LOW"

    def _calculate_overall_risk_score(self, risk_metrics: Dict[str, RiskMetric]) -> float:
        total_score = 0
        total_weight = 0

        for risk_type, metric in risk_metrics.items():
            weight = self.risk_weights.get(risk_type, 0.1)
            total_score += metric.risk_score * weight
            total_weight += weight

        return round(total_score / total_weight if total_weight > 0 else 0, 3)

    def _categorize_risk_level(self, overall_score: float) -> str:
        return self._determine_risk_level(overall_score)

    def _identify_top_risks(self, risk_metrics: Dict[str, RiskMetric]) -> List[str]:
        sorted_risks = sorted(
            risk_metrics.items(),
            key=lambda x: x[1].risk_score,
            reverse=True
        )
        return [risk_type for risk_type, _ in sorted_risks[:3]]

    def _analyze_risk_trends(self, risk_metrics: Dict[str, RiskMetric]) -> Dict[str, str]:
        return {risk_type: metric.trend for risk_type, metric in risk_metrics.items()}

    async def _generate_risk_recommendations(self, risk_metrics: Dict[str, RiskMetric], overall_score: float) -> List[str]:
        try:
            risk_summary = {
                risk_type: {
                    'level': metric.current_level,
                    'score': metric.risk_score,
                    'trend': metric.trend
                }
                for risk_type, metric in risk_metrics.items()
            }

            prompt = f"""
            Based on this comprehensive risk analysis, provide strategic risk management recommendations:

            Overall Risk Score: {overall_score}
            Risk Breakdown: {risk_summary}

            Generate 6 prioritized strategic recommendations for risk management.
            Focus on:
            1. Immediate actions for high-risk areas
            2. Preventive measures for emerging risks
            3. Strategic initiatives for long-term risk reduction

            Return as a JSON array of strings.
            """

            response = await openai.AsyncOpenAI(api_key=settings.openai_api_key).chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )

            import json
            recommendations = json.loads(response.choices[0].message.content)
            return recommendations[:6]

        except Exception as e:
            logger.error(f"Error generating risk recommendations: {str(e)}")
            return [
                "Implement comprehensive risk monitoring dashboard",
                "Develop crisis response and business continuity plans",
                "Establish regular risk assessment and review cycles",
                "Create risk appetite framework and tolerance levels",
                "Enhance stakeholder communication and transparency",
                "Implement continuous improvement in risk processes"
            ]

    def _calculate_confidence_level(self, risk_metrics: Dict[str, RiskMetric]) -> float:
        base_confidence = 0.75

        completeness_factor = min(1.0, len(risk_metrics) / 6)

        confidence = base_confidence * completeness_factor
        return round(confidence, 3)

    async def _load_historical_risk_data(self, session: Session):
        try:
            recent_assessments = session.query(RiskAssessment).order_by(
                RiskAssessment.created_at.desc()
            ).limit(10).all()

            self.historical_data = [assessment.__dict__ for assessment in recent_assessments]

        except Exception as e:
            logger.warning(f"⚠️ Could not load historical risk data: {str(e)}")
            self.historical_data = []

    async def _save_risk_assessment(self, session: Session, risk_portfolio: RiskPortfolio):
        try:
            assessment = RiskAssessment(
                overall_risk_score=risk_portfolio.overall_risk_score,
                risk_category=risk_portfolio.risk_category,
                top_risks=','.join(risk_portfolio.top_risks),
                confidence_level=risk_portfolio.confidence_level,
                created_at=datetime.utcnow()
            )

            session.add(assessment)
            session.commit()
            logger.info("✅ Risk assessment saved to database")

        except Exception as e:
            logger.error(f"❌ Error saving risk assessment: {str(e)}")
            session.rollback()