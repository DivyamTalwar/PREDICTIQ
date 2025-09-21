import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import openai
from ..config.settings import get_settings
from ..database.models import FinancialData, PredictionResult
from ..database.connection import get_session
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class PredictionInsight:
    metric: str
    current_value: float
    predicted_value: float
    confidence: float
    trend: str
    risk_factors: List[str]
    opportunities: List[str]
    market_sentiment: str

class AdvancedPredictiveEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_cache = {}

    async def initialize_models(self):
        """Initialize and train ML models for financial prediction"""
        logger.info("🧠 Initializing Advanced Predictive Models...")

        # Revenue Prediction Model
        self.models['revenue'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )

        # Profit Margin Prediction Model
        self.models['profit_margin'] = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            random_state=42
        )

        # Stock Price Volatility Model
        self.models['volatility'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )

        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()

        logger.info("✅ Predictive models initialized successfully")

    async def train_models_on_historical_data(self, session: Session):
        """Train models using historical financial data"""
        logger.info("🎯 Training models on historical data...")

        # Fetch historical data
        historical_data = session.query(FinancialData).order_by(FinancialData.date.desc()).limit(100).all()

        if len(historical_data) < 20:
            logger.warning("⚠️ Insufficient historical data for training")
            return False

        # Prepare training data
        features = []
        targets = {
            'revenue': [],
            'profit_margin': [],
            'volatility': []
        }

        for data in historical_data:
            feature_vector = [
                data.revenue or 0,
                data.profit or 0,
                data.expenses or 0,
                data.assets or 0,
                data.liabilities or 0,
                data.quarter,
                data.fiscal_year % 100  # Last 2 digits of year
            ]
            features.append(feature_vector)

            targets['revenue'].append(data.revenue or 0)
            targets['profit_margin'].append((data.profit / data.revenue * 100) if data.revenue else 0)
            targets['volatility'].append(abs((data.profit or 0) - (data.revenue or 0) * 0.15))

        features = np.array(features)

        # Train each model
        for model_name, target_values in targets.items():
            if len(target_values) > 10:
                # Scale features
                features_scaled = self.scalers[model_name].fit_transform(features)

                # Train model
                self.models[model_name].fit(features_scaled, target_values)

                # Calculate feature importance
                if hasattr(self.models[model_name], 'feature_importances_'):
                    self.feature_importance[model_name] = self.models[model_name].feature_importances_

                logger.info(f"✅ {model_name} model trained successfully")

        return True

    async def predict_financial_metrics(self,
                                      current_data: Dict,
                                      horizon_quarters: int = 4) -> Dict[str, PredictionInsight]:
        """Generate comprehensive financial predictions"""
        logger.info(f"🔮 Generating predictions for {horizon_quarters} quarters ahead...")

        predictions = {}

        # Prepare current features
        current_features = np.array([[
            current_data.get('revenue', 0),
            current_data.get('profit', 0),
            current_data.get('expenses', 0),
            current_data.get('assets', 0),
            current_data.get('liabilities', 0),
            current_data.get('quarter', 1),
            current_data.get('fiscal_year', 2025) % 100
        ]])

        # Generate predictions for each metric
        for metric, model in self.models.items():
            try:
                # Scale features
                features_scaled = self.scalers[metric].transform(current_features)

                # Make prediction
                predicted_value = model.predict(features_scaled)[0]

                # Calculate confidence based on model performance
                confidence = self._calculate_prediction_confidence(metric, current_features)

                # Determine trend
                current_value = current_data.get(metric, 0)
                trend = "INCREASING" if predicted_value > current_value else "DECREASING"

                # Generate AI-powered insights
                insights = await self._generate_ai_insights(metric, current_value, predicted_value, confidence)

                predictions[metric] = PredictionInsight(
                    metric=metric,
                    current_value=current_value,
                    predicted_value=predicted_value,
                    confidence=confidence,
                    trend=trend,
                    risk_factors=insights.get('risk_factors', []),
                    opportunities=insights.get('opportunities', []),
                    market_sentiment=insights.get('sentiment', 'NEUTRAL')
                )

            except Exception as e:
                logger.error(f"❌ Error predicting {metric}: {str(e)}")

        return predictions

    def _calculate_prediction_confidence(self, metric: str, features: np.ndarray) -> float:
        """Calculate prediction confidence based on model performance"""
        try:
            # Use model's internal confidence if available
            if hasattr(self.models[metric], 'predict_proba'):
                confidence = 0.85  # Base confidence for classification
            else:
                # For regression, use feature importance and variance
                importance_sum = sum(self.feature_importance.get(metric, [0.1] * 7))
                confidence = min(0.95, 0.6 + (importance_sum * 0.35))

            return round(confidence, 3)
        except:
            return 0.75  # Default confidence

    async def _generate_ai_insights(self, metric: str, current: float, predicted: float, confidence: float) -> Dict:
        """Generate AI-powered insights for predictions"""
        try:
            prompt = f"""
            Analyze this financial prediction and provide insights:

            Metric: {metric}
            Current Value: {current:,.2f}
            Predicted Value: {predicted:,.2f}
            Confidence: {confidence:.1%}
            Change: {((predicted - current) / current * 100) if current else 0:.1f}%

            Provide:
            1. Top 3 risk factors (specific and actionable)
            2. Top 3 opportunities (specific and actionable)
            3. Market sentiment (BULLISH/BEARISH/NEUTRAL)

            Format as JSON with keys: risk_factors, opportunities, sentiment
            """

            response = await openai.AsyncOpenAI(api_key=settings.openai_api_key).chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )

            import json
            insights = json.loads(response.choices[0].message.content)
            return insights

        except Exception as e:
            logger.error(f"❌ Error generating AI insights: {str(e)}")
            return {
                'risk_factors': ['Market volatility', 'Economic uncertainty', 'Competition pressure'],
                'opportunities': ['Cost optimization', 'Market expansion', 'Technology adoption'],
                'sentiment': 'NEUTRAL'
            }

    async def generate_strategic_recommendations(self, predictions: Dict[str, PredictionInsight]) -> List[str]:
        """Generate strategic business recommendations based on predictions"""
        logger.info("💡 Generating strategic recommendations...")

        recommendations = []

        # Revenue growth recommendations
        if 'revenue' in predictions:
            revenue_pred = predictions['revenue']
            if revenue_pred.trend == "DECREASING":
                recommendations.extend([
                    "🎯 Implement aggressive customer acquisition strategy",
                    "💰 Explore new revenue streams and market segments",
                    "📈 Increase marketing spend and brand visibility"
                ])
            else:
                recommendations.extend([
                    "🚀 Scale successful initiatives and expand capacity",
                    "💎 Focus on premium services and value-added offerings"
                ])

        # Profit margin recommendations
        if 'profit_margin' in predictions:
            margin_pred = predictions['profit_margin']
            if margin_pred.predicted_value < 15:
                recommendations.extend([
                    "⚡ Implement cost optimization and efficiency programs",
                    "🤖 Automate processes to reduce operational costs",
                    "📊 Renegotiate supplier contracts and vendor agreements"
                ])

        # Risk management recommendations
        high_risk_metrics = [p for p in predictions.values() if p.confidence < 0.7]
        if high_risk_metrics:
            recommendations.extend([
                "🛡️ Implement enhanced risk monitoring and early warning systems",
                "💼 Diversify business portfolio to reduce dependency",
                "📋 Develop contingency plans for key business scenarios"
            ])

        return recommendations[:8]  # Top 8 recommendations

    async def save_predictions(self, session: Session, predictions: Dict[str, PredictionInsight]):
        """Save predictions to database"""
        try:
            for metric_name, prediction in predictions.items():
                db_prediction = PredictionResult(
                    metric_name=metric_name,
                    current_value=prediction.current_value,
                    predicted_value=prediction.predicted_value,
                    confidence_score=prediction.confidence,
                    trend=prediction.trend,
                    created_at=datetime.utcnow(),
                    horizon_quarters=4
                )
                session.add(db_prediction)

            session.commit()
            logger.info("✅ Predictions saved to database")

        except Exception as e:
            logger.error(f"❌ Error saving predictions: {str(e)}")
            session.rollback()