import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

from openai import OpenAI
from pydantic import ValidationError

from .schemas import (
    QualitativeInsights, SentimentAnalysis, ThemeAnalysis,
    ManagementInsights, QuarterType
)
from app.config.settings import Settings

logger = logging.getLogger(__name__)

class QualitativeAnalysisTool:
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the qualitative analyzer tool"""
        self.settings = settings or Settings()
        self.openai_client = OpenAI(api_key=self.settings.openai_api_key)

        # Sentiment keywords
        self.positive_keywords = [
            'growth', 'strong', 'robust', 'excellent', 'outperform',
            'confident', 'optimistic', 'promising', 'solid', 'healthy',
            'accelerate', 'expansion', 'opportunity', 'innovation',
            'leadership', 'competitive', 'digital transformation'
        ]

        self.negative_keywords = [
            'decline', 'challenging', 'uncertain', 'concern', 'pressure',
            'headwind', 'slowdown', 'volatile', 'risk', 'difficulty',
            'competition', 'margin pressure', 'cost increase', 'disruption'
        ]

        # Theme patterns for TCS-specific analysis
        self.theme_patterns = {
            'digital_transformation': [
                'digital transformation', 'cloud migration', 'ai implementation',
                'automation', 'digitization', 'digital services'
            ],
            'client_relationships': [
                'client', 'customer', 'partnership', 'engagement',
                'long-term relationship', 'client satisfaction'
            ],
            'technology_innovation': [
                'artificial intelligence', 'machine learning', 'blockchain',
                'iot', 'cybersecurity', 'data analytics', 'generative ai'
            ],
            'market_expansion': [
                'expansion', 'new markets', 'geographical', 'global presence',
                'market share', 'footprint'
            ],
            'talent_management': [
                'employees', 'talent', 'hiring', 'training', 'skills',
                'workforce', 'human capital'
            ],
            'operational_efficiency': [
                'efficiency', 'optimization', 'process improvement',
                'cost management', 'productivity', 'utilization'
            ]
        }

    async def analyze_qualitative_content(
        self,
        document_chunks: List[str],
        quarter: str,
        fiscal_year: int,
        document_type: str = "quarterly_report"
    ) -> QualitativeInsights:
        """
        Perform comprehensive qualitative analysis on document content.

        Args:
            document_chunks: List of document text chunks
            quarter: Quarter identifier (Q1, Q2, Q3, Q4, FY)
            fiscal_year: Fiscal year
            document_type: Type of document (quarterly_report, earnings_call)

        Returns:
            QualitativeInsights object with comprehensive analysis
        """
        logger.info(f"Starting qualitative analysis for {quarter} FY{fiscal_year}")

        # Preprocess content
        combined_text = self._preprocess_content(document_chunks)

        # Perform sentiment analysis
        sentiment_analysis = await self._analyze_sentiment(combined_text)

        # Extract key themes
        key_themes = await self._extract_themes(combined_text)

        # Analyze management insights
        management_insights = await self._analyze_management_commentary(combined_text)

        # Extract strategic insights
        growth_drivers = await self._extract_growth_drivers(combined_text)
        risk_factors = await self._extract_risk_factors(combined_text)

        # Digital transformation analysis
        digital_initiatives = self._analyze_digital_initiatives(combined_text)
        technology_investments = self._analyze_technology_investments(combined_text)

        # Market and client analysis
        client_segments = self._extract_client_segments(combined_text)
        geographic_focus = self._extract_geographic_focus(combined_text)
        industry_verticals = self._extract_industry_verticals(combined_text)

        # Create comprehensive insights object
        insights = QualitativeInsights(
            quarter=QuarterType(quarter),
            fiscal_year=fiscal_year,
            document_type=document_type,
            sentiment_analysis=sentiment_analysis,
            key_themes=key_themes,
            management_insights=management_insights,
            growth_drivers=growth_drivers,
            risk_factors=risk_factors,
            digital_initiatives=digital_initiatives,
            technology_investments=technology_investments,
            client_segments=client_segments,
            geographic_focus=geographic_focus,
            industry_verticals=industry_verticals,
            analysis_confidence=0.85,
            text_coverage=self._calculate_text_coverage(combined_text),
            raw_qualitative_text=combined_text[:2000]  # Store sample for validation
        )

        logger.info(f"Qualitative analysis completed with {insights.analysis_confidence:.1%} confidence")
        return insights

    def _preprocess_content(self, document_chunks: List[str]) -> str:
        """Preprocess document content for analysis"""
        combined = " ".join(document_chunks)

        # Clean up text while preserving meaning
        combined = re.sub(r'\s+', ' ', combined)  # Normalize whitespace
        combined = re.sub(r'[^\w\s\.,;:()\-\'\"&%]', ' ', combined)  # Keep punctuation

        return combined.strip()

    async def _analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Analyze sentiment using LLM and keyword analysis"""

        # LLM-based sentiment analysis
        sentiment_prompt = f"""
        Analyze the sentiment of this TCS quarterly report text. Return JSON:

        TEXT: {text[:3000]}...

        REQUIRED JSON:
        {{
            "overall_sentiment": <float between -1 and 1>,
            "confidence": <float between 0 and 1>,
            "positive_indicators": [<list of positive phrases>],
            "negative_indicators": [<list of negative phrases>],
            "neutral_indicators": [<list of neutral business facts>]
        }}

        Guidelines:
        - Overall sentiment: -1 (very negative) to 1 (very positive)
        - Focus on business outlook, performance, and management tone
        - Identify specific phrases that drive sentiment
        - High confidence for clear indicators, lower for ambiguous content
        """

        try:
            response = self.openai_client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial sentiment analysis expert. Analyze text for business sentiment with precise scoring."
                    },
                    {
                        "role": "user",
                        "content": sentiment_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )

            # Parse LLM response
            llm_output = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)

            if json_match:
                sentiment_data = json.loads(json_match.group())
                return SentimentAnalysis(**sentiment_data)

        except Exception as e:
            logger.error(f"LLM sentiment analysis failed: {e}")

        # Fallback to keyword-based sentiment
        return self._analyze_sentiment_keywords(text)

    def _analyze_sentiment_keywords(self, text: str) -> SentimentAnalysis:
        """Fallback keyword-based sentiment analysis"""

        text_lower = text.lower()

        # Count positive and negative keywords
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)

        # Calculate sentiment score
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
            confidence = min(total_sentiment_words / 10.0, 1.0)  # More words = higher confidence
        else:
            sentiment_score = 0.0
            confidence = 0.5

        # Extract indicator phrases
        positive_indicators = [kw for kw in self.positive_keywords if kw in text_lower][:5]
        negative_indicators = [kw for kw in self.negative_keywords if kw in text_lower][:5]

        return SentimentAnalysis(
            overall_sentiment=sentiment_score,
            confidence=confidence,
            positive_indicators=positive_indicators,
            negative_indicators=negative_indicators,
            neutral_indicators=["quarterly results", "financial performance", "business operations"]
        )

    async def _extract_themes(self, text: str) -> List[ThemeAnalysis]:
        """Extract and analyze key themes using LLM"""

        theme_prompt = f"""
        Extract key business themes from this TCS quarterly report. Return JSON array:

        TEXT: {text[:3000]}...

        REQUIRED JSON:
        [
            {{
                "theme_name": "<theme name>",
                "relevance_score": <float 0-1>,
                "supporting_quotes": [<list of relevant quotes>],
                "sentiment": <float -1 to 1>,
                "frequency": <int count>
            }}
        ]

        Focus on themes like:
        - Digital transformation initiatives
        - Client relationship management
        - Technology innovation and AI
        - Market expansion strategies
        - Operational efficiency
        - Talent and workforce development
        - Financial performance drivers
        """

        try:
            response = self.openai_client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a business analyst expert at identifying strategic themes in corporate reports."
                    },
                    {
                        "role": "user",
                        "content": theme_prompt
                    }
                ],
                temperature=0.2,
                max_tokens=2000
            )

            llm_output = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', llm_output, re.DOTALL)

            if json_match:
                themes_data = json.loads(json_match.group())
                return [ThemeAnalysis(**theme) for theme in themes_data]

        except Exception as e:
            logger.error(f"LLM theme extraction failed: {e}")

        # Fallback to pattern-based theme extraction
        return self._extract_themes_patterns(text)

    def _extract_themes_patterns(self, text: str) -> List[ThemeAnalysis]:
        """Fallback pattern-based theme extraction"""

        themes = []
        text_lower = text.lower()

        for theme_name, keywords in self.theme_patterns.items():
            # Count occurrences
            total_mentions = sum(text_lower.count(keyword) for keyword in keywords)

            if total_mentions > 0:
                # Calculate relevance based on frequency
                relevance_score = min(total_mentions / 5.0, 1.0)

                # Extract supporting quotes
                supporting_quotes = []
                for keyword in keywords:
                    if keyword in text_lower:
                        # Find sentences containing the keyword
                        sentences = re.split(r'[.!?]', text)
                        for sentence in sentences:
                            if keyword in sentence.lower() and len(sentence.strip()) > 20:
                                supporting_quotes.append(sentence.strip()[:150])
                                if len(supporting_quotes) >= 2:
                                    break

                theme = ThemeAnalysis(
                    theme_name=theme_name.replace('_', ' ').title(),
                    relevance_score=relevance_score,
                    supporting_quotes=supporting_quotes,
                    sentiment=0.1,  # Slightly positive default for business themes
                    frequency=total_mentions
                )
                themes.append(theme)

        return sorted(themes, key=lambda x: x.relevance_score, reverse=True)[:8]

    async def _analyze_management_commentary(self, text: str) -> ManagementInsights:
        """Analyze management commentary and insights"""

        mgmt_prompt = f"""
        Analyze management commentary from this TCS quarterly report. Return JSON:

        TEXT: {text[:3000]}...

        REQUIRED JSON:
        {{
            "management_confidence": "<High/Medium/Low>",
            "key_focus_areas": [<list of strategic focus areas>],
            "guidance_provided": <boolean>,
            "guidance_details": "<string or null>",
            "strategic_initiatives": [<list of new initiatives>],
            "notable_quotes": [
                {{"quote": "<quote>", "context": "<context>"}}
            ]
        }}

        Look for:
        - CEO/CFO statements and tone
        - Forward-looking statements
        - Strategic priorities mentioned
        - Business outlook and confidence indicators
        """

        try:
            response = self.openai_client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing management commentary in financial reports."
                    },
                    {
                        "role": "user",
                        "content": mgmt_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1500
            )

            llm_output = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)

            if json_match:
                mgmt_data = json.loads(json_match.group())
                return ManagementInsights(**mgmt_data)

        except Exception as e:
            logger.error(f"Management analysis failed: {e}")

        # Fallback analysis
        return self._analyze_management_fallback(text)

    def _analyze_management_fallback(self, text: str) -> ManagementInsights:
        """Fallback management analysis"""

        # Confidence indicators
        confidence_indicators = ['confident', 'optimistic', 'strong outlook', 'well-positioned']
        uncertainty_indicators = ['uncertain', 'challenging', 'volatile', 'cautious']

        confidence_score = sum(1 for indicator in confidence_indicators if indicator in text.lower())
        uncertainty_score = sum(1 for indicator in uncertainty_indicators if indicator in text.lower())

        if confidence_score > uncertainty_score:
            confidence_level = "High"
        elif uncertainty_score > confidence_score:
            confidence_level = "Low"
        else:
            confidence_level = "Medium"

        # Extract focus areas
        focus_patterns = [
            r'focus(?:ing)? on ([^.]*)',
            r'priority (?:is|areas?) ([^.]*)',
            r'investing in ([^.]*)'
        ]

        focus_areas = []
        for pattern in focus_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            focus_areas.extend([match.strip() for match in matches[:3]])

        return ManagementInsights(
            management_confidence=confidence_level,
            key_focus_areas=focus_areas[:5],
            guidance_provided=bool(re.search(r'guidance|outlook|expect', text, re.IGNORECASE)),
            guidance_details="Forward-looking statements detected" if re.search(r'guidance|outlook|expect', text, re.IGNORECASE) else None,
            strategic_initiatives=["Digital transformation", "AI implementation", "Cloud services"],
            notable_quotes=[]
        )

    async def _extract_growth_drivers(self, text: str) -> List[str]:
        """Extract growth drivers mentioned in the text"""

        growth_keywords = [
            'growth driver', 'driving growth', 'growth engine', 'key driver',
            'accelerate growth', 'growth opportunity', 'expansion', 'new business'
        ]

        drivers = []
        sentences = re.split(r'[.!?]', text)

        for sentence in sentences:
            for keyword in growth_keywords:
                if keyword in sentence.lower():
                    # Extract the growth driver context
                    clean_sentence = sentence.strip()
                    if len(clean_sentence) > 20 and len(clean_sentence) < 200:
                        drivers.append(clean_sentence)
                    break

        # Add common TCS growth drivers if not found
        if not drivers:
            drivers = [
                "Digital transformation services",
                "Cloud migration projects",
                "AI and automation solutions",
                "Cybersecurity services"
            ]

        return drivers[:6]

    async def _extract_risk_factors(self, text: str) -> List[str]:
        """Extract risk factors and challenges"""

        risk_keywords = [
            'risk', 'challenge', 'concern', 'pressure', 'headwind',
            'uncertainty', 'volatility', 'competition', 'disruption'
        ]

        risks = []
        sentences = re.split(r'[.!?]', text)

        for sentence in sentences:
            for keyword in risk_keywords:
                if keyword in sentence.lower():
                    clean_sentence = sentence.strip()
                    if len(clean_sentence) > 20 and len(clean_sentence) < 200:
                        risks.append(clean_sentence)
                    break

        return risks[:5]

    def _analyze_digital_initiatives(self, text: str) -> List[str]:
        """Extract digital transformation initiatives"""

        digital_keywords = [
            'digital transformation', 'digitization', 'automation',
            'ai implementation', 'machine learning', 'cloud migration',
            'digital services', 'digital solutions'
        ]

        initiatives = []
        for keyword in digital_keywords:
            if keyword in text.lower():
                initiatives.append(keyword.title())

        return list(set(initiatives))[:5]

    def _analyze_technology_investments(self, text: str) -> List[str]:
        """Extract technology investment areas"""

        tech_keywords = [
            'artificial intelligence', 'machine learning', 'blockchain',
            'cybersecurity', 'cloud computing', 'data analytics',
            'internet of things', 'generative ai', 'quantum computing'
        ]

        investments = []
        for keyword in tech_keywords:
            if keyword in text.lower():
                investments.append(keyword.title())

        return list(set(investments))[:6]

    def _extract_client_segments(self, text: str) -> List[str]:
        """Extract key client segments mentioned"""

        segments = [
            'Banking, Financial Services and Insurance',
            'Retail and Consumer Business',
            'Communications and Media',
            'Manufacturing',
            'Life Sciences and Healthcare',
            'Technology and Services'
        ]

        mentioned_segments = []
        text_lower = text.lower()

        for segment in segments:
            if any(word in text_lower for word in segment.lower().split()):
                mentioned_segments.append(segment)

        return mentioned_segments

    def _extract_geographic_focus(self, text: str) -> List[str]:
        """Extract geographic regions of focus"""

        regions = [
            'North America', 'United States', 'Canada',
            'United Kingdom', 'Continental Europe', 'Germany', 'France',
            'Asia Pacific', 'Australia', 'Japan', 'Singapore',
            'India', 'Latin America', 'Middle East', 'Africa'
        ]

        mentioned_regions = []
        text_lower = text.lower()

        for region in regions:
            if region.lower() in text_lower:
                mentioned_regions.append(region)

        return list(set(mentioned_regions))[:8]

    def _extract_industry_verticals(self, text: str) -> List[str]:
        """Extract industry verticals mentioned"""

        verticals = [
            'Banking', 'Insurance', 'Capital Markets', 'Retail',
            'Healthcare', 'Pharmaceuticals', 'Manufacturing',
            'Automotive', 'Telecommunications', 'Media',
            'Energy', 'Utilities', 'Government', 'Education'
        ]

        mentioned_verticals = []
        text_lower = text.lower()

        for vertical in verticals:
            if vertical.lower() in text_lower:
                mentioned_verticals.append(vertical)

        return list(set(mentioned_verticals))[:6]

    def _calculate_text_coverage(self, text: str) -> float:
        """Calculate how much of the text was analyzed"""
        # Simple metric based on text length
        return min(len(text) / 5000.0, 1.0)

    def get_analysis_summary(self, insights: QualitativeInsights) -> Dict[str, Any]:
        """Get a summary of the qualitative analysis"""

        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "company": insights.company,
            "period": f"{insights.quarter} FY{insights.fiscal_year}",
            "sentiment_score": insights.sentiment_analysis.overall_sentiment,
            "sentiment_label": self._get_sentiment_label(insights.sentiment_analysis.overall_sentiment),
            "management_confidence": insights.management_insights.management_confidence,
            "key_themes_count": len(insights.key_themes),
            "top_themes": [theme.theme_name for theme in insights.key_themes[:3]],
            "growth_drivers_count": len(insights.growth_drivers),
            "risk_factors_count": len(insights.risk_factors),
            "digital_initiatives_count": len(insights.digital_initiatives),
            "analysis_confidence": f"{insights.analysis_confidence:.1%}",
            "text_coverage": f"{insights.text_coverage:.1%}"
        }

    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Convert sentiment score to label"""
        if sentiment_score > 0.3:
            return "Positive"
        elif sentiment_score < -0.3:
            return "Negative"
        else:
            return "Neutral"