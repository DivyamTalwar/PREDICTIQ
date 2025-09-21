import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

from openai import OpenAI
from pydantic import ValidationError

from .schemas import FinancialMetrics, SegmentPerformance, GeographicPerformance, FinancialRatios, QuarterType, CurrencyType
from app.config.settings import Settings

logger = logging.getLogger(__name__)

class FinancialDataExtractorTool:
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the financial extractor tool"""
        self.settings = settings or Settings()
        self.openai_client = OpenAI(api_key=self.settings.openai_api_key)

        # Regex patterns for financial data extraction
        self.currency_patterns = {
            'inr_crores': r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:cr|crore|crores)',
            'inr_millions': r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:mn|million|millions)',
            'inr_simple': r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            'usd_millions': r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:mn|million|millions)',
            'usd_billions': r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:bn|billion|billions)',
            'percentage': r'(\d+(?:\.\d+)?)\s*%',
            'growth': r'(?:growth|increase|decrease).*?(\d+(?:\.\d+)?)\s*%',
            'margin': r'(?:margin).*?(\d+(?:\.\d+)?)\s*%'
        }

        # Financial statement keywords
        self.financial_keywords = {
            'revenue': ['revenue', 'total income', 'net sales', 'turnover'],
            'profit': ['net profit', 'profit after tax', 'pat', 'net income'],
            'operating_profit': ['operating profit', 'ebit', 'operating income'],
            'ebitda': ['ebitda', 'earnings before interest'],
            'margin': ['margin', 'profitability']
        }

    async def extract_financial_metrics(
        self,
        document_chunks: List[str],
        quarter: str,
        fiscal_year: int,
        use_llm: bool = True
    ) -> FinancialMetrics:
        """
        Extract comprehensive financial metrics from document chunks.

        Args:
            document_chunks: List of document text chunks
            quarter: Quarter identifier (Q1, Q2, Q3, Q4, FY)
            fiscal_year: Fiscal year
            use_llm: Whether to use LLM for extraction (fallback to regex if False)

        Returns:
            FinancialMetrics object with extracted data
        """
        logger.info(f"Starting financial extraction for {quarter} FY{fiscal_year}")

        # Combine and preprocess chunks
        combined_text = self._preprocess_text(document_chunks)

        # Extract financial data using multiple approaches
        if use_llm:
            try:
                metrics = await self._extract_with_llm(combined_text, quarter, fiscal_year)
                logger.info("LLM-based extraction completed successfully")
            except Exception as e:
                logger.error(f"LLM extraction failed: {e}")
                logger.info("Falling back to pattern-based extraction")
                metrics = self._extract_with_patterns(combined_text, quarter, fiscal_year)
        else:
            metrics = self._extract_with_patterns(combined_text, quarter, fiscal_year)

        # Enhance with pattern-based validation
        metrics = self._validate_and_enhance(metrics, combined_text)

        # Calculate derived metrics
        metrics = self._calculate_derived_metrics(metrics)

        # Set quality metrics
        metrics.data_completeness = metrics.calculate_completeness()
        metrics.source_pages = self._extract_page_numbers(document_chunks)

        logger.info(f"Financial extraction completed with {metrics.data_completeness:.2%} completeness")
        return metrics

    def _preprocess_text(self, document_chunks: List[str]) -> str:
        """Preprocess and clean document text"""
        combined = " ".join(document_chunks)

        # Clean up text
        combined = re.sub(r'\s+', ' ', combined)  # Normalize whitespace
        combined = re.sub(r'[^\w\s\.,;:()\-₹$%]', ' ', combined)  # Remove special chars except financial

        return combined.strip()

    async def _extract_with_llm(self, text: str, quarter: str, fiscal_year: int) -> FinancialMetrics:
        """Extract financial metrics using OpenAI LLM"""

        # Create extraction prompt
        extraction_prompt = self._create_extraction_prompt(text, quarter, fiscal_year)

        try:
            response = self.openai_client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst expert at extracting precise financial data from TCS quarterly reports. Always return valid JSON matching the specified schema."
                    },
                    {
                        "role": "user",
                        "content": extraction_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=4000
            )

            # Parse LLM response
            llm_output = response.choices[0].message.content.strip()

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                financial_data = json.loads(json_str)

                # Clean and validate data before creating object
                # Set default values for required fields
                if 'net_margin' not in financial_data or financial_data['net_margin'] is None:
                    if financial_data.get('revenue', 0) > 0 and financial_data.get('net_profit', 0) > 0:
                        financial_data['net_margin'] = (financial_data['net_profit'] / financial_data['revenue']) * 100
                    else:
                        financial_data['net_margin'] = 0.0

                # Ensure required fields exist
                financial_data.setdefault('revenue', 0.0)
                financial_data.setdefault('net_profit', 0.0)

                # Validate and create FinancialMetrics object
                metrics = FinancialMetrics(**financial_data)
                metrics.extraction_confidence = 0.9  # High confidence for LLM extraction
                return metrics
            else:
                raise ValueError("No valid JSON found in LLM response")

        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            raise

    def _create_extraction_prompt(self, text: str, quarter: str, fiscal_year: int) -> str:
        """Create detailed extraction prompt for LLM"""

        prompt = f"""
Extract financial metrics from this TCS {quarter} FY{fiscal_year} quarterly report text and return as JSON:

TEXT TO ANALYZE:
{text[:4000]}...

REQUIRED JSON SCHEMA:
{{
    "company": "TCS",
    "quarter": "{quarter}",
    "fiscal_year": {fiscal_year},
    "reporting_currency": "INR",
    "revenue": <float in crores>,
    "revenue_growth_yoy": <float percentage>,
    "revenue_growth_qoq": <float percentage>,
    "net_profit": <float in crores>,
    "operating_profit": <float in crores>,
    "net_margin": <float percentage>,
    "operating_margin": <float percentage>,
    "ebitda": <float in crores>,
    "ebitda_margin": <float percentage>,
    "net_profit_growth_yoy": <float percentage>,
    "segments": [
        {{
            "segment_name": "<string>",
            "revenue": <float in crores>,
            "revenue_growth_yoy": <float percentage>,
            "percentage_of_total": <float percentage>
        }}
    ],
    "employee_count": <integer>,
    "extraction_confidence": 0.9
}}

EXTRACTION GUIDELINES:
1. Convert all amounts to crores (₹ crores)
2. Extract percentages as numbers (e.g., 15.5 for 15.5%)
3. Focus on the most recent quarter data
4. Include all major business segments mentioned
5. If exact numbers aren't available, use null
6. Be precise with revenue and profit figures
7. Calculate margins if not explicitly stated

RETURN ONLY VALID JSON:
"""
        return prompt

    def _extract_with_patterns(self, text: str, quarter: str, fiscal_year: int) -> FinancialMetrics:
        """Extract financial metrics using regex patterns"""

        logger.info("Using pattern-based extraction")

        # Initialize metrics with basic info
        metrics = FinancialMetrics(
            quarter=QuarterType(quarter),
            fiscal_year=fiscal_year,
            revenue=0.0,
            net_profit=0.0,
            net_margin=0.0,
            extraction_confidence=0.7  # Lower confidence for pattern matching
        )

        # Extract revenue
        revenue = self._extract_revenue_patterns(text)
        if revenue:
            metrics.revenue = revenue

        # Extract profit metrics
        profit_data = self._extract_profit_patterns(text)
        if profit_data:
            metrics.net_profit = profit_data.get('net_profit', 0.0)
            metrics.operating_profit = profit_data.get('operating_profit')
            metrics.ebitda = profit_data.get('ebitda')

        # Extract margins
        margins = self._extract_margin_patterns(text)
        if margins:
            metrics.net_margin = margins.get('net_margin', 0.0)
            metrics.operating_margin = margins.get('operating_margin')
            metrics.ebitda_margin = margins.get('ebitda_margin')

        # Extract growth rates
        growth_rates = self._extract_growth_patterns(text)
        if growth_rates:
            metrics.revenue_growth_yoy = growth_rates.get('revenue_growth_yoy')
            metrics.net_profit_growth_yoy = growth_rates.get('profit_growth_yoy')

        # Extract segments
        segments = self._extract_segment_patterns(text)
        if segments:
            metrics.segments = segments

        return metrics

    def _extract_revenue_patterns(self, text: str) -> Optional[float]:
        """Extract revenue using pattern matching"""

        revenue_patterns = [
            r'total\s+(?:income|revenue).*?(?:₹|rs)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*cr',
            r'revenue.*?(?:₹|rs)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*cr',
            r'net\s+sales.*?(?:₹|rs)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*cr'
        ]

        for pattern in revenue_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                revenue_str = match.group(1).replace(',', '')
                return float(revenue_str)

        return None

    def _extract_profit_patterns(self, text: str) -> Optional[Dict[str, float]]:
        """Extract profit metrics using patterns"""

        profit_data = {}

        # Net profit patterns
        net_profit_patterns = [
            r'net\s+profit.*?(?:₹|rs)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*cr',
            r'profit\s+after\s+tax.*?(?:₹|rs)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*cr',
            r'pat.*?(?:₹|rs)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*cr'
        ]

        for pattern in net_profit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                profit_data['net_profit'] = float(match.group(1).replace(',', ''))
                break

        # Operating profit patterns
        op_profit_patterns = [
            r'operating\s+profit.*?₹\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*cr',
            r'ebit.*?₹\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*cr'
        ]

        for pattern in op_profit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                profit_data['operating_profit'] = float(match.group(1).replace(',', ''))
                break

        # EBITDA patterns
        ebitda_patterns = [
            r'ebitda.*?₹\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*cr'
        ]

        for pattern in ebitda_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                profit_data['ebitda'] = float(match.group(1).replace(',', ''))
                break

        return profit_data if profit_data else None

    def _extract_margin_patterns(self, text: str) -> Optional[Dict[str, float]]:
        """Extract margin percentages"""

        margins = {}

        # Net margin patterns
        net_margin_patterns = [
            r'net\s+(?:profit\s+)?margin.*?(\d+(?:\.\d+)?)\s*%',
            r'profit\s+margin.*?(\d+(?:\.\d+)?)\s*%'
        ]

        for pattern in net_margin_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                margins['net_margin'] = float(match.group(1))
                break

        # Operating margin patterns
        op_margin_patterns = [
            r'operating\s+margin.*?(\d+(?:\.\d+)?)\s*%'
        ]

        for pattern in op_margin_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                margins['operating_margin'] = float(match.group(1))
                break

        # EBITDA margin patterns
        ebitda_margin_patterns = [
            r'ebitda\s+margin.*?(\d+(?:\.\d+)?)\s*%'
        ]

        for pattern in ebitda_margin_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                margins['ebitda_margin'] = float(match.group(1))
                break

        return margins if margins else None

    def _extract_growth_patterns(self, text: str) -> Optional[Dict[str, float]]:
        """Extract growth rate patterns"""

        growth_data = {}

        # Revenue growth patterns
        revenue_growth_patterns = [
            r'revenue.*?(?:growth|increase).*?(\d+(?:\.\d+)?)\s*%',
            r'(?:growth|increase).*?revenue.*?(\d+(?:\.\d+)?)\s*%',
            r'year.over.year.*?revenue.*?(\d+(?:\.\d+)?)\s*%'
        ]

        for pattern in revenue_growth_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                growth_data['revenue_growth_yoy'] = float(match.group(1))
                break

        # Profit growth patterns
        profit_growth_patterns = [
            r'(?:net\s+)?profit.*?(?:growth|increase).*?(\d+(?:\.\d+)?)\s*%',
            r'(?:growth|increase).*?(?:net\s+)?profit.*?(\d+(?:\.\d+)?)\s*%'
        ]

        for pattern in profit_growth_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                growth_data['profit_growth_yoy'] = float(match.group(1))
                break

        return growth_data if growth_data else None

    def _extract_segment_patterns(self, text: str) -> Optional[List[SegmentPerformance]]:
        """Extract business segment performance data"""

        segments = []

        # Common TCS segment names
        segment_names = [
            'Banking, Financial Services and Insurance',
            'Retail and Consumer Business',
            'Communications and Media',
            'Manufacturing',
            'Technology and Services',
            'Life Sciences and Healthcare'
        ]

        for segment_name in segment_names:
            # Look for segment revenue patterns
            segment_patterns = [
                f'{re.escape(segment_name)}.*?₹\\s*(\\d+(?:,\\d+)*(?:\\.\\d+)?)\\s*cr',
                f'({re.escape(segment_name)}).*?revenue.*?₹\\s*(\\d+(?:,\\d+)*(?:\\.\\d+)?)\\s*cr'
            ]

            for pattern in segment_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 2:
                        revenue_str = match.group(2).replace(',', '')
                    else:
                        revenue_str = match.group(1).replace(',', '')

                    segment = SegmentPerformance(
                        segment_name=segment_name,
                        revenue=float(revenue_str)
                    )
                    segments.append(segment)
                    break

        return segments if segments else None

    def _validate_and_enhance(self, metrics: FinancialMetrics, text: str) -> FinancialMetrics:
        """Validate and enhance extracted metrics"""

        # Calculate missing margins if we have revenue and profits
        if metrics.revenue and metrics.net_profit and not metrics.net_margin:
            metrics.net_margin = (metrics.net_profit / metrics.revenue) * 100

        if metrics.revenue and metrics.operating_profit and not metrics.operating_margin:
            metrics.operating_margin = (metrics.operating_profit / metrics.revenue) * 100

        if metrics.revenue and metrics.ebitda and not metrics.ebitda_margin:
            metrics.ebitda_margin = (metrics.ebitda / metrics.revenue) * 100

        # Add raw text for validation
        metrics.raw_financial_text = text[:1000]  # First 1000 chars for reference

        return metrics

    def _calculate_derived_metrics(self, metrics: FinancialMetrics) -> FinancialMetrics:
        """Calculate derived financial metrics"""

        # Calculate revenue per employee if both available
        if metrics.revenue and metrics.employee_count:
            metrics.revenue_per_employee = (metrics.revenue * 10000000) / metrics.employee_count  # Convert crores to actual amount

        # Set reporting currency
        metrics.reporting_currency = CurrencyType.INR

        return metrics

    def _extract_page_numbers(self, document_chunks: List[str]) -> List[int]:
        """Extract page numbers from document chunks"""
        page_numbers = []

        for chunk in document_chunks:
            # Look for page number patterns
            page_matches = re.findall(r'page\s+(\d+)', chunk, re.IGNORECASE)
            page_numbers.extend([int(p) for p in page_matches])

        return sorted(list(set(page_numbers)))  # Remove duplicates and sort

    def get_extraction_summary(self, metrics: FinancialMetrics) -> Dict[str, Any]:
        """Get a summary of the extraction results"""

        return {
            "extraction_timestamp": datetime.now().isoformat(),
            "company": metrics.company,
            "period": f"{metrics.quarter} FY{metrics.fiscal_year}",
            "key_metrics": {
                "revenue_crores": metrics.revenue,
                "net_profit_crores": metrics.net_profit,
                "net_margin_percent": metrics.net_margin,
                "revenue_growth_yoy": metrics.revenue_growth_yoy
            },
            "segments_extracted": len(metrics.segments) if metrics.segments else 0,
            "data_completeness": f"{metrics.data_completeness:.1%}",
            "extraction_confidence": f"{metrics.extraction_confidence:.1%}",
            "extraction_method": "LLM" if metrics.extraction_confidence > 0.8 else "Pattern-based"
        }