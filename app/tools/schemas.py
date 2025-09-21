# File: app/tools/schemas.py
# Purpose: Pydantic schemas for financial analysis tools
# Dependencies: pydantic, typing, datetime
# Author: AI Assistant
# Date: 2025-09-18
# Phase: 4

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class QuarterType(str, Enum):
    """Enumeration for quarter types"""
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    FY = "FY"  # Full Year

class CurrencyType(str, Enum):
    """Supported currency types"""
    INR = "INR"
    USD = "USD"
    EUR = "EUR"

class SegmentPerformance(BaseModel):
    """Schema for business segment performance data"""
    segment_name: str = Field(..., description="Name of the business segment")
    revenue: float = Field(..., description="Segment revenue in millions")
    revenue_growth_yoy: Optional[float] = Field(None, description="Year-over-year revenue growth percentage")
    revenue_growth_qoq: Optional[float] = Field(None, description="Quarter-over-quarter revenue growth percentage")
    operating_margin: Optional[float] = Field(None, description="Operating margin percentage for the segment")
    percentage_of_total: Optional[float] = Field(None, description="Percentage of total company revenue")
    key_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional segment-specific metrics")

    @validator('revenue_growth_yoy', 'revenue_growth_qoq', 'operating_margin', 'percentage_of_total')
    def validate_percentages(cls, v):
        if v is not None and (v < -100 or v > 1000):
            raise ValueError('Percentage values should be reasonable (-100% to 1000%)')
        return v

class GeographicPerformance(BaseModel):
    """Schema for geographic segment performance"""
    region: str = Field(..., description="Geographic region name")
    revenue: float = Field(..., description="Regional revenue in millions")
    revenue_growth_yoy: Optional[float] = Field(None, description="Year-over-year revenue growth percentage")
    percentage_of_total: Optional[float] = Field(None, description="Percentage of total company revenue")
    key_markets: Optional[List[str]] = Field(default_factory=list, description="Key markets in this region")

class FinancialRatios(BaseModel):
    """Schema for key financial ratios"""
    current_ratio: Optional[float] = Field(None, description="Current assets / Current liabilities")
    debt_to_equity: Optional[float] = Field(None, description="Total debt / Total equity")
    roe: Optional[float] = Field(None, description="Return on Equity percentage")
    roa: Optional[float] = Field(None, description="Return on Assets percentage")
    asset_turnover: Optional[float] = Field(None, description="Revenue / Average Total Assets")
    interest_coverage: Optional[float] = Field(None, description="EBIT / Interest Expense")

class FinancialMetrics(BaseModel):
    """Comprehensive schema for financial metrics extraction"""

    # Basic Information
    company: str = Field(default="TCS", description="Company name")
    quarter: QuarterType = Field(..., description="Quarter being analyzed")
    fiscal_year: int = Field(..., description="Fiscal year")
    reporting_currency: CurrencyType = Field(default=CurrencyType.INR, description="Reporting currency")
    extraction_timestamp: datetime = Field(default_factory=datetime.now, description="When this data was extracted")

    # Revenue Metrics (in millions)
    revenue: float = Field(..., description="Total revenue for the period")
    revenue_growth_yoy: Optional[float] = Field(None, description="Year-over-year revenue growth percentage")
    revenue_growth_qoq: Optional[float] = Field(None, description="Quarter-over-quarter revenue growth percentage")

    # Profitability Metrics (in millions)
    gross_profit: Optional[float] = Field(None, description="Gross profit")
    operating_profit: Optional[float] = Field(None, description="Operating profit/EBIT")
    net_profit: float = Field(..., description="Net profit after tax")
    ebitda: Optional[float] = Field(None, description="Earnings before interest, taxes, depreciation, and amortization")

    # Margin Analysis (percentages)
    gross_margin: Optional[float] = Field(None, description="Gross profit margin percentage")
    operating_margin: Optional[float] = Field(None, description="Operating margin percentage")
    net_margin: float = Field(..., description="Net profit margin percentage")
    ebitda_margin: Optional[float] = Field(None, description="EBITDA margin percentage")

    # Growth Metrics
    net_profit_growth_yoy: Optional[float] = Field(None, description="Year-over-year net profit growth percentage")
    net_profit_growth_qoq: Optional[float] = Field(None, description="Quarter-over-quarter net profit growth percentage")

    # Segment Performance
    segments: Optional[List[SegmentPerformance]] = Field(default_factory=list, description="Business segment performance")
    geographic_segments: Optional[List[GeographicPerformance]] = Field(default_factory=list, description="Geographic performance")

    # Additional Financial Data
    total_assets: Optional[float] = Field(None, description="Total assets in millions")
    total_equity: Optional[float] = Field(None, description="Total shareholders equity in millions")
    total_debt: Optional[float] = Field(None, description="Total debt in millions")
    cash_and_equivalents: Optional[float] = Field(None, description="Cash and cash equivalents in millions")
    free_cash_flow: Optional[float] = Field(None, description="Free cash flow in millions")

    # Financial Ratios
    ratios: Optional[FinancialRatios] = Field(None, description="Key financial ratios")

    # Employee and Operational Metrics
    employee_count: Optional[int] = Field(None, description="Total number of employees")
    revenue_per_employee: Optional[float] = Field(None, description="Revenue per employee in thousands")

    # Confidence and Quality Metrics
    extraction_confidence: float = Field(default=0.0, description="Confidence score of extraction (0-1)")
    data_completeness: float = Field(default=0.0, description="Percentage of fields successfully extracted")
    source_pages: Optional[List[int]] = Field(default_factory=list, description="Source page numbers from document")

    # Raw extracted text for validation
    raw_financial_text: Optional[str] = Field(None, description="Raw text sections used for extraction")

    @validator('fiscal_year')
    def validate_fiscal_year(cls, v):
        if v < 2000 or v > 2030:
            raise ValueError('Fiscal year should be between 2000 and 2030')
        return v

    @validator('extraction_confidence', 'data_completeness')
    def validate_confidence_scores(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Confidence scores should be between 0 and 1')
        return v

    def calculate_completeness(self) -> float:
        """Calculate data completeness based on filled fields"""
        total_fields = 0
        filled_fields = 0

        for field_name, field_value in self.__dict__.items():
            if field_name not in ['extraction_timestamp', 'extraction_confidence', 'data_completeness']:
                total_fields += 1
                if field_value is not None:
                    filled_fields += 1

        return filled_fields / total_fields if total_fields > 0 else 0.0

class SentimentAnalysis(BaseModel):
    """Schema for sentiment analysis results"""
    overall_sentiment: float = Field(..., description="Overall sentiment score (-1 to 1)")
    confidence: float = Field(..., description="Confidence of sentiment analysis (0-1)")
    positive_indicators: List[str] = Field(default_factory=list, description="Positive sentiment indicators")
    negative_indicators: List[str] = Field(default_factory=list, description="Negative sentiment indicators")
    neutral_indicators: List[str] = Field(default_factory=list, description="Neutral indicators")

class ThemeAnalysis(BaseModel):
    """Schema for thematic analysis of qualitative content"""
    theme_name: str = Field(..., description="Name of the identified theme")
    relevance_score: float = Field(..., description="Relevance score (0-1)")
    supporting_quotes: List[str] = Field(default_factory=list, description="Supporting quotes from the text")
    sentiment: float = Field(default=0.0, description="Sentiment associated with this theme (-1 to 1)")
    frequency: int = Field(default=1, description="Number of times theme was mentioned")

class ManagementInsights(BaseModel):
    """Schema for management commentary and insights"""
    management_confidence: str = Field(..., description="Assessment of management confidence (High/Medium/Low)")
    key_focus_areas: List[str] = Field(default_factory=list, description="Key areas management is focusing on")
    guidance_provided: bool = Field(default=False, description="Whether forward guidance was provided")
    guidance_details: Optional[str] = Field(None, description="Details of any guidance provided")
    strategic_initiatives: List[str] = Field(default_factory=list, description="New strategic initiatives mentioned")
    notable_quotes: List[Dict[str, str]] = Field(default_factory=list, description="Notable quotes with context")

class QualitativeInsights(BaseModel):
    """Comprehensive schema for qualitative analysis results"""

    # Basic Information
    company: str = Field(default="TCS", description="Company name")
    quarter: QuarterType = Field(..., description="Quarter being analyzed")
    fiscal_year: int = Field(..., description="Fiscal year")
    document_type: str = Field(..., description="Type of document analyzed (earnings_call, quarterly_report)")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="When this analysis was performed")

    # Sentiment Analysis
    sentiment_analysis: SentimentAnalysis = Field(..., description="Comprehensive sentiment analysis")

    # Thematic Analysis
    key_themes: List[ThemeAnalysis] = Field(default_factory=list, description="Identified themes and their analysis")

    # Management Insights
    management_insights: ManagementInsights = Field(..., description="Management commentary analysis")

    # Growth and Strategy Analysis
    growth_drivers: List[str] = Field(default_factory=list, description="Identified growth drivers")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors and challenges")
    competitive_positioning: Optional[str] = Field(None, description="Assessment of competitive positioning")
    market_outlook: Optional[str] = Field(None, description="Management's view on market outlook")

    # Digital Transformation and Technology Focus
    digital_initiatives: List[str] = Field(default_factory=list, description="Digital transformation initiatives")
    technology_investments: List[str] = Field(default_factory=list, description="Technology investment areas")
    innovation_highlights: List[str] = Field(default_factory=list, description="Innovation and R&D highlights")

    # Client and Market Insights
    client_segments: List[str] = Field(default_factory=list, description="Key client segments mentioned")
    geographic_focus: List[str] = Field(default_factory=list, description="Geographic regions of focus")
    industry_verticals: List[str] = Field(default_factory=list, description="Key industry verticals mentioned")

    # Quality Metrics
    analysis_confidence: float = Field(default=0.0, description="Overall confidence in the analysis (0-1)")
    text_coverage: float = Field(default=0.0, description="Percentage of source text analyzed")
    source_sections: Optional[List[str]] = Field(default_factory=list, description="Document sections analyzed")

    # Raw data for validation
    raw_qualitative_text: Optional[str] = Field(None, description="Raw text sections used for analysis")

    @validator('analysis_confidence', 'text_coverage')
    def validate_scores(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Scores should be between 0 and 1')
        return v

class CombinedAnalysis(BaseModel):
    """Schema for combined financial and qualitative analysis"""

    # Component Analyses
    financial_metrics: FinancialMetrics = Field(..., description="Financial analysis results")
    qualitative_insights: QualitativeInsights = Field(..., description="Qualitative analysis results")

    # Synthesis
    overall_assessment: str = Field(..., description="Overall assessment combining both analyses")
    consistency_score: float = Field(..., description="Consistency between financial and qualitative data (0-1)")
    key_takeaways: List[str] = Field(default_factory=list, description="Key combined insights")

    # Forward-looking Analysis
    growth_outlook: str = Field(..., description="Growth outlook based on combined analysis")
    risk_assessment: str = Field(..., description="Risk assessment combining quantitative and qualitative factors")

    # Confidence Metrics
    overall_confidence: float = Field(..., description="Overall confidence in the combined analysis (0-1)")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="When this combined analysis was performed")

    @validator('consistency_score', 'overall_confidence')
    def validate_confidence_scores(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Confidence scores should be between 0 and 1')
        return v