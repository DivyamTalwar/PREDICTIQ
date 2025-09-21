import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean,
    JSON, ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy.sql import func

Base = declarative_base()

class RequestStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class DocumentType(str, Enum):
    QUARTERLY_REPORT = "quarterly_report"
    ANNUAL_REPORT = "annual_report"
    EARNINGS_TRANSCRIPT = "earnings_transcript"
    INVESTOR_PRESENTATION = "investor_presentation"
    PRESS_RELEASE = "press_release"
    URL_CONTENT = "url_content"

class AgentType(str, Enum):
    FINANCIAL_ANALYZER = "financial_analyzer"
    QUALITATIVE_ANALYZER = "qualitative_analyzer"
    FULL_ANALYSIS = "full_analysis"
    FORECAST_GENERATOR = "forecast_generator"

class RequestLog(Base):
    __tablename__ = "requests_log"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)
    user_query = Column(Text, nullable=False)
    agent_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, default=RequestStatus.PENDING, index=True)
    processing_time_ms = Column(Integer, nullable=True)

    client_ip = Column(String(45), nullable=True) 
    user_agent = Column(String(512), nullable=True)
    api_version = Column(String(10), nullable=True)

    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    tokens_used = Column(Integer, nullable=True)
    documents_processed = Column(Integer, nullable=True, default=0)

    responses = relationship("ResponseLog", back_populates="request", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_timestamp_status', 'timestamp', 'status'),
        Index('idx_agent_type_timestamp', 'agent_type', 'timestamp'),
        CheckConstraint('processing_time_ms >= 0', name='chk_positive_processing_time'),
    )

    def __repr__(self):
        return f"<RequestLog(id={self.id}, status={self.status}, query='{self.user_query[:50]}...')>"

class ResponseLog(Base):
    __tablename__ = "responses_log"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(CHAR(36), ForeignKey("requests_log.id"), nullable=False, index=True)
    response_json = Column(JSON, nullable=False)
    confidence_score = Column(Float, nullable=True)
    tools_used = Column(JSON, nullable=True) 

    created_at = Column(DateTime, default=func.now(), nullable=False)
    response_size_bytes = Column(Integer, nullable=True)

    completeness_score = Column(Float, nullable=True)  
    relevance_score = Column(Float, nullable=True)     

    request = relationship("RequestLog", back_populates="responses")

    # Indexes
    __table_args__ = (
        Index('idx_request_created', 'request_id', 'created_at'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='chk_confidence_range'),
        CheckConstraint('completeness_score >= 0 AND completeness_score <= 1', name='chk_completeness_range'),
        CheckConstraint('relevance_score >= 0 AND relevance_score <= 1', name='chk_relevance_range'),
    )

    def __repr__(self):
        return f"<ResponseLog(id={self.id}, request_id={self.request_id}, confidence={self.confidence_score})>"

class DocumentCache(Base):
    __tablename__ = "document_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_type = Column(String(30), nullable=False, index=True)
    source_url = Column(String(2048), nullable=True)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)  # MD5 or SHA256

    file_path = Column(String(1024), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    original_filename = Column(String(512), nullable=True)

    download_timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)
    last_accessed = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=True)

    doc_metadata = Column(JSON, nullable=True)

    is_processed = Column(Boolean, default=False, nullable=False, index=True)
    processing_error = Column(Text, nullable=True)
    chunks_count = Column(Integer, nullable=True, default=0)

    company = Column(String(100), nullable=True, index=True, default="TCS")
    quarter = Column(Integer, nullable=True)
    fiscal_year = Column(Integer, nullable=True, index=True)

    __table_args__ = (
        Index('idx_company_quarter_year', 'company', 'quarter', 'fiscal_year'),
        Index('idx_document_type_download', 'document_type', 'download_timestamp'),
        Index('idx_processed_timestamp', 'is_processed', 'download_timestamp'),
        CheckConstraint('file_size_bytes > 0', name='chk_positive_file_size'),
        CheckConstraint('quarter >= 1 AND quarter <= 4', name='chk_valid_quarter'),
        CheckConstraint('fiscal_year >= 2000 AND fiscal_year <= 2050', name='chk_valid_fiscal_year'),
        UniqueConstraint('file_path', name='uq_file_path'),
    )

    def __repr__(self):
        return f"<DocumentCache(id={self.id}, type={self.document_type}, company={self.company}, Q{self.quarter} FY{self.fiscal_year})>"

class FinancialMetrics(Base):
    __tablename__ = "financial_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("document_cache.id"), nullable=True, index=True)

    company = Column(String(100), nullable=False, index=True, default="TCS")
    quarter = Column(Integer, nullable=False)
    fiscal_year = Column(Integer, nullable=False, index=True)
    period_ended = Column(String(50), nullable=True)  # e.g., "March 31, 2024"

    revenue = Column(Float, nullable=True)
    revenue_currency = Column(String(3), nullable=True, default="INR")  # INR, USD, etc.
    revenue_units = Column(String(10), nullable=True, default="crores")  # crores, millions

    net_profit = Column(Float, nullable=True)
    gross_profit = Column(Float, nullable=True)
    operating_profit = Column(Float, nullable=True)
    ebitda = Column(Float, nullable=True)

    net_margin = Column(Float, nullable=True)
    gross_margin = Column(Float, nullable=True)
    operating_margin = Column(Float, nullable=True)
    ebitda_margin = Column(Float, nullable=True)

    revenue_growth_yoy = Column(Float, nullable=True)
    profit_growth_yoy = Column(Float, nullable=True)

    # Segment-wise data (JSON structure)
    segments = Column(JSON, nullable=True)  # {"BFSI": 1200, "Retail": 800, ...}
    geographic_revenue = Column(JSON, nullable=True)  # {"North America": 50%, "Europe": 30%, ...}

    employees_count = Column(Integer, nullable=True)
    total_contract_value = Column(Float, nullable=True)  # TCV for quarter
    book_to_bill_ratio = Column(Float, nullable=True)

    operating_cash_flow = Column(Float, nullable=True)
    free_cash_flow = Column(Float, nullable=True)
    cash_and_equivalents = Column(Float, nullable=True)

    extracted_timestamp = Column(DateTime, default=func.now(), nullable=False)
    extraction_confidence = Column(Float, nullable=True)  # 0.0 to 1.0
    extraction_method = Column(String(50), nullable=True)  # "manual", "llm", "pattern_match"

    is_verified = Column(Boolean, default=False, nullable=False)
    verification_notes = Column(Text, nullable=True)

    document = relationship("DocumentCache")

    __table_args__ = (
        Index('idx_company_quarter_year_metrics', 'company', 'quarter', 'fiscal_year'),
        Index('idx_fiscal_year_quarter', 'fiscal_year', 'quarter'),
        Index('idx_extracted_timestamp', 'extracted_timestamp'),
        UniqueConstraint('company', 'quarter', 'fiscal_year', name='uq_company_period'),
        CheckConstraint('quarter >= 1 AND quarter <= 4', name='chk_valid_quarter_metrics'),
        CheckConstraint('fiscal_year >= 2000 AND fiscal_year <= 2050', name='chk_valid_fiscal_year_metrics'),
        CheckConstraint('extraction_confidence >= 0 AND extraction_confidence <= 1', name='chk_extraction_confidence_range'),
        CheckConstraint('net_margin >= -100 AND net_margin <= 100', name='chk_valid_margin_range'),
    )

    def __repr__(self):
        return f"<FinancialMetrics(company={self.company}, Q{self.quarter} FY{self.fiscal_year}, revenue={self.revenue})>"

class QualitativeInsights(Base):
    __tablename__ = "qualitative_insights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("document_cache.id"), nullable=True, index=True)

    company = Column(String(100), nullable=False, index=True, default="TCS")
    quarter = Column(Integer, nullable=False)
    fiscal_year = Column(Integer, nullable=False, index=True)

    overall_sentiment = Column(Float, nullable=True)  # -1.0 to 1.0
    management_confidence = Column(String(20), nullable=True)  # "high", "medium", "low"
    market_outlook_sentiment = Column(Float, nullable=True)

    key_themes = Column(JSON, nullable=True)  # ["digital transformation", "AI growth", ...]
    growth_drivers = Column(JSON, nullable=True)
    risk_factors = Column(JSON, nullable=True)
    strategic_initiatives = Column(JSON, nullable=True)

    notable_quotes = Column(JSON, nullable=True)  # [{"speaker": "CEO", "quote": "...", "context": "..."}]
    forward_looking_statements = Column(JSON, nullable=True)

    competitive_mentions = Column(JSON, nullable=True)  # Mentions of competitors
    market_position_indicators = Column(JSON, nullable=True)

    client_mentions = Column(JSON, nullable=True)
    deal_pipeline_indicators = Column(JSON, nullable=True)

    technology_focus_areas = Column(JSON, nullable=True)  # ["AI/ML", "Cloud", "Automation", ...]
    innovation_investments = Column(JSON, nullable=True)

    analyzed_timestamp = Column(DateTime, default=func.now(), nullable=False)
    analysis_confidence = Column(Float, nullable=True)  # 0.0 to 1.0
    analysis_method = Column(String(50), nullable=True)  # "llm", "rule_based", "hybrid"

    source_sections = Column(JSON, nullable=True)  # Which sections were analyzed
    word_count = Column(Integer, nullable=True)

    document = relationship("DocumentCache")

    __table_args__ = (
        Index('idx_company_quarter_year_insights', 'company', 'quarter', 'fiscal_year'),
        Index('idx_sentiment_analysis', 'overall_sentiment', 'analyzed_timestamp'),
        CheckConstraint('quarter >= 1 AND quarter <= 4', name='chk_valid_quarter_insights'),
        CheckConstraint('fiscal_year >= 2000 AND fiscal_year <= 2050', name='chk_valid_fiscal_year_insights'),
        CheckConstraint('overall_sentiment >= -1 AND overall_sentiment <= 1', name='chk_sentiment_range'),
        CheckConstraint('analysis_confidence >= 0 AND analysis_confidence <= 1', name='chk_analysis_confidence_range'),
        UniqueConstraint('company', 'quarter', 'fiscal_year', name='uq_company_period_insights'),
    )

    def __repr__(self):
        return f"<QualitativeInsights(company={self.company}, Q{self.quarter} FY{self.fiscal_year}, sentiment={self.overall_sentiment})>"

class SystemMetrics(Base):
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)  # "ms", "bytes", "count", "%"

    recorded_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    category = Column(String(50), nullable=True, index=True)  # "performance", "usage", "error"

    # Additional context
    metric_metadata = Column(JSON, nullable=True)

    # Indexes
    __table_args__ = (
        Index('idx_metric_name_recorded', 'metric_name', 'recorded_at'),
        Index('idx_category_recorded', 'category', 'recorded_at'),
    )

    def __repr__(self):
        return f"<SystemMetrics(name={self.metric_name}, value={self.metric_value}, recorded_at={self.recorded_at})>"