import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import uuid

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.exc import SQLAlchemyError

from .models import (
    RequestLog, ResponseLog, DocumentCache, FinancialMetrics,
    QualitativeInsights, SystemMetrics, RequestStatus, DocumentType, AgentType
)

logger = logging.getLogger(__name__)

class CRUDError(Exception):
    """Custom exception for CRUD operations."""
    pass

class RequestLogCRUD:
    """CRUD operations for RequestLog table."""

    def __init__(self, db_manager=None):
        """Initialize with optional db_manager."""
        self.db_manager = db_manager

    @staticmethod
    def create(
        db: Session,
        user_query: str,
        agent_type: str,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        api_version: Optional[str] = None
    ) -> RequestLog:
        """Create a new request log entry."""
        try:
            request_log = RequestLog(
                user_query=user_query,
                agent_type=agent_type,
                client_ip=client_ip,
                user_agent=user_agent,
                api_version=api_version,
                started_at=datetime.utcnow()
            )
            db.add(request_log)
            db.commit()
            db.refresh(request_log)
            logger.info(f"Created request log: {request_log.id}")
            return request_log
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating request log: {e}")
            raise CRUDError(f"Failed to create request log: {e}")

    @staticmethod
    def update_status(
        db: Session,
        request_id: str,
        status: RequestStatus,
        processing_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        tokens_used: Optional[int] = None,
        documents_processed: Optional[int] = None
    ) -> Optional[RequestLog]:
        """Update request status and metadata."""
        try:
            request_log = db.query(RequestLog).filter(RequestLog.id == request_id).first()
            if not request_log:
                logger.warning(f"Request log not found: {request_id}")
                return None

            request_log.status = status
            if processing_time_ms is not None:
                request_log.processing_time_ms = processing_time_ms
            if error_message is not None:
                request_log.error_message = error_message
            if tokens_used is not None:
                request_log.tokens_used = tokens_used
            if documents_processed is not None:
                request_log.documents_processed = documents_processed

            if status == RequestStatus.COMPLETED:
                request_log.completed_at = datetime.utcnow()

            db.commit()
            db.refresh(request_log)
            logger.info(f"Updated request log status: {request_id} -> {status}")
            return request_log
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error updating request log: {e}")
            raise CRUDError(f"Failed to update request log: {e}")

    @staticmethod
    def get_by_id(db: Session, request_id: str) -> Optional[RequestLog]:
        """Get request log by ID."""
        try:
            return db.query(RequestLog).filter(RequestLog.id == request_id).first()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching request log: {e}")
            return None

    @staticmethod
    def get_recent(db: Session, limit: int = 100, agent_type: Optional[str] = None) -> List[RequestLog]:
        """Get recent request logs."""
        try:
            query = db.query(RequestLog)
            if agent_type:
                query = query.filter(RequestLog.agent_type == agent_type)
            return query.order_by(desc(RequestLog.timestamp)).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching recent requests: {e}")
            return []

    @staticmethod
    def get_by_status(db: Session, status: RequestStatus, limit: int = 100) -> List[RequestLog]:
        """Get requests by status."""
        try:
            return db.query(RequestLog).filter(
                RequestLog.status == status
            ).order_by(desc(RequestLog.timestamp)).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching requests by status: {e}")
            return []

class ResponseLogCRUD:
    """CRUD operations for ResponseLog table."""

    def __init__(self, db_manager=None):
        """Initialize with optional db_manager."""
        self.db_manager = db_manager

    @staticmethod
    def create(
        db: Session,
        request_id: str,
        response_json: Dict[str, Any],
        confidence_score: Optional[float] = None,
        tools_used: Optional[List[str]] = None,
        completeness_score: Optional[float] = None,
        relevance_score: Optional[float] = None
    ) -> ResponseLog:
        """Create a new response log entry."""
        try:
            response_size = len(str(response_json))
            response_log = ResponseLog(
                request_id=request_id,
                response_json=response_json,
                confidence_score=confidence_score,
                tools_used=tools_used,
                response_size_bytes=response_size,
                completeness_score=completeness_score,
                relevance_score=relevance_score
            )
            db.add(response_log)
            db.commit()
            db.refresh(response_log)
            logger.info(f"Created response log for request: {request_id}")
            return response_log
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating response log: {e}")
            raise CRUDError(f"Failed to create response log: {e}")

    @staticmethod
    def get_by_request_id(db: Session, request_id: str) -> List[ResponseLog]:
        """Get all responses for a request."""
        try:
            return db.query(ResponseLog).filter(
                ResponseLog.request_id == request_id
            ).order_by(desc(ResponseLog.created_at)).all()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching responses: {e}")
            return []

class DocumentCacheCRUD:
    """CRUD operations for DocumentCache table."""

    def __init__(self, db_manager=None):
        """Initialize with optional db_manager."""
        self.db_manager = db_manager

    @staticmethod
    def create(
        db: Session,
        document_type: DocumentType,
        file_path: str,
        file_hash: str,
        file_size_bytes: int,
        source_url: Optional[str] = None,
        original_filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        company: str = "TCS",
        quarter: Optional[int] = None,
        fiscal_year: Optional[int] = None
    ) -> DocumentCache:
        """Create a new document cache entry."""
        try:
            document = DocumentCache(
                document_type=document_type,
                source_url=source_url,
                file_hash=file_hash,
                file_path=file_path,
                file_size_bytes=file_size_bytes,
                original_filename=original_filename,
                metadata=metadata,
                company=company,
                quarter=quarter,
                fiscal_year=fiscal_year
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            logger.info(f"Created document cache entry: {document.id}")
            return document
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating document cache: {e}")
            raise CRUDError(f"Failed to create document cache: {e}")

    @staticmethod
    def get_by_hash(db: Session, file_hash: str) -> Optional[DocumentCache]:
        """Get document by file hash."""
        try:
            return db.query(DocumentCache).filter(DocumentCache.file_hash == file_hash).first()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching document by hash: {e}")
            return None

    @staticmethod
    def get_by_company_period(
        db: Session,
        company: str = "TCS",
        quarter: Optional[int] = None,
        fiscal_year: Optional[int] = None,
        document_type: Optional[DocumentType] = None
    ) -> List[DocumentCache]:
        """Get documents by company and period."""
        try:
            query = db.query(DocumentCache).filter(DocumentCache.company == company)

            if quarter:
                query = query.filter(DocumentCache.quarter == quarter)
            if fiscal_year:
                query = query.filter(DocumentCache.fiscal_year == fiscal_year)
            if document_type:
                query = query.filter(DocumentCache.document_type == document_type)

            return query.order_by(desc(DocumentCache.download_timestamp)).all()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching documents: {e}")
            return []

    @staticmethod
    def mark_processed(
        db: Session,
        document_id: int,
        chunks_count: int = 0,
        processing_error: Optional[str] = None
    ) -> Optional[DocumentCache]:
        """Mark document as processed."""
        try:
            document = db.query(DocumentCache).filter(DocumentCache.id == document_id).first()
            if not document:
                return None

            document.is_processed = processing_error is None
            document.chunks_count = chunks_count
            document.processing_error = processing_error
            document.last_accessed = datetime.utcnow()

            db.commit()
            db.refresh(document)
            return document
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error marking document as processed: {e}")
            return None

    @staticmethod
    def get_unprocessed(db: Session, limit: int = 50) -> List[DocumentCache]:
        """Get unprocessed documents."""
        try:
            return db.query(DocumentCache).filter(
                DocumentCache.is_processed == False
            ).order_by(asc(DocumentCache.download_timestamp)).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching unprocessed documents: {e}")
            return []

class FinancialMetricsCRUD:
    """CRUD operations for FinancialMetrics table."""

    def __init__(self, db_manager=None):
        """Initialize with optional db_manager."""
        self.db_manager = db_manager

    @staticmethod
    def create(
        db: Session,
        company: str,
        quarter: int,
        fiscal_year: int,
        revenue: Optional[float] = None,
        net_profit: Optional[float] = None,
        net_margin: Optional[float] = None,
        revenue_growth_yoy: Optional[float] = None,
        segments: Optional[Dict[str, Any]] = None,
        document_id: Optional[int] = None,
        **kwargs
    ) -> FinancialMetrics:
        """Create or update financial metrics."""
        try:
            # Check if metrics already exist for this period
            existing = db.query(FinancialMetrics).filter(
                and_(
                    FinancialMetrics.company == company,
                    FinancialMetrics.quarter == quarter,
                    FinancialMetrics.fiscal_year == fiscal_year
                )
            ).first()

            if existing:
                # Update existing record
                for key, value in kwargs.items():
                    if hasattr(existing, key) and value is not None:
                        setattr(existing, key, value)

                if revenue is not None:
                    existing.revenue = revenue
                if net_profit is not None:
                    existing.net_profit = net_profit
                if net_margin is not None:
                    existing.net_margin = net_margin
                if revenue_growth_yoy is not None:
                    existing.revenue_growth_yoy = revenue_growth_yoy
                if segments is not None:
                    existing.segments = segments

                existing.extracted_timestamp = datetime.utcnow()
                db.commit()
                db.refresh(existing)
                logger.info(f"Updated financial metrics: {company} Q{quarter} FY{fiscal_year}")
                return existing
            else:
                # Create new record
                metrics = FinancialMetrics(
                    company=company,
                    quarter=quarter,
                    fiscal_year=fiscal_year,
                    revenue=revenue,
                    net_profit=net_profit,
                    net_margin=net_margin,
                    revenue_growth_yoy=revenue_growth_yoy,
                    segments=segments,
                    document_id=document_id,
                    **kwargs
                )
                db.add(metrics)
                db.commit()
                db.refresh(metrics)
                logger.info(f"Created financial metrics: {company} Q{quarter} FY{fiscal_year}")
                return metrics

        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating/updating financial metrics: {e}")
            raise CRUDError(f"Failed to save financial metrics: {e}")

    @staticmethod
    def get_by_period(
        db: Session,
        company: str = "TCS",
        quarter: Optional[int] = None,
        fiscal_year: Optional[int] = None
    ) -> List[FinancialMetrics]:
        """Get financial metrics by period."""
        try:
            query = db.query(FinancialMetrics).filter(FinancialMetrics.company == company)

            if quarter:
                query = query.filter(FinancialMetrics.quarter == quarter)
            if fiscal_year:
                query = query.filter(FinancialMetrics.fiscal_year == fiscal_year)

            return query.order_by(
                desc(FinancialMetrics.fiscal_year),
                desc(FinancialMetrics.quarter)
            ).all()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching financial metrics: {e}")
            return []

    @staticmethod
    def get_latest_quarters(db: Session, company: str = "TCS", count: int = 8) -> List[FinancialMetrics]:
        """Get latest quarters of financial metrics."""
        try:
            return db.query(FinancialMetrics).filter(
                FinancialMetrics.company == company
            ).order_by(
                desc(FinancialMetrics.fiscal_year),
                desc(FinancialMetrics.quarter)
            ).limit(count).all()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching latest financial metrics: {e}")
            return []

    @staticmethod
    def calculate_growth_trends(db: Session, company: str = "TCS", quarters: int = 4) -> Dict[str, Any]:
        """Calculate growth trends from financial metrics."""
        try:
            metrics = FinancialMetricsCRUD.get_latest_quarters(db, company, quarters)
            if len(metrics) < 2:
                return {"error": "Insufficient data for trend calculation"}

            trends = {
                "revenue_trend": [],
                "profit_trend": [],
                "margin_trend": [],
                "quarters": []
            }

            for metric in reversed(metrics):  # Order chronologically
                trends["quarters"].append(f"Q{metric.quarter} FY{metric.fiscal_year}")
                trends["revenue_trend"].append(metric.revenue)
                trends["profit_trend"].append(metric.net_profit)
                trends["margin_trend"].append(metric.net_margin)

            return trends
        except Exception as e:
            logger.error(f"Error calculating growth trends: {e}")
            return {"error": str(e)}

class QualitativeInsightsCRUD:
    """CRUD operations for QualitativeInsights table."""

    def __init__(self, db_manager=None):
        """Initialize with optional db_manager."""
        self.db_manager = db_manager

    @staticmethod
    def create(
        db: Session,
        company: str,
        quarter: int,
        fiscal_year: int,
        overall_sentiment: Optional[float] = None,
        key_themes: Optional[List[str]] = None,
        growth_drivers: Optional[List[str]] = None,
        risk_factors: Optional[List[str]] = None,
        document_id: Optional[int] = None,
        **kwargs
    ) -> QualitativeInsights:
        """Create or update qualitative insights."""
        try:
            # Check if insights already exist
            existing = db.query(QualitativeInsights).filter(
                and_(
                    QualitativeInsights.company == company,
                    QualitativeInsights.quarter == quarter,
                    QualitativeInsights.fiscal_year == fiscal_year
                )
            ).first()

            if existing:
                # Update existing
                if overall_sentiment is not None:
                    existing.overall_sentiment = overall_sentiment
                if key_themes is not None:
                    existing.key_themes = key_themes
                if growth_drivers is not None:
                    existing.growth_drivers = growth_drivers
                if risk_factors is not None:
                    existing.risk_factors = risk_factors

                for key, value in kwargs.items():
                    if hasattr(existing, key) and value is not None:
                        setattr(existing, key, value)

                existing.analyzed_timestamp = datetime.utcnow()
                db.commit()
                db.refresh(existing)
                return existing
            else:
                # Create new
                insights = QualitativeInsights(
                    company=company,
                    quarter=quarter,
                    fiscal_year=fiscal_year,
                    overall_sentiment=overall_sentiment,
                    key_themes=key_themes,
                    growth_drivers=growth_drivers,
                    risk_factors=risk_factors,
                    document_id=document_id,
                    **kwargs
                )
                db.add(insights)
                db.commit()
                db.refresh(insights)
                return insights

        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating qualitative insights: {e}")
            raise CRUDError(f"Failed to save qualitative insights: {e}")

    @staticmethod
    def get_by_period(
        db: Session,
        company: str = "TCS",
        quarter: Optional[int] = None,
        fiscal_year: Optional[int] = None
    ) -> List[QualitativeInsights]:
        """Get qualitative insights by period."""
        try:
            query = db.query(QualitativeInsights).filter(QualitativeInsights.company == company)

            if quarter:
                query = query.filter(QualitativeInsights.quarter == quarter)
            if fiscal_year:
                query = query.filter(QualitativeInsights.fiscal_year == fiscal_year)

            return query.order_by(
                desc(QualitativeInsights.fiscal_year),
                desc(QualitativeInsights.quarter)
            ).all()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching qualitative insights: {e}")
            return []

class SystemMetricsCRUD:
    """CRUD operations for SystemMetrics table."""

    def __init__(self, db_manager=None):
        """Initialize with optional db_manager."""
        self.db_manager = db_manager

    @staticmethod
    def create(
        db: Session,
        metric_name: str,
        metric_value: float,
        metric_unit: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SystemMetrics:
        """Create a system metric entry."""
        try:
            metric = SystemMetrics(
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=metric_unit,
                category=category,
                metadata=metadata
            )
            db.add(metric)
            db.commit()
            db.refresh(metric)
            return metric
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating system metric: {e}")
            raise CRUDError(f"Failed to create system metric: {e}")

    @staticmethod
    def get_recent_metrics(
        db: Session,
        metric_name: Optional[str] = None,
        category: Optional[str] = None,
        hours: int = 24,
        limit: int = 1000
    ) -> List[SystemMetrics]:
        """Get recent system metrics."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            query = db.query(SystemMetrics).filter(SystemMetrics.recorded_at >= cutoff_time)

            if metric_name:
                query = query.filter(SystemMetrics.metric_name == metric_name)
            if category:
                query = query.filter(SystemMetrics.category == category)

            return query.order_by(desc(SystemMetrics.recorded_at)).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching system metrics: {e}")
            return []

# Convenience functions
def create_request_log(db: Session, **kwargs) -> RequestLog:
    """Convenience function to create request log."""
    return RequestLogCRUD.create(db, **kwargs)

def create_response_log(db: Session, **kwargs) -> ResponseLog:
    """Convenience function to create response log."""
    return ResponseLogCRUD.create(db, **kwargs)

def create_document_cache(db: Session, **kwargs) -> DocumentCache:
    """Convenience function to create document cache."""
    return DocumentCacheCRUD.create(db, **kwargs)

def create_financial_metrics(db: Session, **kwargs) -> FinancialMetrics:
    """Convenience function to create financial metrics."""
    return FinancialMetricsCRUD.create(db, **kwargs)

def create_qualitative_insights(db: Session, **kwargs) -> QualitativeInsights:
    """Convenience function to create qualitative insights."""
    return QualitativeInsightsCRUD.create(db, **kwargs)

def log_system_metric(db: Session, **kwargs) -> SystemMetrics:
    """Convenience function to log system metric."""
    return SystemMetricsCRUD.create(db, **kwargs)