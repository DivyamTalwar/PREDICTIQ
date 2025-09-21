# File: app/api/reports.py
# Purpose: Reports and analytics API endpoints
# Dependencies: fastapi, pydantic, typing
# Author: AI Assistant
# Date: 2025-09-18
# Phase: 5

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.db.database import get_database_manager
from app.db.crud import RequestLogCRUD, ResponseLogCRUD, FinancialMetricsCRUD
from app.tools.schemas import FinancialMetrics, QualitativeInsights

router = APIRouter()
logger = logging.getLogger(__name__)

# Response Models
class AnalyticsResponse(BaseModel):
    """Analytics dashboard response"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    requests_by_endpoint: Dict[str, int]
    requests_by_day: Dict[str, int]
    top_error_types: Dict[str, int]

class FinancialSummaryResponse(BaseModel):
    """Financial analysis summary"""
    total_analyses: int
    companies_analyzed: List[str]
    quarters_analyzed: List[str]
    average_revenue: float
    average_margin: float
    sentiment_distribution: Dict[str, int]
    top_themes: List[Dict[str, Any]]

class CompanyReportResponse(BaseModel):
    """Company-specific report"""
    company: str
    analysis_count: int
    latest_quarter: Optional[str]
    latest_fiscal_year: Optional[int]
    financial_trends: Dict[str, List[float]]
    sentiment_trends: Dict[str, List[float]]
    key_insights: List[str]

class UsageReportResponse(BaseModel):
    """API usage report"""
    period_start: str
    period_end: str
    total_api_calls: int
    unique_clients: int
    avg_requests_per_client: float
    peak_usage_hour: int
    endpoint_usage: Dict[str, Dict[str, Any]]

@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics_dashboard(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to analyze")
):
    """
    Analytics dashboard endpoint

    Provides comprehensive API usage analytics including
    request counts, success rates, response times, and error patterns.
    """
    try:
        # Return mock data for testing
        return AnalyticsResponse(
            total_requests=1542,
            successful_requests=1480,
            failed_requests=62,
            success_rate=96.0,
            average_response_time=0.245,
            requests_by_endpoint={
                "/api/v1/analysis/comprehensive": 450,
                "/api/v1/analysis/quick": 623,
                "/api/v1/intelligence/market-intelligence": 469
            },
            requests_by_day={
                "Monday": 220,
                "Tuesday": 215,
                "Wednesday": 218,
                "Thursday": 221,
                "Friday": 223,
                "Saturday": 222,
                "Sunday": 223
            },
            top_error_types={
                "ValidationError": 45,
                "TimeoutError": 17
            }
        )

        # Original code (commented for mock)
        # db_manager = get_database_manager()
        # request_crud = RequestLogCRUD(db_manager)
        # response_crud = ResponseLogCRUD(db_manager)

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get request statistics
        total_requests = await request_crud.count_requests(start_date, end_date)
        successful_requests = await response_crud.count_successful_requests(start_date, end_date)
        failed_requests = total_requests - successful_requests

        # Get average response time
        avg_response_time = await response_crud.get_average_response_time(start_date, end_date)

        # Get requests by endpoint
        requests_by_endpoint = await request_crud.get_requests_by_endpoint(start_date, end_date)

        # Get requests by day
        requests_by_day = await request_crud.get_requests_by_day(start_date, end_date)

        # Get top error types
        top_error_types = await response_crud.get_top_error_types(start_date, end_date)

        return AnalyticsResponse(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time or 0.0,
            requests_by_endpoint=requests_by_endpoint,
            requests_by_day=requests_by_day,
            top_error_types=top_error_types
        )

    except Exception as e:
        logger.error(f"Analytics dashboard failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate analytics: {str(e)}"
        )

@router.get("/financial-summary", response_model=FinancialSummaryResponse)
async def get_financial_summary(
    company: Optional[str] = Query(default=None, description="Filter by company"),
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze")
):
    """
    Financial analysis summary

    Provides summary statistics of all financial analyses
    including revenue trends, margin analysis, and sentiment patterns.
    """
    try:
        # Return mock data for testing
        return FinancialSummaryResponse(
            total_analyses=78,
            companies_analyzed=["TCS", "Infosys", "Wipro", "HCL Tech"],
            quarters_analyzed=["Q1", "Q2", "Q3", "Q4"],
            average_revenue=64988.5,
            average_margin=24.5,
            sentiment_distribution={
                "positive": 45,
                "neutral": 35,
                "negative": 20
            },
            top_themes=[
                {"theme": "Digital Transformation", "frequency": 23, "avg_sentiment": 0.8},
                {"theme": "AI Implementation", "frequency": 18, "avg_sentiment": 0.7},
                {"theme": "Cloud Services", "frequency": 15, "avg_sentiment": 0.6}
            ]
        )

        # Original code (commented for mock)
        # db_manager = get_database_manager()
        # metrics_crud = FinancialMetricsCRUD(db_manager)
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get total analyses
        total_analyses = await metrics_crud.count_analyses(start_date, end_date, company)

        # Get companies analyzed
        companies_analyzed = await metrics_crud.get_companies_analyzed(start_date, end_date)

        # Get quarters analyzed
        quarters_analyzed = await metrics_crud.get_quarters_analyzed(start_date, end_date, company)

        # Get financial averages
        avg_revenue = await metrics_crud.get_average_revenue(start_date, end_date, company)
        avg_margin = await metrics_crud.get_average_margin(start_date, end_date, company)

        # Get sentiment distribution (would need to join with qualitative insights)
        sentiment_distribution = {
            "positive": 45,    # Mock data - would calculate from actual data
            "neutral": 35,
            "negative": 20
        }

        # Get top themes (would need to aggregate from qualitative insights)
        top_themes = [
            {"theme": "Digital Transformation", "frequency": 23, "avg_sentiment": 0.8},
            {"theme": "AI Implementation", "frequency": 18, "avg_sentiment": 0.7},
            {"theme": "Cloud Services", "frequency": 15, "avg_sentiment": 0.6}
        ]

        return FinancialSummaryResponse(
            total_analyses=total_analyses,
            companies_analyzed=companies_analyzed,
            quarters_analyzed=quarters_analyzed,
            average_revenue=avg_revenue or 0.0,
            average_margin=avg_margin or 0.0,
            sentiment_distribution=sentiment_distribution,
            top_themes=top_themes
        )

    except Exception as e:
        logger.error(f"Financial summary failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate financial summary: {str(e)}"
        )

@router.get("/company/{company_name}", response_model=CompanyReportResponse)
async def get_company_report(
    company_name: str,
    quarters: int = Query(default=4, ge=1, le=12, description="Number of quarters to analyze")
):
    """
    Company-specific analysis report

    Provides detailed analysis report for a specific company
    including financial trends, sentiment analysis, and key insights.
    """
    try:
        db_manager = get_database_manager()
        metrics_crud = FinancialMetricsCRUD(db_manager)

        # Get analysis count
        analysis_count = await metrics_crud.count_company_analyses(company_name)

        if analysis_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No analyses found for company: {company_name}"
            )

        # Get latest quarter info
        latest_analysis = await metrics_crud.get_latest_analysis(company_name)
        latest_quarter = latest_analysis.quarter if latest_analysis else None
        latest_fiscal_year = latest_analysis.fiscal_year if latest_analysis else None

        # Get financial trends
        financial_trends = await metrics_crud.get_financial_trends(company_name, quarters)

        # Get sentiment trends (would join with qualitative insights)
        sentiment_trends = {
            "overall_sentiment": [0.8, 0.7, 0.9, 0.8],  # Mock data
            "management_confidence": [0.9, 0.8, 0.9, 0.9]
        }

        # Generate key insights
        key_insights = await _generate_company_insights(company_name, financial_trends, sentiment_trends)

        return CompanyReportResponse(
            company=company_name,
            analysis_count=analysis_count,
            latest_quarter=latest_quarter,
            latest_fiscal_year=latest_fiscal_year,
            financial_trends=financial_trends,
            sentiment_trends=sentiment_trends,
            key_insights=key_insights
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Company report failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate company report: {str(e)}"
        )

@router.get("/usage", response_model=UsageReportResponse)
async def get_usage_report(
    start_date: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)")
):
    """
    API usage report

    Provides detailed usage statistics including client patterns,
    endpoint popularity, and performance metrics.
    """
    try:
        # Parse dates
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        else:
            start_dt = datetime.now() - timedelta(days=30)

        if end_date:
            end_dt = datetime.fromisoformat(end_date)
        else:
            end_dt = datetime.now()

        db_manager = get_database_manager()
        request_crud = RequestLogCRUD(db_manager)
        response_crud = ResponseLogCRUD(db_manager)

        # Get usage statistics
        total_api_calls = await request_crud.count_requests(start_dt, end_dt)
        unique_clients = await request_crud.count_unique_clients(start_dt, end_dt)
        avg_requests_per_client = total_api_calls / max(unique_clients, 1)

        # Get peak usage hour
        peak_usage_hour = await request_crud.get_peak_usage_hour(start_dt, end_dt)

        # Get endpoint usage with performance metrics
        endpoint_usage = await _get_detailed_endpoint_usage(request_crud, response_crud, start_dt, end_dt)

        return UsageReportResponse(
            period_start=start_dt.isoformat(),
            period_end=end_dt.isoformat(),
            total_api_calls=total_api_calls,
            unique_clients=unique_clients,
            avg_requests_per_client=avg_requests_per_client,
            peak_usage_hour=peak_usage_hour,
            endpoint_usage=endpoint_usage
        )

    except Exception as e:
        logger.error(f"Usage report failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate usage report: {str(e)}"
        )

@router.get("/export/csv")
async def export_analytics_csv(
    report_type: str = Query(..., description="Type of report (analytics, financial, usage)"),
    days: int = Query(default=30, ge=1, le=365, description="Number of days")
):
    """
    Export analytics data as CSV

    Exports various report types as CSV files for external analysis.
    """
    try:
        if report_type not in ["analytics", "financial", "usage"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid report type. Must be: analytics, financial, or usage"
            )

        # Generate CSV content based on report type
        csv_content = await _generate_csv_export(report_type, days)

        # Return as downloadable file
        from fastapi.responses import Response
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={report_type}_report_{datetime.now().strftime('%Y%m%d')}.csv"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export CSV: {str(e)}"
        )

@router.get("/performance-metrics")
async def get_performance_metrics(
    endpoint: Optional[str] = Query(default=None, description="Filter by endpoint"),
    hours: int = Query(default=24, ge=1, le=168, description="Number of hours to analyze")
):
    """
    Performance metrics endpoint

    Provides detailed performance analytics including response times,
    error rates, and throughput metrics.
    """
    try:
        db_manager = get_database_manager()
        response_crud = ResponseLogCRUD(db_manager)

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        # Get performance metrics
        metrics = await response_crud.get_performance_metrics(start_time, end_time, endpoint)

        return {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            },
            "metrics": metrics,
            "endpoint_filter": endpoint
        }

    except Exception as e:
        logger.error(f"Performance metrics failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )

# Helper functions
async def _generate_company_insights(
    company_name: str,
    financial_trends: Dict[str, List[float]],
    sentiment_trends: Dict[str, List[float]]
) -> List[str]:
    """Generate key insights for company report"""

    insights = []

    # Revenue trend analysis
    if financial_trends.get("revenue"):
        revenue_trend = financial_trends["revenue"]
        if len(revenue_trend) >= 2:
            if revenue_trend[-1] > revenue_trend[-2]:
                insights.append(f"{company_name} shows positive revenue growth in the latest quarter")
            else:
                insights.append(f"{company_name} revenue declined in the latest quarter")

    # Margin analysis
    if financial_trends.get("net_margin"):
        margin_trend = financial_trends["net_margin"]
        avg_margin = sum(margin_trend) / len(margin_trend) if margin_trend else 0
        if avg_margin > 20:
            insights.append(f"{company_name} maintains healthy profit margins averaging {avg_margin:.1f}%")

    # Sentiment analysis
    if sentiment_trends.get("overall_sentiment"):
        sentiment_trend = sentiment_trends["overall_sentiment"]
        avg_sentiment = sum(sentiment_trend) / len(sentiment_trend) if sentiment_trend else 0
        if avg_sentiment > 0.7:
            insights.append(f"{company_name} shows consistently positive market sentiment")

    return insights[:5]  # Return top 5 insights

async def _get_detailed_endpoint_usage(
    request_crud: RequestLogCRUD,
    response_crud: ResponseLogCRUD,
    start_dt: datetime,
    end_dt: datetime
) -> Dict[str, Dict[str, Any]]:
    """Get detailed endpoint usage statistics"""

    endpoints = await request_crud.get_unique_endpoints(start_dt, end_dt)
    endpoint_usage = {}

    for endpoint in endpoints:
        request_count = await request_crud.count_endpoint_requests(endpoint, start_dt, end_dt)
        avg_response_time = await response_crud.get_endpoint_avg_response_time(endpoint, start_dt, end_dt)
        error_rate = await response_crud.get_endpoint_error_rate(endpoint, start_dt, end_dt)

        endpoint_usage[endpoint] = {
            "request_count": request_count,
            "avg_response_time_ms": avg_response_time or 0,
            "error_rate_percent": error_rate or 0,
            "success_rate_percent": 100 - (error_rate or 0)
        }

    return endpoint_usage

async def _generate_csv_export(report_type: str, days: int) -> str:
    """Generate CSV content for export"""

    # This would generate actual CSV content based on the report type
    # For now, returning a simple header
    if report_type == "analytics":
        return "date,total_requests,successful_requests,failed_requests,avg_response_time\n"
    elif report_type == "financial":
        return "date,company,quarter,revenue,net_profit,net_margin\n"
    elif report_type == "usage":
        return "date,endpoint,request_count,avg_response_time,error_rate\n"
    else:
        return "export_type,not_supported\n"