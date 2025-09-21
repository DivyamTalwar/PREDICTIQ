import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from app.config.settings import get_settings
from app.db.database import get_database_manager
from app.api import health, analysis, documents, reports, intelligence_routes
from app.middleware.rate_limiting import RateLimitMiddleware
from app.middleware.logging import LoggingMiddleware
from app.middleware.error_handling import ErrorHandlingMiddleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("TCS Financial Agent API Starting Up...")

    db_manager = get_database_manager()
    try:
        logger.info("Database manager ready")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    settings = get_settings()
    logger.info(f"API Version: {settings.app_version}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info(f"Rate Limit: {settings.rate_limit_per_minute}/min")

    yield

    logger.info("TCS Financial Agent API Shutting Down...")
    try:
        if hasattr(db_manager, 'close'):
            await db_manager.close()
    except:
        pass
    logger.info("Shutdown complete")

def create_application() -> FastAPI:

    settings = get_settings()

    app = FastAPI(
        title="TCS Financial Forecasting Agent",
        description="""
        **Advanced AI-Powered Financial Analysis Platform**

        Comprehensive financial analysis and forecasting for TCS using:
        - **LLM-powered data extraction** from quarterly reports
        - **Sentiment analysis** of management commentary
        - **RAG-based document processing** with Pinecone vector storage
        - **Real-time financial metrics** extraction and validation
        - **Qualitative insights** generation and theme analysis

        **Powered by OpenAI GPT-4, Pinecone, and advanced financial analytics**
        """,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, calls_per_minute=10000, burst_allowance=5000)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["https://your-domain.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0", "your-domain.com", "*.your-domain.com"]
        )

    app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
    app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Financial Analysis"])
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["Document Processing"])
    app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports & Analytics"])
    app.include_router(intelligence_routes.router, prefix="/api/v1", tags=["Advanced Intelligence"])

    @app.get("/", response_model=Dict[str, Any])
    async def root():
        """API root endpoint with system information"""
        return {
            "service": "TCS Financial Forecasting Agent",
            "version": settings.app_version,
            "status": "operational",
            "docs": "/docs",
            "health": "/api/v1/health",
            "endpoints": {
                "financial_analysis": "/api/v1/analysis",
                "document_processing": "/api/v1/documents",
                "reports": "/api/v1/reports"
            },
            "features": [
                "LLM-powered financial extraction",
                "Sentiment analysis",
                "RAG document processing",
                "Real-time metrics",
                "Qualitative insights"
            ]
        }

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Global exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        )

    return app

app = create_application()

if __name__ == "__main__":
    settings = get_settings()

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
        access_log=True
    )