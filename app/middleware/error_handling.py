import logging
import traceback
import time
from typing import Dict, Any

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response

        except HTTPException as e:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.warning(f"HTTP Exception: {e.status_code} - {e.detail} [Request: {request_id}]")
            raise

        except ValidationError as e:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.warning(f"Validation Error: {e} [Request: {request_id}]")

            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": "Validation Error",
                    "message": "Request data validation failed",
                    "details": e.errors(),
                    "request_id": request_id
                }
            )

        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")
            error_id = f"ERR_{int(time.time())}_{request_id[:8]}"

            logger.error(
                f"Unhandled Exception [{error_id}]: {type(e).__name__}: {e}",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "error_id": error_id,
                    "method": request.method,
                    "url": str(request.url),
                    "client_ip": self._get_client_ip(request)
                }
            )

            error_response = self._create_error_response(e, error_id, request_id)
            return JSONResponse(
                status_code=error_response["status_code"],
                content=error_response["content"]
            )

    def _get_client_ip(self, request: Request) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _create_error_response(self, error: Exception, error_id: str, request_id: str) -> Dict[str, Any]:

        error_type = type(error).__name__
        error_message = str(error)

        # Database connection errors
        if any(db_error in error_type.lower() for db_error in ["connection", "database", "sql"]):
            return {
                "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
                "content": {
                    "error": "Service Temporarily Unavailable",
                    "message": "Database service is currently unavailable. Please try again later.",
                    "error_id": error_id,
                    "request_id": request_id
                }
            }

        elif "openai" in error_type.lower() or "api" in error_type.lower():
            return {
                "status_code": status.HTTP_502_BAD_GATEWAY,
                "content": {
                    "error": "External Service Error",
                    "message": "AI service is currently unavailable. Please try again later.",
                    "error_id": error_id,
                    "request_id": request_id
                }
            }

        elif any(vector_error in error_type.lower() for vector_error in ["pinecone", "vector", "index"]):
            return {
                "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
                "content": {
                    "error": "Vector Store Unavailable",
                    "message": "Vector search service is currently unavailable.",
                    "error_id": error_id,
                    "request_id": request_id
                }
            }

        elif any(file_error in error_type.lower() for file_error in ["file", "io", "permission"]):
            return {
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "content": {
                    "error": "File Processing Error",
                    "message": "Unable to process the uploaded file.",
                    "error_id": error_id,
                    "request_id": request_id
                }
            }

        elif "timeout" in error_type.lower():
            return {
                "status_code": status.HTTP_504_GATEWAY_TIMEOUT,
                "content": {
                    "error": "Request Timeout",
                    "message": "The request took too long to process. Please try again.",
                    "error_id": error_id,
                    "request_id": request_id
                }
            }

        elif "memory" in error_type.lower():
            return {
                "status_code": status.HTTP_507_INSUFFICIENT_STORAGE,
                "content": {
                    "error": "Resource Limit Exceeded",
                    "message": "Request requires too many resources. Please try with smaller data.",
                    "error_id": error_id,
                    "request_id": request_id
                }
            }

        else:
            return {
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "content": {
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred. Please contact support if the problem persists.",
                    "error_id": error_id,
                    "request_id": request_id
                }
            }

class ErrorReporter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def report_extraction_error(
        self,
        request_id: str,
        error: Exception,
        document_type: str,
        context: Dict[str, Any]
    ):

        self.logger.error(
            f"Financial extraction failed [Request: {request_id}]",
            extra={
                "request_id": request_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "document_type": document_type,
                "context": context
            },
            exc_info=True
        )

    def report_analysis_error(
        self,
        request_id: str,
        error: Exception,
        analysis_type: str,
        context: Dict[str, Any]
    ):

        self.logger.error(
            f"Qualitative analysis failed [Request: {request_id}]",
            extra={
                "request_id": request_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "analysis_type": analysis_type,
                "context": context
            },
            exc_info=True
        )

    def report_processing_error(
        self,
        request_id: str,
        error: Exception,
        processing_stage: str,
        context: Dict[str, Any]
    ):

        self.logger.error(
            f"Document processing failed [Request: {request_id}]",
            extra={
                "request_id": request_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "processing_stage": processing_stage,
                "context": context
            },
            exc_info=True
        )