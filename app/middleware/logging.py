import time
import logging
import uuid
import json
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Log request start
        start_time = time.time()
        await self._log_request(request, request_id)

        # Process request
        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response
            await self._log_response(request, response, request_id, process_time)

            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.4f}"

            return response

        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            await self._log_error(request, e, request_id, process_time)
            raise

    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request details"""

        # Get client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "Unknown")

        # Skip request body logging for now to avoid consuming the stream
        request_body = "<body not logged>"

        # Create log entry
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "headers": dict(request.headers),
            "request_body": request_body,
            "timestamp": time.time()
        }

        # Log with appropriate level
        if self._is_health_check(request.url.path):
            logger.debug(f"Request started: {request.method} {request.url.path} [{request_id}]")
        else:
            logger.info(f"Request started: {request.method} {request.url.path} [{request_id}]")
            logger.debug(f"Request details: {json.dumps(log_data, default=str)}")

    async def _log_response(self, request: Request, response: Response, request_id: str, process_time: float):
        """Log response details"""

        # Create log entry
        log_data = {
            "request_id": request_id,
            "status_code": response.status_code,
            "process_time_seconds": process_time,
            "response_headers": dict(response.headers),
            "timestamp": time.time()
        }

        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = "error"
        elif response.status_code >= 400:
            log_level = "warning"
        else:
            log_level = "info"

        # Log message
        status_emoji = self._get_status_emoji(response.status_code)
        message = f"Request completed: {request.method} {request.url.path} {status_emoji} {response.status_code} ({process_time:.3f}s) [{request_id}]"

        if self._is_health_check(request.url.path):
            logger.debug(message)
        else:
            getattr(logger, log_level)(message)
            logger.debug(f"Response details: {json.dumps(log_data, default=str)}")

    async def _log_error(self, request: Request, error: Exception, request_id: str, process_time: float):
        """Log request errors"""

        log_data = {
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "process_time_seconds": process_time,
            "method": request.method,
            "url": str(request.url),
            "timestamp": time.time()
        }

        logger.error(f"Request failed: {request.method} {request.url.path} - {type(error).__name__}: {error} ({process_time:.3f}s) [{request_id}]")
        logger.debug(f"Error details: {json.dumps(log_data, default=str)}")

    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP"""
        # Check forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _is_health_check(self, path: str) -> bool:
        """Check if request is a health check"""
        health_paths = ["/health", "/ping", "/ready", "/live", "/api/v1/health"]
        return any(health_path in path for health_path in health_paths)

    def _get_status_emoji(self, status_code: int) -> str:
        """Get emoji for status code"""
        if status_code < 300:
            return "✅"
        elif status_code < 400:
            return "↩️"
        elif status_code < 500:
            return "⚠️"
        else:
            return "❌"

class StructuredLogger:
    """
    Structured logger for consistent JSON logging format
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def log_api_request(
        self,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        process_time: float,
        client_ip: str,
        user_agent: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Log API request in structured format"""

        log_entry = {
            "event_type": "api_request",
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "process_time_seconds": process_time,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "error": error,
            "timestamp": time.time()
        }

        if status_code >= 500:
            self.logger.error(json.dumps(log_entry))
        elif status_code >= 400:
            self.logger.warning(json.dumps(log_entry))
        else:
            self.logger.info(json.dumps(log_entry))

    def log_business_event(
        self,
        event_type: str,
        data: dict,
        request_id: Optional[str] = None
    ):
        """Log business events"""

        log_entry = {
            "event_type": event_type,
            "request_id": request_id,
            "data": data,
            "timestamp": time.time()
        }

        self.logger.info(json.dumps(log_entry, default=str))