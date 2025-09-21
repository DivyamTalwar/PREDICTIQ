# File: app/middleware/rate_limiting.py
# Purpose: Rate limiting middleware for API protection
# Dependencies: fastapi, time, collections
# Author: AI Assistant
# Date: 2025-09-18
# Phase: 5

import time
import logging
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm

    Implements per-IP rate limiting to prevent abuse and ensure
    fair usage of the API resources.
    """

    def __init__(self, app, calls_per_minute: int = 60, burst_allowance: int = 50):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.burst_allowance = burst_allowance
        self.window_size = 60.0  # 1 minute window

        # Store request timestamps per IP
        self.request_history: Dict[str, Deque[float]] = defaultdict(deque)

        # Track burst requests
        self.burst_tracker: Dict[str, Tuple[float, int]] = {}

    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # Check rate limits
        if not self._is_allowed(client_ip, current_time):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.calls_per_minute} requests per minute allowed",
                    "retry_after": self._get_retry_after(client_ip, current_time)
                },
                headers={
                    "Retry-After": str(self._get_retry_after(client_ip, current_time)),
                    "X-RateLimit-Limit": str(self.calls_per_minute),
                    "X-RateLimit-Remaining": str(self._get_remaining_requests(client_ip, current_time)),
                    "X-RateLimit-Reset": str(int(current_time + self.window_size))
                }
            )

        # Record the request
        self._record_request(client_ip, current_time)

        # Process the request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(self.calls_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(self._get_remaining_requests(client_ip, current_time))
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window_size))

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded headers first (proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"

    def _is_allowed(self, client_ip: str, current_time: float) -> bool:
        """Check if request is allowed based on rate limits"""

        # Clean old requests outside the window
        self._cleanup_old_requests(client_ip, current_time)

        # Get current request count
        request_count = len(self.request_history[client_ip])

        # Check regular rate limit
        if request_count >= self.calls_per_minute:
            return False

        # Check burst protection
        if not self._check_burst_limit(client_ip, current_time):
            return False

        return True

    def _check_burst_limit(self, client_ip: str, current_time: float) -> bool:
        """Check burst rate limiting (prevent rapid successive requests)"""

        if client_ip not in self.burst_tracker:
            self.burst_tracker[client_ip] = (current_time, 1)
            return True

        last_time, count = self.burst_tracker[client_ip]

        # Reset burst counter if enough time has passed
        if current_time - last_time > 1.0:  # 1 second burst window
            self.burst_tracker[client_ip] = (current_time, 1)
            return True

        # Check if burst limit exceeded
        if count >= self.burst_allowance:
            return False

        # Update burst counter
        self.burst_tracker[client_ip] = (current_time, count + 1)
        return True

    def _record_request(self, client_ip: str, current_time: float):
        """Record a new request for the client"""
        self.request_history[client_ip].append(current_time)

    def _cleanup_old_requests(self, client_ip: str, current_time: float):
        """Remove requests outside the current window"""
        cutoff_time = current_time - self.window_size

        while (self.request_history[client_ip] and
               self.request_history[client_ip][0] < cutoff_time):
            self.request_history[client_ip].popleft()

    def _get_remaining_requests(self, client_ip: str, current_time: float) -> int:
        """Get number of remaining requests for the client"""
        self._cleanup_old_requests(client_ip, current_time)
        used_requests = len(self.request_history[client_ip])
        return max(0, self.calls_per_minute - used_requests)

    def _get_retry_after(self, client_ip: str, current_time: float) -> int:
        """Get retry-after time in seconds"""
        if not self.request_history[client_ip]:
            return 0

        # Return time until oldest request exits the window
        oldest_request = self.request_history[client_ip][0]
        retry_after = oldest_request + self.window_size - current_time
        return max(1, int(retry_after))

    def get_stats(self) -> Dict[str, int]:
        """Get rate limiting statistics"""
        current_time = time.time()

        # Clean up old data
        for client_ip in list(self.request_history.keys()):
            self._cleanup_old_requests(client_ip, current_time)

        # Calculate stats
        total_ips = len(self.request_history)
        total_requests = sum(len(requests) for requests in self.request_history.values())
        active_ips = len([ip for ip, requests in self.request_history.items() if requests])

        return {
            "total_tracked_ips": total_ips,
            "active_ips_in_window": active_ips,
            "total_requests_in_window": total_requests,
            "rate_limit_per_minute": self.calls_per_minute,
            "burst_allowance": self.burst_allowance
        }