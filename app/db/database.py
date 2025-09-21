import logging
from contextlib import contextmanager
from typing import Generator, Optional
import time

from sqlalchemy import create_engine, event, pool, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from sqlalchemy.pool import QueuePool

from app.config import settings
from .models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:

    def __init__(self):
        self.engine: Optional[object] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._setup_database()

    def _setup_database(self) -> None:
        try:
            if not all([settings.mysql_user, settings.mysql_password, settings.mysql_database]):
                logger.info("Using in-memory SQLite for testing/development")
                database_url = "sqlite:///./tcs_financial.db"
            else:

                from urllib.parse import quote_plus
                encoded_password = quote_plus(settings.mysql_password)
                database_url = (
                    f"mysql+pymysql://{settings.mysql_user}:{encoded_password}"
                    f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
                    f"?charset=utf8mb4"
                )

            engine_config = {
                "poolclass": QueuePool,
                "pool_size": 20,  # Number of connections to maintain
                "max_overflow": 30,  # Additional connections beyond pool_size
                "pool_timeout": 30,  # Timeout for getting connection from pool
                "pool_recycle": 3600,  # Recycle connections after 1 hour
                "pool_pre_ping": True,  # Validate connections before use
                "echo": settings.debug,  # Log SQL queries in debug mode
                "echo_pool": settings.debug,  # Log pool events in debug mode
            }

            # MySQL-specific configurations
            if database_url.startswith("mysql"):
                engine_config.update({
                    "connect_args": {
                        "charset": "utf8mb4",
                        "autocommit": False,
                        "connect_timeout": 60,
                        "read_timeout": 60,
                        "write_timeout": 60,
                    }
                })

            self.engine = create_engine(database_url, **engine_config)

            self._setup_event_listeners()

            self.SessionLocal = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )

            logger.info("Database engine and session factory configured successfully")

        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            raise

    def _setup_event_listeners(self) -> None:

        @event.listens_for(self.engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Event listener for new database connections."""
            logger.debug("New database connection established")

        @event.listens_for(self.engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Event listener for connection checkout from pool."""
            logger.debug("Connection checked out from pool")

        @event.listens_for(self.engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Event listener for connection checkin to pool."""
            logger.debug("Connection checked in to pool")

        @event.listens_for(self.engine, "invalidate")
        def on_invalidate(dbapi_connection, connection_record, exception):
            """Event listener for connection invalidation."""
            logger.warning(f"Connection invalidated: {exception}")

    def create_tables(self) -> None:
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def drop_tables(self) -> None:
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise

    def get_session(self) -> Session:
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()

    @contextmanager
    def get_session_context(self) -> Generator[Session, None, None]:
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:
        try:
            with self.get_session_context() as session:
                result = session.execute(text("SELECT 1 as test"))
                test_value = result.scalar()

                if test_value == 1:
                    logger.info("Database connection test successful")
                    return True
                else:
                    logger.error("Database connection test failed: unexpected result")
                    return False

        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def get_connection_info(self) -> dict:
        if not self.engine:
            return {"status": "not_connected"}

        try:
            pool = self.engine.pool
            return {
                "status": "connected",
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid(),
                "total_connections": pool.size() + pool.overflow(),
                "database_url": str(self.engine.url).replace(str(self.engine.url.password), "***")
            }
        except Exception as e:
            logger.error(f"Error getting connection info: {e}")
            return {"status": "error", "error": str(e)}

    def health_check(self) -> dict:
        start_time = time.time()

        health_status = {
            "database": {
                "status": "unknown",
                "response_time_ms": 0,
                "connection_info": {},
                "tables_exist": False,
                "error": None
            }
        }

        try:
            connection_test = self.test_connection()
            health_status["database"]["status"] = "healthy" if connection_test else "unhealthy"

            health_status["database"]["connection_info"] = self.get_connection_info()

            with self.get_session_context() as session:
                try:
                    # Try to query one of our tables
                    session.execute(text("SELECT COUNT(*) FROM requests_log LIMIT 1"))
                    health_status["database"]["tables_exist"] = True
                except Exception:
                    health_status["database"]["tables_exist"] = False

        except Exception as e:
            health_status["database"]["status"] = "error"
            health_status["database"]["error"] = str(e)

        finally:
            health_status["database"]["response_time_ms"] = int((time.time() - start_time) * 1000)

        return health_status

    def execute_raw_sql(self, sql: str, params: dict = None) -> any:
        try:
            with self.get_session_context() as session:
                result = session.execute(text(sql), params or {})
                return result.fetchall()
        except Exception as e:
            logger.error(f"Raw SQL execution failed: {e}")
            raise

    def get_table_sizes(self) -> dict:
        table_sizes = {}

        try:
            with self.get_session_context() as session:
                tables = [
                    "requests_log",
                    "responses_log",
                    "document_cache",
                    "financial_metrics",
                    "qualitative_insights",
                    "system_metrics"
                ]

                for table in tables:
                    try:
                        result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        table_sizes[table] = count
                    except Exception as e:
                        table_sizes[table] = f"Error: {e}"

        except Exception as e:
            logger.error(f"Error getting table sizes: {e}")
            table_sizes["error"] = str(e)

        return table_sizes

    def cleanup_old_logs(self, days_to_keep: int = 30) -> dict:
        cleanup_results = {}

        try:
            with self.get_session_context() as session:
                # Clean old request logs
                request_result = session.execute(
                    text("DELETE FROM requests_log WHERE timestamp < DATE_SUB(NOW(), INTERVAL :days DAY)"),
                    {"days": days_to_keep}
                )
                cleanup_results["requests_deleted"] = request_result.rowcount

                # Clean old system metrics
                metrics_result = session.execute(
                    text("DELETE FROM system_metrics WHERE recorded_at < DATE_SUB(NOW(), INTERVAL :days DAY)"),
                    {"days": days_to_keep}
                )
                cleanup_results["metrics_deleted"] = metrics_result.rowcount

                logger.info(f"Cleanup completed: {cleanup_results}")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            cleanup_results["error"] = str(e)

        return cleanup_results

    def close(self) -> None:
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

db_manager = DatabaseManager()

engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

def get_db() -> Generator[Session, None, None]:
    with db_manager.get_session_context() as session:
        yield session

def init_database() -> None:
    db_manager.create_tables()

def test_database_connection() -> bool:
    return db_manager.test_connection()

def get_database_health() -> dict:
    return db_manager.health_check()

def get_database_manager() -> DatabaseManager:
    return db_manager