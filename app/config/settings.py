from typing import Optional, Literal
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    app_name: str = Field(default="TCS Financial Forecasting Agent", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", env="LOG_LEVEL"
    )

    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo", env="OPENAI_MODEL")

    pinecone_api_key: str = Field(default="", env="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="", env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="tcs-financial-rag", env="PINECONE_INDEX_NAME")

    llama_cloud_api_key: str = Field(default="", env="LLAMA_CLOUD_API_KEY")

    jina_api_key: str = Field(default="", env="JINA_API_KEY")

    mysql_host: str = Field(default="localhost", env="MYSQL_HOST")
    mysql_port: int = Field(default=3306, env="MYSQL_PORT")
    mysql_user: str = Field(default="", env="MYSQL_USER")
    mysql_password: str = Field(default="", env="MYSQL_PASSWORD")
    mysql_database: str = Field(default="tcs_financial_db", env="MYSQL_DATABASE")

    screener_cookies: str = Field(default="", env="SCREENER_COOKIES")

    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    max_concurrent_requests: int = Field(default=20, env="MAX_CONCURRENT_REQUESTS")

    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")

    data_directory: Path = Field(default=Path("./data"))
    logs_directory: Path = Field(default=Path("./logs"))
    vectorstore_directory: Path = Field(default=Path("./vectorstore"))

    @validator("mysql_port")
    def validate_mysql_port(cls, v):
        """Validate MySQL port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError("MySQL port must be between 1 and 65535")
        return v

    @validator("rate_limit_per_minute")
    def validate_rate_limit(cls, v):
        """Validate rate limit is positive."""
        if v <= 0:
            raise ValueError("Rate limit must be positive")
        return v

    @property
    def database_url(self) -> str:
        """Generate MySQL database URL for SQLAlchemy."""
        return f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug

    def create_directories(self) -> None:
        """Create required directories if they don't exist."""
        for directory in [self.data_directory, self.logs_directory, self.vectorstore_directory]:
            directory.mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class DevelopmentSettings(Settings):
    """Development-specific settings."""
    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    openai_model: str = "gpt-3.5-turbo"  # Use cheaper model for development


class ProductionSettings(Settings):
    """Production-specific settings."""
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    openai_model: str = "gpt-4-turbo"


class TestSettings(Settings):
    """Test-specific settings."""
    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    mysql_database: str = "tcs_financial_db_test"


def get_settings() -> Settings:
    """
    Factory function to get appropriate settings based on environment.

    Returns:
        Settings: Configuration object based on current environment
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()

    if environment == "production":
        return ProductionSettings()
    elif environment == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()


# Global settings instance
settings = get_settings()