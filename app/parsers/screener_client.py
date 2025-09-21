import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re

from app.config import settings

logger = logging.getLogger(__name__)

class ScreenerClient:

    def __init__(self):
        self.session = requests.Session()
        self.base_url = "https://www.screener.in"
        self.data_dir = settings.data_directory / "screener"
        self.cache_dir = settings.data_directory / "cache"
        self.last_request_time = 0
        self.rate_limit_delay = 2.0  # seconds between requests

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._setup_session()

    def _setup_session(self) -> None:
        """Configure session with proper headers and authentication."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session.headers.update(headers)

        # Set cookies if provided
        if settings.screener_cookies:
            self._set_cookies(settings.screener_cookies)

    def _set_cookies(self, cookie_string: str) -> None:
        """Parse and set cookies from string format."""
        try:
            for cookie in cookie_string.split(';'):
                if '=' in cookie:
                    name, value = cookie.strip().split('=', 1)
                    self.session.cookies.set(name, value, domain='.screener.in')
            logger.info("Authentication cookies configured")
        except Exception as e:
            logger.error(f"Failed to set cookies: {e}")

    def _rate_limit(self) -> None:
        """Implement rate limiting to avoid being blocked."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _make_request(self, url: str, timeout: int = 30) -> Optional[requests.Response]:
        """Make HTTP request with error handling and retries."""
        self._rate_limit()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(f"Making request to: {url} (attempt {attempt + 1})")
                response = self.session.get(url, timeout=timeout)

                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    wait_time = (attempt + 1) * 10
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"HTTP {response.status_code}: {response.text[:200]}")

            except requests.exceptions.Timeout:
                logger.error(f"Request timeout on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed on attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)

        return None

    def _generate_filename(self, quarter: str, fiscal_year: int, doc_type: str) -> str:
        """Generate standardized filename for documents."""
        return f"TCS_Q{quarter}_FY{fiscal_year}_{doc_type}.pdf"

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for integrity verification."""
        if not file_path.exists():
            return ""

        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _validate_pdf(self, file_path: Path) -> bool:
        """Validate that downloaded file is a valid PDF."""
        if not file_path.exists():
            return False

        # Check file size (should be > 1KB for valid PDF)
        if file_path.stat().st_size < 1024:
            logger.error(f"File too small: {file_path}")
            return False

        # Check PDF header
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    logger.error(f"Invalid PDF header: {file_path}")
                    return False
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return False

        logger.info(f"PDF validation successful: {file_path}")
        return True

    def _is_cached(self, filename: str, max_age_days: int = 30) -> bool:
        """Check if file is already cached and not too old."""
        file_path = self.data_dir / filename

        if not file_path.exists():
            return False

        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
        if file_age.days > max_age_days:
            logger.info(f"Cached file too old: {filename}")
            return False

        # Validate file integrity
        if not self._validate_pdf(file_path):
            logger.warning(f"Cached file validation failed: {filename}")
            file_path.unlink()  # Remove corrupted file
            return False

        logger.info(f"Using cached file: {filename}")
        return True

    def get_tcs_company_page(self) -> Optional[str]:
        """Get TCS company page URL from Screener.in."""
        search_url = f"{self.base_url}/company/TCS/"
        response = self._make_request(search_url)

        if response:
            return search_url
        return None

    def get_quarterly_reports_urls(self, company_url: str) -> List[Dict[str, str]]:
        """Extract quarterly report download URLs from company page."""
        response = self._make_request(company_url)
        if not response:
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        reports = []

        try:
            # Look for annual report links
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                text = link.get_text(strip=True).lower()

                # Match quarterly and annual reports
                if any(keyword in text for keyword in ['annual report', 'quarterly', 'q1', 'q2', 'q3', 'q4']):
                    if href.endswith('.pdf') or 'pdf' in href:
                        # Extract quarter and year from text
                        quarter_match = re.search(r'q([1-4])', text)
                        year_match = re.search(r'(20\d{2})', text)

                        quarter = quarter_match.group(1) if quarter_match else '4'
                        year = int(year_match.group(1)) if year_match else datetime.now().year

                        reports.append({
                            'url': href if href.startswith('http') else self.base_url + href,
                            'quarter': quarter,
                            'fiscal_year': year,
                            'type': 'quarterly_report',
                            'title': text
                        })

            logger.info(f"Found {len(reports)} quarterly reports")
            return reports[:10]  # Limit to latest 10 reports

        except Exception as e:
            logger.error(f"Error parsing quarterly reports: {e}")
            return []

    def download_document(self, url: str, quarter: str, fiscal_year: int, doc_type: str) -> Optional[Path]:
        """Download document from URL with validation and caching."""
        filename = self._generate_filename(quarter, fiscal_year, doc_type)
        file_path = self.data_dir / filename

        # Check cache first
        if self._is_cached(filename):
            return file_path

        logger.info(f"Downloading {doc_type} for Q{quarter} FY{fiscal_year}")

        response = self._make_request(url, timeout=60)
        if not response:
            logger.error(f"Failed to download: {url}")
            return None

        try:
            # Write file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Validate downloaded file
            if self._validate_pdf(file_path):
                logger.info(f"Successfully downloaded: {filename}")

                # Store metadata
                metadata = {
                    'url': url,
                    'quarter': quarter,
                    'fiscal_year': fiscal_year,
                    'doc_type': doc_type,
                    'download_time': datetime.now().isoformat(),
                    'file_hash': self._get_file_hash(file_path),
                    'file_size': file_path.stat().st_size
                }

                metadata_file = self.cache_dir / f"{filename}.metadata.json"
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

                return file_path
            else:
                logger.error(f"Downloaded file validation failed: {filename}")
                if file_path.exists():
                    file_path.unlink()
                return None

        except Exception as e:
            logger.error(f"Error saving file: {e}")
            if file_path.exists():
                file_path.unlink()
            return None

    def download_tcs_quarterly_reports(self, num_quarters: int = 3) -> List[Path]:
        """Download latest TCS quarterly reports."""
        logger.info(f"Starting download of {num_quarters} TCS quarterly reports")

        # Get company page
        company_url = self.get_tcs_company_page()
        if not company_url:
            logger.error("Failed to get TCS company page")
            return []

        # Get report URLs
        reports = self.get_quarterly_reports_urls(company_url)
        if not reports:
            logger.error("No quarterly reports found")
            return []

        # Download reports
        downloaded_files = []
        for report in reports[:num_quarters]:
            try:
                file_path = self.download_document(
                    report['url'],
                    report['quarter'],
                    report['fiscal_year'],
                    report['type']
                )
                if file_path:
                    downloaded_files.append(file_path)

            except Exception as e:
                logger.error(f"Error downloading report: {e}")
                continue

        logger.info(f"Successfully downloaded {len(downloaded_files)} reports")
        return downloaded_files

    def get_cached_reports(self) -> List[Path]:
        """Get list of all cached report files."""
        pdf_files = list(self.data_dir.glob("*.pdf"))
        valid_files = [f for f in pdf_files if self._validate_pdf(f)]

        logger.info(f"Found {len(valid_files)} cached reports")
        return valid_files

    def cleanup_old_files(self, max_age_days: int = 90) -> None:
        """Remove files older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0

        for file_path in self.data_dir.iterdir():
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    file_path.unlink()
                    removed_count += 1

        logger.info(f"Cleaned up {removed_count} old files")