import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from urllib.parse import urlparse, urljoin
import re
from typing import Tuple
import httpx
from bs4 import BeautifulSoup
import requests

from app.config import settings

logger = logging.getLogger(__name__)

class URLProcessor:

    def __init__(self):
        self.jina_base_url = "https://r.jina.ai/"
        self.cache_dir = settings.data_directory / "url_cache"
        self.processed_dir = settings.data_directory / "url_processed"

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # HTTP clients
        self.sync_client = requests.Session()
        self.async_client = None

        self._setup_clients()

    def _setup_clients(self) -> None:
        """Setup HTTP clients with proper headers."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }

        self.sync_client.headers.update(headers)

        # Add Jina API key if available
        if settings.jina_api_key:
            headers['Authorization'] = f'Bearer {settings.jina_api_key}'

        logger.info("HTTP clients configured")

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self.async_client is None:
            self.async_client = httpx.AsyncClient(
                headers=self.sync_client.headers,
                timeout=30.0,
                follow_redirects=True
            )
        return self.async_client

    def _generate_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()

    def _is_cached(self, url: str, max_age_hours: int = 24) -> Optional[Dict[str, Any]]:
        """Check if URL content is cached and still valid."""
        cache_key = self._generate_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            # Check cache age
            cache_time = datetime.fromisoformat(cached_data['cached_at'])
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600

            if age_hours <= max_age_hours:
                logger.info(f"Using cached content for: {url}")
                return cached_data
            else:
                logger.info(f"Cache expired for: {url}")
                cache_file.unlink()

        except Exception as e:
            logger.error(f"Error reading cache: {e}")

        return None

    def _save_to_cache(self, url: str, content: Dict[str, Any]) -> None:
        """Save processed content to cache."""
        cache_key = self._generate_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            cache_data = {
                'url': url,
                'cached_at': datetime.now().isoformat(),
                'content': content
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Cached content for: {url}")

        except Exception as e:
            logger.error(f"Error saving to cache: {e}")

    async def _parse_with_jina(self, url: str) -> Optional[Dict[str, Any]]:
        """Parse URL content using Jina Reader API."""
        try:
            jina_url = f"{self.jina_base_url}{url}"

            client = await self._get_async_client()

            logger.info(f"Parsing with Jina: {url}")
            response = await client.get(jina_url)

            if response.status_code == 200:
                content = response.text

                # Parse the Jina response
                result = {
                    'method': 'jina',
                    'raw_content': content,
                    'text': self._clean_text(content),
                    'metadata': {
                        'url': url,
                        'parsed_at': datetime.now().isoformat(),
                        'content_length': len(content),
                        'parser': 'jina_reader'
                    }
                }

                # Extract structured data
                result.update(self._extract_structured_data(content, url))

                logger.info(f"Jina parsing successful for: {url}")
                return result

            else:
                logger.warning(f"Jina API returned {response.status_code} for: {url}")
                return None

        except Exception as e:
            logger.error(f"Jina parsing failed for {url}: {e}")
            return None

    def _parse_with_beautifulsoup(self, url: str) -> Optional[Dict[str, Any]]:
        """Parse URL content using BeautifulSoup."""
        try:
            logger.info(f"Parsing with BeautifulSoup: {url}")
            response = self.sync_client.get(url, timeout=30)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Extract text content
                text = soup.get_text()
                clean_text = self._clean_text(text)

                # Extract structured data
                structured_data = self._extract_html_structure(soup, url)

                result = {
                    'method': 'beautifulsoup',
                    'raw_content': response.text,
                    'text': clean_text,
                    'metadata': {
                        'url': url,
                        'parsed_at': datetime.now().isoformat(),
                        'content_length': len(clean_text),
                        'parser': 'beautifulsoup',
                        'status_code': response.status_code
                    }
                }

                result.update(structured_data)

                logger.info(f"BeautifulSoup parsing successful for: {url}")
                return result

            else:
                logger.warning(f"HTTP {response.status_code} for: {url}")
                return None

        except Exception as e:
            logger.error(f"BeautifulSoup parsing failed for {url}: {e}")
            return None

    def _parse_with_selenium(self, url: str) -> Optional[Dict[str, Any]]:
        """Parse URL content using Selenium for JavaScript-heavy pages."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            logger.info(f"Parsing with Selenium: {url}")

            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')

            driver = webdriver.Chrome(options=chrome_options)

            try:
                driver.get(url)

                # Wait for page to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )

                # Get page source and text content
                page_source = driver.page_source
                text_content = driver.find_element(By.TAG_NAME, "body").text

                # Parse with BeautifulSoup for structure
                soup = BeautifulSoup(page_source, 'html.parser')
                structured_data = self._extract_html_structure(soup, url)

                result = {
                    'method': 'selenium',
                    'raw_content': page_source,
                    'text': self._clean_text(text_content),
                    'metadata': {
                        'url': url,
                        'parsed_at': datetime.now().isoformat(),
                        'content_length': len(text_content),
                        'parser': 'selenium',
                        'page_title': driver.title
                    }
                }

                result.update(structured_data)

                logger.info(f"Selenium parsing successful for: {url}")
                return result

            finally:
                driver.quit()

        except ImportError:
            logger.warning("Selenium not available")
            return None
        except Exception as e:
            logger.error(f"Selenium parsing failed for {url}: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _extract_structured_data(self, content: str, url: str) -> Dict[str, Any]:
        """Extract structured data from content."""
        structured = {
            'title': self._extract_title(content),
            'headings': self._extract_headings(content),
            'links': self._extract_links(content, url),
            'financial_data': self._extract_financial_mentions(content),
            'dates': self._extract_dates(content)
        }

        return structured

    def _extract_html_structure(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract structured data from HTML using BeautifulSoup."""
        structured = {}

        # Title
        title_tag = soup.find('title')
        structured['title'] = title_tag.get_text().strip() if title_tag else ""

        # Headings
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f'h{level}'):
                headings.append({
                    'level': level,
                    'text': heading.get_text().strip()
                })
        structured['headings'] = headings

        # Links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('/'):
                href = urljoin(url, href)
            links.append({
                'text': link.get_text().strip(),
                'url': href
            })
        structured['links'] = links[:50]  # Limit to 50 links

        # Meta tags
        meta_tags = {}
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property', '')
            content = meta.get('content', '')
            if name and content:
                meta_tags[name] = content
        structured['meta_tags'] = meta_tags

        # Tables (important for financial data)
        tables = []
        for table in soup.find_all('table'):
            table_data = []
            for row in table.find_all('tr'):
                row_data = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                if row_data:
                    table_data.append(row_data)
            if table_data:
                tables.append(table_data)
        structured['tables'] = tables

        return structured

    def _extract_title(self, content: str) -> str:
        """Extract title from content."""
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and len(line) < 200:  # Reasonable title length
                return line
        return ""

    def _extract_headings(self, content: str) -> List[str]:
        """Extract headings from content."""
        headings = []
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            # Look for lines that could be headings
            if (line and
                len(line) < 100 and
                not line.endswith('.') and
                (line.isupper() or line.istitle())):
                headings.append(line)

        return headings[:20]  # Limit to 20 headings

    def _extract_links(self, content: str, base_url: str) -> List[Dict[str, str]]:
        """Extract links from content."""
        import re

        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, content)

        links = []
        for url in urls[:20]:  # Limit to 20 URLs
            links.append({
                'url': url,
                'text': url
            })

        return links

    def _extract_financial_mentions(self, content: str) -> List[str]:
        """Extract financial terms and numbers from content."""
        import re

        financial_patterns = [
            r'₹[\d,]+(?:\.\d+)?\s*(?:crore|lakh)?',  # Indian currency
            r'\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?',  # USD
            r'revenue.*?(?:₹|\\$)[\d,]+',  # Revenue mentions
            r'profit.*?(?:₹|\\$)[\d,]+',   # Profit mentions
            r'Q[1-4]\s+(?:FY|20\d{2})',   # Quarter mentions
            r'growth.*?\d+(?:\.\d+)?%',    # Growth percentages
        ]

        mentions = []
        for pattern in financial_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            mentions.extend(matches)

        return mentions[:20]  # Limit results

    def _extract_dates(self, content: str) -> List[str]:
        """Extract date mentions from content."""
        import re

        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD or YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            dates.extend(matches)

        return dates[:10]  # Limit results

    async def process_url(self, url: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Process a single URL with multiple parsing strategies."""
        logger.info(f"Processing URL: {url}")

        # Check cache first
        if use_cache:
            cached_result = self._is_cached(url)
            if cached_result:
                return cached_result['content']

        # Try parsing strategies in order of preference
        strategies = [
            self._parse_with_jina,
            self._parse_with_beautifulsoup,
            self._parse_with_selenium
        ]

        for strategy in strategies:
            try:
                if asyncio.iscoroutinefunction(strategy):
                    result = await strategy(url)
                else:
                    result = strategy(url)

                if result:
                    # Save to cache
                    if use_cache:
                        self._save_to_cache(url, result)

                    logger.info(f"Successfully processed URL with {result.get('method', 'unknown')}: {url}")
                    return result

            except Exception as e:
                logger.error(f"Strategy {strategy.__name__} failed for {url}: {e}")
                continue

        logger.error(f"All parsing strategies failed for: {url}")
        return None

    def process_url_sync(self, url: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Synchronous wrapper for URL processing."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.process_url(url, use_cache))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.process_url(url, use_cache))
            finally:
                loop.close()

    async def process_multiple_urls(self, urls: List[str], use_cache: bool = True, max_concurrent: int = 5) -> Dict[str, Any]:
        """Process multiple URLs concurrently."""
        logger.info(f"Processing {len(urls)} URLs with max {max_concurrent} concurrent requests")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(url: str) -> Tuple[str, Optional[Dict[str, Any]]]:
            async with semaphore:
                result = await self.process_url(url, use_cache)
                return url, result

        # Process URLs concurrently
        tasks = [process_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Organize results
        processed_results = {}
        success_count = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue

            url, content = result
            if content:
                processed_results[url] = content
                success_count += 1
            else:
                logger.warning(f"Failed to process: {url}")

        logger.info(f"Successfully processed {success_count}/{len(urls)} URLs")

        return {
            'processed_urls': processed_results,
            'summary': {
                'total_urls': len(urls),
                'successful': success_count,
                'failed': len(urls) - success_count,
                'processing_time': datetime.now().isoformat()
            }
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.async_client:
            await self.async_client.aclose()
            self.async_client = None

    def clear_cache(self, max_age_days: int = 7) -> int:
        """Clear old cache files."""
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_time < cutoff_date:
                    cache_file.unlink()
                    removed_count += 1
            except Exception as e:
                logger.error(f"Error removing cache file {cache_file}: {e}")

        logger.info(f"Cleared {removed_count} old cache files")
        return removed_count