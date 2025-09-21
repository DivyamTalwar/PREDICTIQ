import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from llama_parse import LlamaParse

from app.config import settings

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.llama_parse = None
        self.node_parser = None
        self.processed_dir = settings.data_directory / "processed"
        self.chunks_dir = settings.data_directory / "chunks"

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        self._initialize_parsers()

    def _initialize_parsers(self) -> None:
        try:
            if settings.llama_cloud_api_key:
                self.llama_parse = LlamaParse(
                    api_key=settings.llama_cloud_api_key,
                    result_type="markdown",  # Best for structured data
                    num_workers=4,
                    verbose=True,
                    language="en",
                    parsing_instruction="""
                    This is a financial quarterly report. Please:
                    1. Preserve all numerical data and tables exactly
                    2. Maintain structure for financial statements
                    3. Keep section headers and hierarchy
                    4. Extract charts and graphs descriptions
                    5. Preserve currency symbols and units
                    """
                )
                logger.info("LlamaParse initialized successfully")
            else:
                logger.warning("LlamaParse API key not configured")

            self.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=1024,
                chunk_overlap=200,
                include_metadata=True,
                include_prev_next_rel=True
            )

            logger.info("Node parser initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize parsers: {e}")
            raise

    def _extract_financial_metadata(self, content: str, filename: str) -> Dict[str, Any]:
        metadata = {
            "filename": filename,
            "processed_time": datetime.now().isoformat(),
            "document_type": self._identify_document_type(content),
            "quarter_info": self._extract_quarter_info(filename, content),
            "sections": self._identify_sections(content),
            "financial_tables": self._count_financial_tables(content),
            "has_charts": self._has_charts_or_graphs(content)
        }

        return metadata

    def _identify_document_type(self, content: str) -> str:
        content_lower = content.lower()

        if any(term in content_lower for term in ['quarterly results', 'q1', 'q2', 'q3', 'q4']):
            return "quarterly_report"
        elif any(term in content_lower for term in ['annual report', 'annual results']):
            return "annual_report"
        elif any(term in content_lower for term in ['earnings call', 'transcript']):
            return "earnings_transcript"
        elif any(term in content_lower for term in ['investor presentation', 'presentation']):
            return "investor_presentation"
        else:
            return "financial_document"

    def _extract_quarter_info(self, filename: str, content: str) -> Dict[str, Any]:
        import re

        quarter_info = {}

        filename_match = re.search(r'Q(\d)_FY(\d{4})', filename)
        if filename_match:
            quarter_info['quarter'] = int(filename_match.group(1))
            quarter_info['fiscal_year'] = int(filename_match.group(2))

        content_quarter = re.search(r'Q(\d)\s+(FY\s*)?(\d{4})', content)
        if content_quarter:
            quarter_info['quarter'] = int(content_quarter.group(1))
            quarter_info['fiscal_year'] = int(content_quarter.group(3))

        period_match = re.search(r'quarter ended\s+(\w+\s+\d{1,2},?\s+\d{4})', content, re.IGNORECASE)
        if period_match:
            quarter_info['period_ended'] = period_match.group(1)

        return quarter_info

    def _identify_sections(self, content: str) -> List[str]:
        import re

        sections = []

        section_patterns = [
            r'management discussion',
            r'financial highlights',
            r'consolidated results',
            r'segment performance',
            r'balance sheet',
            r'cash flow',
            r'profit.*loss',
            r'revenue analysis',
            r'outlook',
            r'risk factors',
            r'key metrics'
        ]

        for pattern in section_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                sections.append(pattern.replace(r'\.*', ' ').title())

        return sections

    def _count_financial_tables(self, content: str) -> int:
        import re

        table_indicators = [
            r'\|.*\|.*\|',  
            r'revenue.*\d+',
            r'profit.*\d+', 
            r'₹.*crore',    
            r'\$.*million', 
        ]

        table_count = 0
        for pattern in table_indicators:
            matches = re.findall(pattern, content, re.IGNORECASE)
            table_count += len(matches)

        return min(table_count // 10, 50)  # Rough estimate

    def _has_charts_or_graphs(self, content: str) -> bool:
        chart_indicators = ['chart', 'graph', 'figure', 'exhibit', 'visualization']
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in chart_indicators)

    def process_pdf(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return None

        logger.info(f"Processing PDF: {pdf_path.name}")

        try:
            processed_file = self.processed_dir / f"{pdf_path.stem}.json"
            if processed_file.exists():
                logger.info(f"Using cached processed file: {processed_file}")
                with open(processed_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

            if not self.llama_parse:
                raise RuntimeError("LlamaParse not initialized - check LLAMA_CLOUD_API_KEY")

            documents = self.llama_parse.load_data(str(pdf_path))
            logger.info(f"LlamaParse extracted {len(documents)} documents")

            if not documents:
                logger.error("No content extracted from PDF")
                return None

            combined_content = "\n\n".join([doc.text for doc in documents])

            metadata = self._extract_financial_metadata(combined_content, pdf_path.name)

            chunks = self._create_chunks(combined_content, metadata)

            result = {
                "metadata": metadata,
                "raw_content": combined_content,
                "chunks": chunks,
                "processing_stats": {
                    "total_length": len(combined_content),
                    "num_chunks": len(chunks),
                    "avg_chunk_size": len(combined_content) // len(chunks) if chunks else 0
                }
            }

            # Save processed data
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully processed PDF: {pdf_path.name}")
            return result

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path.name}: {e}")
            return None

    def _fallback_pdf_extraction(self, pdf_path: Path) -> List[Document]:
        try:
            import PyPDF2

            documents = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        doc = Document(
                            text=text,
                            metadata={
                                "page_number": page_num + 1,
                                "source": str(pdf_path)
                            }
                        )
                        documents.append(doc)

            logger.info(f"Fallback extraction: {len(documents)} pages")
            return documents

        except ImportError:
            logger.error("PyPDF2 not available for fallback extraction")
            return []
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return []

    def _create_chunks(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.node_parser:
            logger.error("Node parser not initialized")
            return []

        try:
            doc = Document(text=content, metadata=metadata)

            nodes = self.node_parser.get_nodes_from_documents([doc])

            chunks = []
            for i, node in enumerate(nodes):
                chunk_data = {
                    "chunk_id": f"{metadata.get('filename', 'unknown')}_{i}",
                    "text": node.text,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "chunk_size": len(node.text),
                        "start_char_idx": getattr(node, 'start_char_idx', None),
                        "end_char_idx": getattr(node, 'end_char_idx', None)
                    }
                }

                if hasattr(node, 'prev_node') and node.prev_node:
                    chunk_data["metadata"]["prev_chunk"] = f"{metadata.get('filename', 'unknown')}_{i-1}"
                if hasattr(node, 'next_node') and node.next_node:
                    chunk_data["metadata"]["next_chunk"] = f"{metadata.get('filename', 'unknown')}_{i+1}"

                chunks.append(chunk_data)

            chunks_file = self.chunks_dir / f"{metadata.get('filename', 'unknown')}_chunks.json"
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)

            logger.info(f"Created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            return []

    def process_multiple_pdfs(self, pdf_paths: List[Path]) -> Dict[str, Any]:
        logger.info(f"Processing {len(pdf_paths)} PDF files")

        results = {}
        success_count = 0

        for pdf_path in pdf_paths:
            try:
                result = self.process_pdf(pdf_path)
                if result:
                    results[pdf_path.name] = result
                    success_count += 1
                else:
                    logger.warning(f"Failed to process: {pdf_path.name}")

            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")

        logger.info(f"Successfully processed {success_count}/{len(pdf_paths)} PDFs")

        return {
            "processed_files": results,
            "summary": {
                "total_files": len(pdf_paths),
                "successful": success_count,
                "failed": len(pdf_paths) - success_count,
                "processing_time": datetime.now().isoformat()
            }
        }

    def get_processed_files(self) -> List[Path]:
        return list(self.processed_dir.glob("*.json"))

    def get_chunks_for_file(self, filename: str) -> Optional[List[Dict[str, Any]]]:
        chunks_file = self.chunks_dir / f"{filename}_chunks.json"

        if chunks_file.exists():
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading chunks for {filename}: {e}")

        return None

    def search_chunks_by_content(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        matching_chunks = []

        for chunks_file in self.chunks_dir.glob("*_chunks.json"):
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)

                for chunk in chunks:
                    if query_lower in chunk['text'].lower():
                        chunk['relevance_score'] = chunk['text'].lower().count(query_lower)
                        matching_chunks.append(chunk)

            except Exception as e:
                logger.error(f"Error searching in {chunks_file}: {e}")

        matching_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return matching_chunks[:max_results]