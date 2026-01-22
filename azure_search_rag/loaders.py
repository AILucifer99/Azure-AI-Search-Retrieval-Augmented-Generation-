"""
Document loaders for Azure Search RAG system.

This module provides PDF loading functionality with advanced
text and table extraction capabilities.
"""

from typing import List, Dict, Any
import fitz  # PyMuPDF
from langchain_core.documents import Document


class CustomPyMuPDFLoader:
    """Advanced PDF loader using PyMuPDF with table extraction support."""
    
    def __init__(self, file_path: str):
        """
        Initialize the PDF loader.
        
        Args:
            file_path: Path to the PDF file to load.
        """
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """
        Load and parse the PDF document.
        
        Returns:
            List of Document objects, one per page.
            
        Raises:
            Exception: If PDF loading fails.
        """
        documents = []
        
        try:
            pdf_document = fitz.open(self.file_path)
            print(f"📄 Loading PDF: {self.file_path}")
            print(f"   Pages: {len(pdf_document)}")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Extract text
                text = page.get_text("text")
                
                # Extract tables
                tables_text = self._extract_tables(page)
                
                # Combine text and tables
                full_content = text
                if tables_text:
                    full_content += "\n\n" + tables_text
                
                # Create metadata
                metadata = self._create_metadata(page, page_num, pdf_document)
                
                if full_content.strip():
                    doc = Document(
                        page_content=full_content,
                        metadata=metadata
                    )
                    documents.append(doc)
            
            pdf_document.close()
            print(f"   ✓ Extracted {len(documents)} pages")
            
        except Exception as e:
            print(f"   ❌ Error loading PDF: {e}")
            raise
        
        return documents
    
    def _extract_tables(self, page) -> str:
        """
        Extract tables from a PDF page.
        
        Args:
            page: PyMuPDF page object.
            
        Returns:
            Formatted string containing table data.
        """
        tables_text = ""
        try:
            tables = page.find_tables()
            if tables and len(tables.tables) > 0:
                for table_num, table in enumerate(tables.tables):
                    tables_text += f"\n\n=== Table {table_num + 1} ===\n"
                    table_data = table.extract()
                    for row in table_data:
                        cells = [str(cell).strip() if cell else "" for cell in row]
                        tables_text += " | ".join(cells) + "\n"
        except Exception:
            # Silently ignore table extraction errors
            pass
        return tables_text
    
    def _create_metadata(self, page, page_num: int, pdf_document) -> Dict[str, Any]:
        """
        Create metadata for a PDF page.
        
        Args:
            page: PyMuPDF page object.
            page_num: Zero-indexed page number.
            pdf_document: PyMuPDF document object.
            
        Returns:
            Dictionary containing page metadata.
        """
        rect = page.rect
        images = page.get_images()
        doc_metadata = pdf_document.metadata
        
        metadata = {
            "source": self.file_path,
            "file_path": self.file_path,
            "page": page_num + 1,  # 1-indexed for display
            "page_number": page_num,  # 0-indexed for processing
            "total_pages": len(pdf_document),
            "page_width": round(rect.width, 2),
            "page_height": round(rect.height, 2),
            "image_count": len(images),
            "title": doc_metadata.get("title", ""),
            "author": doc_metadata.get("author", ""),
            "subject": doc_metadata.get("subject", ""),
            "keywords": doc_metadata.get("keywords", ""),
            "creator": doc_metadata.get("creator", ""),
            "producer": doc_metadata.get("producer", ""),
            "format": doc_metadata.get("format", ""),
        }
        
        # Remove empty values but keep zeros and False
        metadata = {k: v for k, v in metadata.items() if v or isinstance(v, (int, float))}
        
        return metadata
