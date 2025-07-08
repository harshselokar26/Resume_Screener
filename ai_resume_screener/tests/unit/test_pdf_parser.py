"""
Unit Tests for PDF Parser Service

This module contains unit tests for the PDF parser functionality
including text extraction, metadata extraction, and file validation.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from app.services.pdf_parser import PDFParser
from app.utils.exceptions import FileProcessingError


class TestPDFParser:
    """Test class for PDF parser functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create PDF parser instance for testing."""
        return PDFParser()
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing."""
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    
    @pytest.fixture
    def create_test_pdf(self, temp_dir, sample_pdf_content):
        """Create a test PDF file."""
        pdf_path = os.path.join(temp_dir, "test_resume.pdf")
        with open(pdf_path, "wb") as f:
            f.write(sample_pdf_content)
        return pdf_path
    
    # ===== INITIALIZATION TESTS =====
    
    def test_parser_initialization(self, parser):
        """Test PDF parser initialization."""
        assert parser is not None
        assert hasattr(parser, 'executor')
    
    def test_parser_logs_available_parsers(self, parser, caplog):
        """Test that parser logs available parsing libraries."""
        parser._log_available_parsers()
        assert "Available parsers:" in caplog.text
    
    # ===== PDF TEXT EXTRACTION TESTS =====
    
    @pytest.mark.asyncio
    async def test_extract_text_from_pdf_success(self, parser, create_test_pdf):
        """Test successful PDF text extraction."""
        with patch.object(parser, '_extract_with_pymupdf') as mock_extract:
            mock_extract.return_value = "Sample resume text content"
            
            result = await parser.extract_text_from_pdf(create_test_pdf)
            
            assert result == "Sample resume text content"
            mock_extract.assert_called_once_with(create_test_pdf)
    
    @pytest.mark.asyncio
    async def test_extract_text_file_not_found(self, parser):
        """Test PDF extraction with non-existent file."""
        with pytest.raises(FileProcessingError, match="File not found"):
            await parser.extract_text_from_pdf("nonexistent_file.pdf")
    
    @pytest.mark.asyncio
    async def test_extract_text_fallback_to_pdfminer(self, parser, create_test_pdf):
        """Test fallback to pdfminer when PyMuPDF fails."""
        with patch.object(parser, '_extract_with_pymupdf') as mock_pymupdf, \
             patch.object(parser, '_extract_with_pdfminer') as mock_pdfminer:
            
            mock_pymupdf.return_value = ""  # Empty result
            mock_pdfminer.return_value = "Extracted with pdfminer"
            
            result = await parser.extract_text_from_pdf(create_test_pdf)
            
            assert result == "Extracted with pdfminer"
            mock_pdfminer.assert_called_once_with(create_test_pdf)
    
    @pytest.mark.asyncio
    async def test_extract_text_no_parsers_available(self, parser, create_test_pdf):
        """Test error when no PDF parsers are available."""
        with patch('app.services.pdf_parser.PYMUPDF_AVAILABLE', False), \
             patch('app.services.pdf_parser.PDFMINER_AVAILABLE', False):
            
            with pytest.raises(FileProcessingError, match="No PDF parsing libraries available"):
                await parser.extract_text_from_pdf(create_test_pdf)
    
    # ===== DOC/DOCX EXTRACTION TESTS =====
    
    @pytest.mark.asyncio
    async def test_extract_text_from_docx(self, parser, temp_dir):
        """Test DOCX text extraction."""
        docx_path = os.path.join(temp_dir, "test_resume.docx")
        
        # Create mock DOCX file
        with open(docx_path, "wb") as f:
            f.write(b"PK\x03\x04")  # ZIP signature for DOCX
        
        with patch.object(parser, '_extract_from_docx') as mock_extract:
            mock_extract.return_value = "DOCX content"
            
            result = await parser.extract_text_from_doc(docx_path)
            
            assert result == "DOCX content"
            mock_extract.assert_called_once_with(docx_path)
    
    @pytest.mark.asyncio
    async def test_extract_text_from_doc(self, parser, temp_dir):
        """Test DOC text extraction."""
        doc_path = os.path.join(temp_dir, "test_resume.doc")
        
        # Create mock DOC file
        with open(doc_path, "wb") as f:
            f.write(b"\xd0\xcf\x11\xe0")  # OLE signature for DOC
        
        with patch.object(parser, '_extract_from_doc') as mock_extract:
            mock_extract.return_value = "DOC content"
            
            result = await parser.extract_text_from_doc(doc_path)
            
            assert result == "DOC content"
            mock_extract.assert_called_once_with(doc_path)
    
    @pytest.mark.asyncio
    async def test_extract_unsupported_format(self, parser, temp_dir):
        """Test extraction with unsupported file format."""
        txt_path = os.path.join(temp_dir, "test_file.txt")
        with open(txt_path, "w") as f:
            f.write("Plain text content")
        
        with pytest.raises(FileProcessingError, match="Unsupported file format"):
            await parser.extract_text_from_doc(txt_path)
    
    # ===== METADATA EXTRACTION TESTS =====
    
    @pytest.mark.asyncio
    async def test_extract_metadata_success(self, parser, create_test_pdf):
        """Test successful metadata extraction."""
        with patch('app.services.pdf_parser.PYMUPDF_AVAILABLE', True):
            mock_metadata = {
                "page_count": 2,
                "author": "John Doe",
                "title": "Resume",
                "creationDate": "2023-01-01",
                "modDate": "2023-01-02"
            }
            
            with patch('fitz.open') as mock_fitz:
                mock_doc = Mock()
                mock_doc.page_count = 2
                mock_doc.metadata = mock_metadata
                mock_fitz.return_value = mock_doc
                
                result = await parser.extract_metadata(create_test_pdf)
                
                assert result["page_count"] == 2
                assert result["author"] == "John Doe"
                assert result["title"] == "Resume"
    
    @pytest.mark.asyncio
    async def test_extract_metadata_file_stats_only(self, parser, create_test_pdf):
        """Test metadata extraction with only file stats."""
        with patch('app.services.pdf_parser.PYMUPDF_AVAILABLE', False):
            result = await parser.extract_metadata(create_test_pdf)
            
            assert "file_size" in result
            assert result["file_size"] > 0
            assert result["page_count"] == 0  # Default when PDF metadata unavailable
    
    # ===== FILE VALIDATION TESTS =====
    
    @pytest.mark.asyncio
    async def test_validate_pdf_file_success(self, parser, create_test_pdf):
        """Test successful PDF file validation."""
        with patch.object(parser, '_validate_pdf') as mock_validate:
            mock_validate.return_value = True
            
            result = await parser.validate_file(create_test_pdf)
            
            assert result is True
            mock_validate.assert_called_once_with(create_test_pdf)
    
    @pytest.mark.asyncio
    async def test_validate_docx_file_success(self, parser, temp_dir):
        """Test successful DOCX file validation."""
        docx_path = os.path.join(temp_dir, "test.docx")
        with open(docx_path, "wb") as f:
            f.write(b"PK\x03\x04")
        
        with patch.object(parser, '_validate_doc') as mock_validate:
            mock_validate.return_value = True
            
            result = await parser.validate_file(docx_path)
            
            assert result is True
            mock_validate.assert_called_once_with(docx_path)
    
    @pytest.mark.asyncio
    async def test_validate_nonexistent_file(self, parser):
        """Test validation of non-existent file."""
        result = await parser.validate_file("nonexistent_file.pdf")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_unsupported_extension(self, parser, temp_dir):
        """Test validation of unsupported file extension."""
        txt_path = os.path.join(temp_dir, "test.txt")
        with open(txt_path, "w") as f:
            f.write("Plain text")
        
        result = await parser.validate_file(txt_path)
        assert result is False
    
    # ===== HEALTH CHECK TESTS =====
    
    @pytest.mark.asyncio
    async def test_health_check_with_parsers(self, parser):
        """Test health check when parsers are available."""
        with patch('app.services.pdf_parser.PYMUPDF_AVAILABLE', True):
            result = await parser.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_no_parsers(self, parser):
        """Test health check when no parsers are available."""
        with patch('app.services.pdf_parser.PYMUPDF_AVAILABLE', False), \
             patch('app.services.pdf_parser.PDFMINER_AVAILABLE', False):
            
            result = await parser.health_check()
            assert result is False
    
    # ===== SUPPORTED FORMATS TESTS =====
    
    def test_get_supported_formats_all_available(self, parser):
        """Test getting supported formats when all libraries are available."""
        with patch('app.services.pdf_parser.PYMUPDF_AVAILABLE', True), \
             patch('app.services.pdf_parser.PYTHON_DOCX_AVAILABLE', True), \
             patch('app.services.pdf_parser.WIN32COM_AVAILABLE', True):
            
            formats = parser.get_supported_formats()
            
            assert '.pdf' in formats
            assert '.docx' in formats
            assert '.doc' in formats
    
    def test_get_supported_formats_limited(self, parser):
        """Test getting supported formats with limited libraries."""
        with patch('app.services.pdf_parser.PYMUPDF_AVAILABLE', True), \
             patch('app.services.pdf_parser.PYTHON_DOCX_AVAILABLE', False), \
             patch('app.services.pdf_parser.WIN32COM_AVAILABLE', False):
            
            formats = parser.get_supported_formats()
            
            assert '.pdf' in formats
            assert '.docx' not in formats
            assert '.doc' not in formats
    
    # ===== PRIVATE METHOD TESTS =====
    
    @pytest.mark.asyncio
    async def test_extract_with_pymupdf(self, parser, create_test_pdf):
        """Test PyMuPDF extraction method."""
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_page = Mock()
            mock_page.get_text.return_value = "Extracted text"
            mock_doc.page_count = 1
            mock_doc.__getitem__.return_value = mock_page
            mock_fitz.return_value = mock_doc
            
            result = await parser._extract_with_pymupdf(create_test_pdf)
            
            assert result == "Extracted text"
            mock_fitz.assert_called_once_with(create_test_pdf)
    
    @pytest.mark.asyncio
    async def test_extract_with_pdfminer(self, parser, create_test_pdf):
        """Test pdfminer extraction method."""
        with patch('app.services.pdf_parser.pdfminer_extract') as mock_extract:
            mock_extract.return_value = "Extracted with pdfminer"
            
            result = await parser._extract_with_pdfminer(create_test_pdf)
            
            assert result == "Extracted with pdfminer"
            mock_extract.assert_called_once()
    
    # ===== ERROR HANDLING TESTS =====
    
    @pytest.mark.asyncio
    async def test_extraction_with_corrupted_file(self, parser, temp_dir):
        """Test extraction with corrupted PDF file."""
        corrupted_path = os.path.join(temp_dir, "corrupted.pdf")
        with open(corrupted_path, "wb") as f:
            f.write(b"Not a valid PDF content")
        
        with patch.object(parser, '_extract_with_pymupdf') as mock_extract:
            mock_extract.side_effect = Exception("Corrupted file")
            
            with pytest.raises(FileProcessingError, match="Failed to extract text from PDF"):
                await parser.extract_text_from_pdf(corrupted_path)
    
    @pytest.mark.asyncio
    async def test_metadata_extraction_error_handling(self, parser, create_test_pdf):
        """Test metadata extraction error handling."""
        with patch('fitz.open') as mock_fitz:
            mock_fitz.side_effect = Exception("File error")
            
            result = await parser.extract_metadata(create_test_pdf)
            
            # Should return basic file stats even if PDF metadata fails
            assert "file_size" in result
            assert result["page_count"] == 0
    
    # ===== PERFORMANCE TESTS =====
    
    @pytest.mark.asyncio
    async def test_extraction_performance(self, parser, create_test_pdf, assert_timing):
        """Test that extraction completes within reasonable time."""
        with patch.object(parser, '_extract_with_pymupdf') as mock_extract:
            mock_extract.return_value = "Fast extraction"
            
            def extract():
                import asyncio
                return asyncio.run(parser.extract_text_from_pdf(create_test_pdf))
            
            result = assert_timing(extract, max_time=1.0)  # Should complete within 1 second
            assert result == "Fast extraction"
    
    # ===== INTEGRATION TESTS =====
    
    @pytest.mark.asyncio
    async def test_full_pdf_processing_workflow(self, parser, create_test_pdf):
        """Test complete PDF processing workflow."""
        with patch.object(parser, '_extract_with_pymupdf') as mock_extract, \
             patch.object(parser, '_validate_pdf') as mock_validate:
            
            mock_validate.return_value = True
            mock_extract.return_value = "Complete resume text"
            
            # Validate file
            is_valid = await parser.validate_file(create_test_pdf)
            assert is_valid is True
            
            # Extract text
            text = await parser.extract_text_from_pdf(create_test_pdf)
            assert text == "Complete resume text"
            
            # Extract metadata
            metadata = await parser.extract_metadata(create_test_pdf)
            assert "file_size" in metadata
