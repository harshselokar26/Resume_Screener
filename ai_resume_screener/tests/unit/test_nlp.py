"""
Unit Tests for NLP Processor Service

This module contains unit tests for the NLP processing functionality
including skill extraction, text analysis, and keyword identification.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from app.services.nlp_processor import NLPProcessor
from app.utils.exceptions import NLPProcessingError
from tests import SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION


class TestNLPProcessor:
    """Test class for NLP processor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create NLP processor instance for testing."""
        return NLPProcessor()
    
    @pytest.fixture
    def mock_nlp_model(self):
        """Create mock spaCy NLP model."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_nlp.return_value = mock_doc
        return mock_nlp, mock_doc
    
    @pytest.fixture
    def sample_skills_db(self):
        """Sample skills database for testing."""
        return {
            "technical_skills": [
                "Python", "JavaScript", "React", "Node.js", 
                "PostgreSQL", "MongoDB", "AWS", "Docker"
            ],
            "soft_skills": [
                "Leadership", "Communication", "Problem Solving", "Teamwork"
            ],
            "certifications": [
                "AWS Certified", "Scrum Master", "PMP"
            ]
        }
    
    # ===== INITIALIZATION TESTS =====
    
    def test_processor_initialization(self, processor):
        """Test NLP processor initialization."""
        assert processor is not None
        assert hasattr(processor, 'executor')
        assert hasattr(processor, 'skills_db')
    
    def test_processor_with_spacy_unavailable(self):
        """Test processor initialization when spaCy is unavailable."""
        with patch('app.services.nlp_processor.SPACY_AVAILABLE', False):
            processor = NLPProcessor()
            assert processor.nlp is None
    
    def test_skills_database_loading(self, processor, sample_skills_db):
        """Test skills database loading."""
        with patch.object(processor, '_load_skills_database'):
            processor.skills_db = sample_skills_db
            
            assert "technical_skills" in processor.skills_db
            assert "Python" in processor.skills_db["technical_skills"]
    
    # ===== TEXT CLEANING TESTS =====
    
    @pytest.mark.asyncio
    async def test_clean_text_basic(self, processor):
        """Test basic text cleaning functionality."""
        dirty_text = "  This   is    a   test   text!!!   "
        cleaned = await processor.clean_text(dirty_text)
        
        assert cleaned == "This is a test text!"
        assert not cleaned.startswith(" ")
        assert not cleaned.endswith(" ")
    
    @pytest.mark.asyncio
    async def test_clean_text_special_characters(self, processor):
        """Test cleaning text with special characters."""
        text_with_special = "Hello@#$%^&*()World!!!"
        cleaned = await processor.clean_text(text_with_special)
        
        assert "@#$%^&*()" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned
    
    @pytest.mark.asyncio
    async def test_clean_text_empty_input(self, processor):
        """Test cleaning empty or None text."""
        assert await processor.clean_text("") == ""
        assert await processor.clean_text(None) == ""
    
    # ===== SKILLS EXTRACTION TESTS =====
    
    @pytest.mark.asyncio
    async def test_extract_skills_success(self, processor, mock_nlp_model, sample_skills_db):
        """Test successful skills extraction."""
        mock_nlp, mock_doc = mock_nlp_model
        processor.nlp = mock_nlp
        processor.skills_db = sample_skills_db
        
        # Mock phrase matcher results
        mock_matches = [
            (1, 0, 1),  # Technical skill match
            (2, 2, 3),  # Soft skill match
        ]
        
        with patch.object(processor, 'phrase_matcher') as mock_matcher:
            mock_matcher.return_value = mock_matches
            mock_span1 = Mock()
            mock_span1.text = "python"
            mock_span2 = Mock()
            mock_span2.text = "leadership"
            
            mock_doc.__getitem__.side_effect = [mock_span1, mock_span2]
            
            # Mock vocab strings
            processor.nlp.vocab.strings = {1: "TECHNICAL_SKILL", 2: "SOFT_SKILL"}
            
            result = await processor.extract_skills("Python developer with leadership skills")
            
            assert "technical_skills" in result
            assert "soft_skills" in result
            assert "certifications" in result
    
    @pytest.mark.asyncio
    async def test_extract_skills_no_nlp_model(self, processor):
        """Test skills extraction when NLP model is not available."""
        processor.nlp = None
        
        result = await processor.extract_skills("Some text")
        
        assert result == {"technical_skills": [], "soft_skills": [], "certifications": []}
    
    @pytest.mark.asyncio
    async def test_extract_skills_empty_text(self, processor):
        """Test skills extraction with empty text."""
        result = await processor.extract_skills("")
        
        assert result == {"technical_skills": [], "soft_skills": [], "certifications": []}
    
    @pytest.mark.asyncio
    async def test_extract_skills_error_handling(self, processor):
        """Test skills extraction error handling."""
        processor.nlp = Mock()
        processor.nlp.side_effect = Exception("NLP error")
        
        with pytest.raises(NLPProcessingError, match="Failed to extract skills"):
            await processor.extract_skills("Some text")
    
    # ===== EXPERIENCE EXTRACTION TESTS =====
    
    @pytest.mark.asyncio
    async def test_extract_experience_success(self, processor, mock_nlp_model):
        """Test successful experience extraction."""
        mock_nlp, mock_doc = mock_nlp_model
        processor.nlp = mock_nlp
        
        # Mock matcher results for experience
        mock_matches = [(1, 0, 3)]  # Experience match
        
        with patch.object(processor, 'matcher') as mock_matcher:
            mock_matcher.return_value = mock_matches
            
            # Mock span with years
            mock_span = Mock()
            mock_token = Mock()
            mock_token.text = "5"
            mock_token.like_num = True
            mock_span.__iter__ = Mock(return_value=iter([mock_token]))
            mock_span.text = "5 years experience"
            mock_doc.__getitem__.return_value = mock_span
            
            # Mock vocab strings
            processor.nlp.vocab.strings = {1: "EXPERIENCE_0"}
            
            result = await processor.extract_experience("5 years of experience in software development")
            
            assert "total_years" in result
            assert "details" in result
    
    @pytest.mark.asyncio
    async def test_extract_experience_no_matches(self, processor, mock_nlp_model):
        """Test experience extraction with no matches found."""
        mock_nlp, mock_doc = mock_nlp_model
        processor.nlp = mock_nlp
        
        with patch.object(processor, 'matcher') as mock_matcher:
            mock_matcher.return_value = []  # No matches
            
            result = await processor.extract_experience("Software developer")
            
            assert result["total_years"] == 0
            assert result["details"] == []
    
    # ===== EDUCATION EXTRACTION TESTS =====
    
    @pytest.mark.asyncio
    async def test_extract_education_success(self, processor, mock_nlp_model):
        """Test successful education extraction."""
        mock_nlp, mock_doc = mock_nlp_model
        processor.nlp = mock_nlp
        
        # Mock sentences
        mock_sent1 = Mock()
        mock_sent1.text = "Bachelor's degree in Computer Science from MIT"
        mock_sent2 = Mock()
        mock_sent2.text = "Worked as a software engineer"
        
        mock_doc.sents = [mock_sent1, mock_sent2]
        
        result = await processor.extract_education("Bachelor's degree in Computer Science from MIT. Worked as a software engineer.")
        
        assert len(result) >= 1
        assert any("Bachelor's degree" in item["text"] for item in result)
    
    @pytest.mark.asyncio
    async def test_extract_education_no_education_found(self, processor, mock_nlp_model):
        """Test education extraction when no education is found."""
        mock_nlp, mock_doc = mock_nlp_model
        processor.nlp = mock_nlp
        
        # Mock sentences without education keywords
        mock_sent = Mock()
        mock_sent.text = "Worked as a software engineer for 5 years"
        mock_doc.sents = [mock_sent]
        
        result = await processor.extract_education("Worked as a software engineer for 5 years")
        
        assert result == []
    
    # ===== KEYWORD EXTRACTION TESTS =====
    
    @pytest.mark.asyncio
    async def test_extract_keywords_success(self, processor, mock_nlp_model):
        """Test successful keyword extraction."""
        mock_nlp, mock_doc = mock_nlp_model
        processor.nlp = mock_nlp
        
        # Mock tokens
        mock_tokens = []
        for word, pos in [("python", "NOUN"), ("developer", "NOUN"), ("the", "DET"), ("and", "CCONJ")]:
            token = Mock()
            token.text = word
            token.lemma_ = word
            token.pos_ = pos
            token.is_stop = pos in ["DET", "CCONJ"]
            token.is_punct = False
            token.is_space = False
            mock_tokens.append(token)
        
        mock_doc.__iter__ = Mock(return_value=iter(mock_tokens))
        
        result = await processor.extract_keywords("Python developer", top_n=10)
        
        assert isinstance(result, list)
        assert len(result) <= 10
    
    @pytest.mark.asyncio
    async def test_extract_keywords_empty_text(self, processor):
        """Test keyword extraction with empty text."""
        result = await processor.extract_keywords("")
        assert result == []
    
    # ===== BASIC INFO EXTRACTION TESTS =====
    
    @pytest.mark.asyncio
    async def test_extract_basic_info_success(self, processor, mock_nlp_model):
        """Test successful basic info extraction."""
        mock_nlp, mock_doc = mock_nlp_model
        processor.nlp = mock_nlp
        
        # Mock entities
        mock_person = Mock()
        mock_person.label_ = "PERSON"
        mock_person.text = "John Doe"
        
        mock_org = Mock()
        mock_org.label_ = "ORG"
        mock_org.text = "TechCorp"
        
        mock_doc.ents = [mock_person, mock_org]
        
        # Mock matcher for emails/phones
        with patch.object(processor, 'matcher') as mock_matcher:
            mock_matcher.return_value = []  # No email/phone matches for simplicity
            
            result = await processor.extract_basic_info("John Doe works at TechCorp")
            
            assert "names" in result
            assert "organizations" in result
            assert "emails" in result
            assert "phones" in result
    
    # ===== RELEVANT SKILLS TESTS =====
    
    @pytest.mark.asyncio
    async def test_find_relevant_skills(self, processor):
        """Test finding relevant skills between resume and job description."""
        with patch.object(processor, 'extract_skills') as mock_extract:
            # Mock skills extraction results
            resume_skills = {
                "technical_skills": ["Python", "JavaScript", "React"],
                "soft_skills": ["Leadership", "Communication"],
                "certifications": []
            }
            
            job_skills = {
                "technical_skills": ["Python", "Java", "React"],
                "soft_skills": ["Leadership", "Teamwork"],
                "certifications": []
            }
            
            mock_extract.side_effect = [resume_skills, job_skills]
            
            result = await processor.find_relevant_skills(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION)
            
            # Should find common skills: Python, React, Leadership
            assert "Python" in result
            assert "React" in result
            assert "Leadership" in result
            assert "Java" not in result  # Not in resume
            assert "Teamwork" not in result  # Not in resume
    
    # ===== JOB ANALYSIS TESTS =====
    
    @pytest.mark.asyncio
    async def test_extract_required_skills(self, processor):
        """Test extracting required skills from job description."""
        with patch.object(processor, 'extract_skills') as mock_extract:
            mock_extract.return_value = {
                "technical_skills": ["Python", "Django", "PostgreSQL"],
                "soft_skills": ["Communication", "Teamwork"],
                "certifications": ["AWS Certified"]
            }
            
            result = await processor.extract_required_skills(SAMPLE_JOB_DESCRIPTION)
            
            assert "Python" in result
            assert "Django" in result
            assert "Communication" in result
            assert "AWS Certified" in result
    
    @pytest.mark.asyncio
    async def test_extract_requirements(self, processor, mock_nlp_model):
        """Test extracting job requirements."""
        mock_nlp, mock_doc = mock_nlp_model
        processor.nlp = mock_nlp
        
        # Mock sentences
        mock_sent1 = Mock()
        mock_sent1.text = "5+ years of experience required"
        mock_sent2 = Mock()
        mock_sent2.text = "Must have Python skills"
        mock_sent3 = Mock()
        mock_sent3.text = "We offer competitive salary"
        
        mock_doc.sents = [mock_sent1, mock_sent2, mock_sent3]
        
        result = await processor.extract_requirements("5+ years of experience required. Must have Python skills. We offer competitive salary.")
        
        # Should find sentences with requirement indicators
        requirement_texts = [req for req in result]
        assert any("required" in req for req in requirement_texts)
        assert any("Must have" in req for req in requirement_texts)
        assert not any("competitive salary" in req for req in requirement_texts)
    
    @pytest.mark.asyncio
    async def test_determine_job_level(self, processor):
        """Test determining job seniority level."""
        test_cases = [
            ("Entry level position for new graduates", "entry"),
            ("Junior developer with 1-2 years experience", "entry"),
            ("Mid-level engineer with 3-5 years experience", "mid"),
            ("Senior software engineer with 7+ years", "senior"),
            ("Lead developer and team manager", "lead"),
            ("Director of Engineering position", "executive"),
            ("Software developer position", "unknown")  # No clear indicators
        ]
        
        for description, expected_level in test_cases:
            result = await processor.determine_job_level(description)
            assert result == expected_level
    
    # ===== HEALTH CHECK TESTS =====
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, processor):
        """Test health check when NLP processor is healthy."""
        processor.nlp = Mock()  # Mock NLP model exists
        
        result = await processor.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, processor):
        """Test health check when NLP processor is unhealthy."""
        processor.nlp = None  # No NLP model
        
        result = await processor.health_check()
        assert result is False
    
    # ===== PERFORMANCE TESTS =====
    
    @pytest.mark.asyncio
    async def test_skills_extraction_performance(self, processor, assert_timing):
        """Test that skills extraction completes within reasonable time."""
        with patch.object(processor, 'extract_skills') as mock_extract:
            mock_extract.return_value = {"technical_skills": [], "soft_skills": [], "certifications": []}
            
            def extract():
                import asyncio
                return asyncio.run(processor.extract_skills(SAMPLE_RESUME_TEXT))
            
            result = assert_timing(extract, max_time=2.0)  # Should complete within 2 seconds
            assert isinstance(result, dict)
    
    # ===== ERROR HANDLING TESTS =====
    
    @pytest.mark.asyncio
    async def test_extract_skills_exception_handling(self, processor):
        """Test skills extraction exception handling."""
        processor.nlp = Mock()
        processor.nlp.side_effect = Exception("Unexpected error")
        
        with pytest.raises(NLPProcessingError):
            await processor.extract_skills("Some text")
    
    @pytest.mark.asyncio
    async def test_extract_experience_exception_handling(self, processor):
        """Test experience extraction exception handling."""
        processor.nlp = Mock()
        processor.nlp.side_effect = Exception("Unexpected error")
        
        result = await processor.extract_experience("Some text")
        assert result == {"total_years": 0, "details": []}
    
    # ===== INTEGRATION TESTS =====
    
    @pytest.mark.asyncio
    async def test_full_nlp_workflow(self, processor):
        """Test complete NLP processing workflow."""
        with patch.object(processor, 'extract_skills') as mock_skills, \
             patch.object(processor, 'extract_experience') as mock_exp, \
             patch.object(processor, 'extract_education') as mock_edu, \
             patch.object(processor, 'extract_keywords') as mock_keywords:
            
            # Mock all extraction methods
            mock_skills.return_value = {"technical_skills": ["Python"], "soft_skills": [], "certifications": []}
            mock_exp.return_value = {"total_years": 5, "details": []}
            mock_edu.return_value = [{"text": "Bachelor's degree", "type": "education"}]
            mock_keywords.return_value = ["python", "developer", "software"]
            
            # Test complete workflow
            skills = await processor.extract_skills(SAMPLE_RESUME_TEXT)
            experience = await processor.extract_experience(SAMPLE_RESUME_TEXT)
            education = await processor.extract_education(SAMPLE_RESUME_TEXT)
            keywords = await processor.extract_keywords(SAMPLE_RESUME_TEXT)
            
            assert "technical_skills" in skills
            assert "total_years" in experience
            assert len(education) >= 0
            assert isinstance(keywords, list)
