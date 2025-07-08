"""
Unit Tests for Resume Scorer Service

This module contains unit tests for the resume scoring functionality
including similarity calculation, skill matching, and recommendation generation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from app.services.scorer import ResumeScorer
from app.utils.exceptions import ScoringError
from tests import SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION


class TestResumeScorer:
    """Test class for resume scorer functionality."""
    
    @pytest.fixture
    def scorer(self):
        """Create resume scorer instance for testing."""
        return ResumeScorer()
    
    @pytest.fixture
    def mock_vectorizer(self):
        """Create mock TF-IDF vectorizer."""
        mock_vectorizer = Mock()
        mock_matrix = Mock()
        mock_matrix.shape = (2, 100)
        mock_vectorizer.fit_transform.return_value = mock_matrix
        return mock_vectorizer, mock_matrix
    
    @pytest.fixture
    def sample_skills_data(self):
        """Sample skills data for testing."""
        return {
            "resume_skills": {
                "technical_skills": ["Python", "JavaScript", "React", "PostgreSQL"],
                "soft_skills": ["Leadership", "Communication"],
                "certifications": ["AWS Certified"]
            },
            "job_skills": {
                "technical_skills": ["Python", "Java", "React", "MongoDB"],
                "soft_skills": ["Leadership", "Teamwork"],
                "certifications": ["AWS Certified", "Scrum Master"]
            }
        }
    
    # ===== INITIALIZATION TESTS =====
    
    def test_scorer_initialization(self, scorer):
        """Test resume scorer initialization."""
        assert scorer is not None
        assert hasattr(scorer, 'executor')
        assert hasattr(scorer, 'nlp_processor')
    
    def test_scorer_with_sklearn_unavailable(self):
        """Test scorer initialization when scikit-learn is unavailable."""
        with patch('app.services.scorer.SKLEARN_AVAILABLE', False):
            scorer = ResumeScorer()
            assert scorer.tfidf_vectorizer is None
    
    # ===== SIMILARITY CALCULATION TESTS =====
    
    @pytest.mark.asyncio
    async def test_calculate_similarity_success(self, scorer, mock_vectorizer):
        """Test successful similarity calculation."""
        mock_vectorizer_obj, mock_matrix = mock_vectorizer
        scorer.tfidf_vectorizer = mock_vectorizer_obj
        
        with patch('app.services.scorer.cosine_similarity') as mock_cosine:
            # Mock cosine similarity result
            mock_cosine.return_value = [[1.0, 0.85], [0.85, 1.0]]
            
            result = await scorer.calculate_similarity(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION)
            
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0
            assert result == 0.85
    
    @pytest.mark.asyncio
    async def test_calculate_similarity_empty_inputs(self, scorer):
        """Test similarity calculation with empty inputs."""
        with pytest.raises(ScoringError, match="Both resume text and job description are required"):
            await scorer.calculate_similarity("", SAMPLE_JOB_DESCRIPTION)
        
        with pytest.raises(ScoringError, match="Both resume text and job description are required"):
            await scorer.calculate_similarity(SAMPLE_RESUME_TEXT, "")
    
    @pytest.mark.asyncio
    async def test_calculate_similarity_no_vectorizer(self, scorer):
        """Test similarity calculation when vectorizer is not available."""
        scorer.tfidf_vectorizer = None
        
        with pytest.raises(ScoringError, match="TF-IDF vectorizer not available"):
            await scorer.calculate_similarity(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION)
    
    @pytest.mark.asyncio
    async def test_calculate_similarity_score_bounds(self, scorer, mock_vectorizer):
        """Test that similarity scores are properly bounded between 0 and 1."""
        mock_vectorizer_obj, mock_matrix = mock_vectorizer
        scorer.tfidf_vectorizer = mock_vectorizer_obj
        
        test_cases = [
            ([[1.0, 1.5], [1.5, 1.0]], 1.0),  # Score > 1 should be capped at 1
            ([[1.0, -0.1], [-0.1, 1.0]], 0.0),  # Score < 0 should be capped at 0
            ([[1.0, 0.5], [0.5, 1.0]], 0.5),   # Normal score should remain unchanged
        ]
        
        for cosine_result, expected in test_cases:
            with patch('app.services.scorer.cosine_similarity') as mock_cosine:
                mock_cosine.return_value = cosine_result
                
                result = await scorer.calculate_similarity(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION)
                assert result == expected
    
    # ===== SKILLS SIMILARITY TESTS =====
    
    @pytest.mark.asyncio
    async def test_calculate_skills_similarity_success(self, scorer):
        """Test successful skills similarity calculation."""
        resume_skills = ["Python", "JavaScript", "React", "PostgreSQL"]
        job_skills = ["Python", "Java", "React", "MongoDB"]
        
        result = await scorer.calculate_skills_similarity(resume_skills, job_skills)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Should find 2 matching skills (Python, React) out of 6 total unique skills
        assert result > 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_skills_similarity_empty_lists(self, scorer):
        """Test skills similarity with empty skill lists."""
        result1 = await scorer.calculate_skills_similarity([], ["Python", "Java"])
        assert result1 == 0.0
        
        result2 = await scorer.calculate_skills_similarity(["Python", "Java"], [])
        assert result2 == 0.0
        
        result3 = await scorer.calculate_skills_similarity([], [])
        assert result3 == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_skills_similarity_identical_skills(self, scorer):
        """Test skills similarity with identical skill sets."""
        skills = ["Python", "JavaScript", "React"]
        
        result = await scorer.calculate_skills_similarity(skills, skills)
        
        # Identical skills should give high similarity
        assert result > 0.8
    
    @pytest.mark.asyncio
    async def test_calculate_skills_similarity_no_overlap(self, scorer):
        """Test skills similarity with no overlapping skills."""
        resume_skills = ["Python", "JavaScript"]
        job_skills = ["Java", "C++"]
        
        result = await scorer.calculate_skills_similarity(resume_skills, job_skills)
        
        # No overlap should give low similarity
        assert result == 0.0
    
    # ===== MATCHING SKILLS TESTS =====
    
    @pytest.mark.asyncio
    async def test_find_matching_skills_success(self, scorer, sample_skills_data):
        """Test successful matching skills identification."""
        with patch.object(scorer.nlp_processor, 'extract_skills') as mock_extract:
            mock_extract.side_effect = [
                sample_skills_data["resume_skills"],
                sample_skills_data["job_skills"]
            ]
            
            result = await scorer.find_matching_skills(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION)
            
            assert isinstance(result, list)
            # Should find matching skills: Python, React, Leadership, AWS Certified
            assert "Python" in result
            assert "React" in result
            assert "Leadership" in result
            assert "AWS Certified" in result
            # Should not find non-matching skills
            assert "Java" not in result
            assert "MongoDB" not in result
    
    @pytest.mark.asyncio
    async def test_find_matching_skills_no_matches(self, scorer):
        """Test matching skills when no skills match."""
        with patch.object(scorer.nlp_processor, 'extract_skills') as mock_extract:
            mock_extract.side_effect = [
                {"technical_skills": ["Python"], "soft_skills": [], "certifications": []},
                {"technical_skills": ["Java"], "soft_skills": [], "certifications": []}
            ]
            
            result = await scorer.find_matching_skills(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION)
            
            assert result == []
    
    # ===== MISSING SKILLS TESTS =====
    
    @pytest.mark.asyncio
    async def test_find_missing_skills_success(self, scorer, sample_skills_data):
        """Test successful missing skills identification."""
        with patch.object(scorer.nlp_processor, 'extract_skills') as mock_extract:
            mock_extract.side_effect = [
                sample_skills_data["resume_skills"],
                sample_skills_data["job_skills"]
            ]
            
            result = await scorer.find_missing_skills(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION)
            
            assert isinstance(result, list)
            # Should find missing skills: Java, MongoDB, Teamwork, Scrum Master
            assert "Java" in result
            assert "MongoDB" in result
            assert "Teamwork" in result
            assert "Scrum Master" in result
            # Should not find skills that are present in resume
            assert "Python" not in result
            assert "React" not in result
    
    @pytest.mark.asyncio
    async def test_find_missing_skills_no_missing(self, scorer):
        """Test missing skills when all required skills are present."""
        with patch.object(scorer.nlp_processor, 'extract_skills') as mock_extract:
            # Same skills in both resume and job
            same_skills = {"technical_skills": ["Python", "React"], "soft_skills": [], "certifications": []}
            mock_extract.side_effect = [same_skills, same_skills]
            
            result = await scorer.find_missing_skills(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION)
            
            assert result == []
    
    @pytest.mark.asyncio
    async def test_find_missing_skills_error_handling(self, scorer):
        """Test missing skills error handling."""
        with patch.object(scorer.nlp_processor, 'extract_skills') as mock_extract:
            mock_extract.side_effect = Exception("NLP error")
            
            result = await scorer.find_missing_skills(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION)
            
            assert result == []
    
    # ===== DETAILED ANALYSIS TESTS =====
    
    @pytest.mark.asyncio
    async def test_get_detailed_analysis_success(self, scorer, sample_skills_data):
        """Test successful detailed analysis generation."""
        with patch.object(scorer.nlp_processor, 'extract_skills') as mock_skills, \
             patch.object(scorer.nlp_processor, 'extract_experience') as mock_exp, \
             patch.object(scorer.nlp_processor, 'determine_job_level') as mock_level, \
             patch.object(scorer, 'calculate_skills_similarity') as mock_sim:
            
            # Mock all the dependencies
            mock_skills.side_effect = [
                sample_skills_data["resume_skills"],
                sample_skills_data["job_skills"]
            ]
            mock_exp.return_value = {"total_years": 5, "details": []}
            mock_level.return_value = "senior"
            mock_sim.return_value = 0.75
            
            result = await scorer.get_detailed_analysis(
                SAMPLE_RESUME_TEXT, 
                SAMPLE_JOB_DESCRIPTION, 
                0.85
            )
            
            assert "overall_similarity" in result
            assert "skills_similarity" in result
            assert "technical_skills_match_ratio" in result
            assert "experience_years" in result
            assert "job_level" in result
            assert "strengths" in result
            assert "weaknesses" in result
            assert "overall_assessment" in result
            
            assert result["overall_similarity"] == 0.85
            assert result["experience_years"] == 5
            assert result["job_level"] == "senior"
    
    @pytest.mark.asyncio
    async def test_get_detailed_analysis_error_handling(self, scorer):
        """Test detailed analysis error handling."""
        with patch.object(scorer.nlp_processor, 'extract_skills') as mock_extract:
            mock_extract.side_effect = Exception("Analysis error")
            
            result = await scorer.get_detailed_analysis(
                SAMPLE_RESUME_TEXT, 
                SAMPLE_JOB_DESCRIPTION, 
                0.85
            )
            
            assert "overall_similarity" in result
            assert "error" in result
            assert result["overall_similarity"] == 0.85
    
    # ===== RECOMMENDATION TESTS =====
    
    @pytest.mark.asyncio
    async def test_get_recommendation_highly_recommended(self, scorer):
        """Test recommendation for highly recommended candidate."""
        matching_skills = ["Python", "React", "JavaScript", "PostgreSQL", "Leadership"]
        missing_skills = ["Docker"]
        
        result = await scorer.get_recommendation(0.85, matching_skills, missing_skills)
        
        assert "decision" in result
        assert "confidence" in result
        assert "reasons" in result
        assert "suggestions" in result
        
        assert result["decision"] == "Highly Recommended"
        assert result["confidence"] == "High"
        assert len(result["reasons"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_recommendation_not_recommended(self, scorer):
        """Test recommendation for not recommended candidate."""
        matching_skills = ["Python"]
        missing_skills = ["React", "JavaScript", "PostgreSQL", "Docker", "AWS"]
        
        result = await scorer.get_recommendation(0.25, matching_skills, missing_skills)
        
        assert result["decision"] == "Not Recommended"
        assert result["confidence"] == "Low"
        assert "Poor match" in result["reasons"]
    
    @pytest.mark.asyncio
    async def test_get_recommendation_consider(self, scorer):
        """Test recommendation for consider candidate."""
        matching_skills = ["Python", "React"]
        missing_skills = ["JavaScript", "PostgreSQL"]
        
        result = await scorer.get_recommendation(0.55, matching_skills, missing_skills)
        
        assert result["decision"] == "Consider"
        assert result["confidence"] == "Medium"
    
    @pytest.mark.asyncio
    async def test_get_recommendation_error_handling(self, scorer):
        """Test recommendation error handling."""
        # Pass invalid data to trigger error
        with patch('builtins.len', side_effect=Exception("Length error")):
            result = await scorer.get_recommendation(0.85, ["Python"], ["Java"])
            
            assert "decision" in result
            assert "error" in result
            assert result["decision"] == "Unable to determine"
    
    # ===== BATCH SCORING TESTS =====
    
    @pytest.mark.asyncio
    async def test_batch_score_resumes_success(self, scorer):
        """Test successful batch resume scoring."""
        resume_texts = [
            "Python developer with 5 years experience",
            "Java developer with React skills",
            "Data scientist with machine learning expertise"
        ]
        
        with patch.object(scorer, 'calculate_similarity') as mock_sim, \
             patch.object(scorer, 'find_matching_skills') as mock_match, \
             patch.object(scorer, 'find_missing_skills') as mock_missing:
            
            mock_sim.side_effect = [0.85, 0.65, 0.45]
            mock_match.side_effect = [["Python"], ["Java", "React"], ["Machine Learning"]]
            mock_missing.side_effect = [["Docker"], ["Python"], ["Python", "SQL"]]
            
            result = await scorer.batch_score_resumes(resume_texts, SAMPLE_JOB_DESCRIPTION)
            
            assert len(result) == 3
            assert all("similarity_score" in item for item in result)
            assert all("matching_skills" in item for item in result)
            assert all("missing_skills" in item for item in result)
            
            # Results should be sorted by similarity score (descending)
            assert result[0]["similarity_score"] >= result[1]["similarity_score"]
            assert result[1]["similarity_score"] >= result[2]["similarity_score"]
    
    @pytest.mark.asyncio
    async def test_batch_score_resumes_with_errors(self, scorer):
        """Test batch scoring with some errors."""
        resume_texts = ["Valid resume", "Another resume"]
        
        with patch.object(scorer, 'calculate_similarity') as mock_sim:
            mock_sim.side_effect = [0.85, Exception("Scoring error")]
            
            result = await scorer.batch_score_resumes(resume_texts, SAMPLE_JOB_DESCRIPTION)
            
            assert len(result) == 2
            assert "similarity_score" in result[0]
            assert "error" in result[1]
    
    # ===== HEALTH CHECK TESTS =====
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, scorer):
        """Test health check when all components are healthy."""
        with patch('app.services.scorer.SKLEARN_AVAILABLE', True), \
             patch.object(scorer.nlp_processor, 'health_check') as mock_nlp_health:
            
            scorer.tfidf_vectorizer = Mock()  # Mock vectorizer exists
            mock_nlp_health.return_value = True
            
            result = await scorer.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_sklearn_unavailable(self, scorer):
        """Test health check when sklearn is unavailable."""
        with patch('app.services.scorer.SKLEARN_AVAILABLE', False):
            result = await scorer.health_check()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_nlp_unhealthy(self, scorer):
        """Test health check when NLP processor is unhealthy."""
        with patch('app.services.scorer.SKLEARN_AVAILABLE', True), \
             patch.object(scorer.nlp_processor, 'health_check') as mock_nlp_health:
            
            scorer.tfidf_vectorizer = Mock()
            mock_nlp_health.return_value = False
            
            result = await scorer.health_check()
            assert result is False
    
    # ===== PERFORMANCE TESTS =====
    
    @pytest.mark.asyncio
    async def test_similarity_calculation_performance(self, scorer, mock_vectorizer, assert_timing):
        """Test that similarity calculation completes within reasonable time."""
        mock_vectorizer_obj, mock_matrix = mock_vectorizer
        scorer.tfidf_vectorizer = mock_vectorizer_obj
        
        with patch('app.services.scorer.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[1.0, 0.85], [0.85, 1.0]]
            
            def calculate():
                import asyncio
                return asyncio.run(scorer.calculate_similarity(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION))
            
            result = assert_timing(calculate, max_time=2.0)  # Should complete within 2 seconds
            assert isinstance(result, float)
    
    # ===== ERROR HANDLING TESTS =====
    
    @pytest.mark.asyncio
    async def test_calculate_similarity_exception_handling(self, scorer, mock_vectorizer):
        """Test similarity calculation exception handling."""
        mock_vectorizer_obj, mock_matrix = mock_vectorizer
        scorer.tfidf_vectorizer = mock_vectorizer_obj
        
        with patch('app.services.scorer.cosine_similarity') as mock_cosine:
            mock_cosine.side_effect = Exception("Calculation error")
            
            with pytest.raises(ScoringError, match="Failed to calculate similarity"):
                await scorer.calculate_similarity(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION)
    
    @pytest.mark.asyncio
    async def test_skills_similarity_exception_handling(self, scorer):
        """Test skills similarity exception handling."""
        # Pass invalid data that would cause an exception
        with patch('builtins.set', side_effect=Exception("Set error")):
            result = await scorer.calculate_skills_similarity(["Python"], ["Java"])
            assert result == 0.0
    
    # ===== INTEGRATION TESTS =====
    
    @pytest.mark.asyncio
    async def test_full_scoring_workflow(self, scorer, sample_skills_data):
        """Test complete scoring workflow."""
        with patch.object(scorer, 'calculate_similarity') as mock_sim, \
             patch.object(scorer, 'find_matching_skills') as mock_match, \
             patch.object(scorer, 'find_missing_skills') as mock_missing, \
             patch.object(scorer, 'get_detailed_analysis') as mock_analysis, \
             patch.object(scorer, 'get_recommendation') as mock_rec:
            
            # Mock all methods
            mock_sim.return_value = 0.85
            mock_match.return_value = ["Python", "React"]
            mock_missing.return_value = ["Docker", "Kubernetes"]
            mock_analysis.return_value = {"overall_similarity": 0.85, "strengths": ["Good skills"]}
            mock_rec.return_value = {"decision": "Recommended", "confidence": "High"}
            
            # Test complete workflow
            similarity = await scorer.calculate_similarity(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION)
            matching = await scorer.find_matching_skills(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION)
            missing = await scorer.find_missing_skills(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION)
            analysis = await scorer.get_detailed_analysis(SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION, similarity)
            recommendation = await scorer.get_recommendation(similarity, matching, missing)
            
            assert similarity == 0.85
            assert "Python" in matching
            assert "Docker" in missing
            assert "overall_similarity" in analysis
            assert recommendation["decision"] == "Recommended"
    
    # ===== EDGE CASE TESTS =====
    
    @pytest.mark.asyncio
    async def test_similarity_with_identical_texts(self, scorer, mock_vectorizer):
        """Test similarity calculation with identical texts."""
        mock_vectorizer_obj, mock_matrix = mock_vectorizer
        scorer.tfidf_vectorizer = mock_vectorizer_obj
        
        with patch('app.services.scorer.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[1.0, 1.0], [1.0, 1.0]]
            
            result = await scorer.calculate_similarity(SAMPLE_RESUME_TEXT, SAMPLE_RESUME_TEXT)
            
            assert result == 1.0
    
    @pytest.mark.asyncio
    async def test_skills_similarity_case_insensitive(self, scorer):
        """Test that skills similarity is case insensitive."""
        resume_skills = ["python", "javascript"]
        job_skills = ["Python", "JavaScript"]
        
        result = await scorer.calculate_skills_similarity(resume_skills, job_skills)
        
        # Should find matches despite case differences
        assert result > 0.8
    
    @pytest.mark.asyncio
    async def test_recommendation_with_many_skills(self, scorer):
        """Test recommendation with large number of skills."""
        many_matching_skills = [f"Skill_{i}" for i in range(20)]
        few_missing_skills = ["Missing_1", "Missing_2"]
        
        result = await scorer.get_recommendation(0.75, many_matching_skills, few_missing_skills)
        
        assert result["decision"] in ["Recommended", "Highly Recommended"]
        assert "Strong skills alignment" in result["reasons"]
