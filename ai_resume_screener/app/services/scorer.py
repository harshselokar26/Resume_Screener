"""
Resume Scorer Service for AI Resume Screener

This module handles similarity calculation between resumes and job descriptions
using various NLP techniques including TF-IDF and cosine similarity.
"""

import logging
from typing import List, Dict, Any, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# ML libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from app.config.settings import settings
from app.services.nlp_processor import NLPProcessor
from app.utils.exceptions import ScoringError

# Setup logging
logger = logging.getLogger(__name__)


class ResumeScorer:
    """
    Resume scoring service using NLP and ML techniques.
    """
    
    def __init__(self):
        """Initialize resume scorer with NLP processor and vectorizers."""
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.nlp_processor = NLPProcessor()
        self.tfidf_vectorizer = None
        self.skills_vectorizer = None
        self._initialize_vectorizers()
    
    def _initialize_vectorizers(self):
        """Initialize TF-IDF vectorizers."""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available. Please install: pip install scikit-learn")
            return
        
        try:
            # Main TF-IDF vectorizer for general text
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                lowercase=True,
                strip_accents='unicode'
            )
            
            # Skills-specific vectorizer
            self.skills_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                lowercase=True,
                strip_accents='unicode'
            )
            
            logger.info("TF-IDF vectorizers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vectorizers: {str(e)}")
    
    async def calculate_similarity(self, resume_text: str, job_description: str) -> float:
        """
        Calculate similarity score between resume and job description.
        
        Args:
            resume_text: Resume text content
            job_description: Job description text
            
        Returns:
            float: Similarity score between 0 and 1
            
        Raises:
            ScoringError: If similarity calculation fails
        """
        if not resume_text or not job_description:
            raise ScoringError("Both resume text and job description are required")
        
        if not self.tfidf_vectorizer:
            raise ScoringError("TF-IDF vectorizer not available")
        
        try:
            def _calculate():
                # Clean texts
                texts = [resume_text.lower(), job_description.lower()]
                
                # Vectorize texts
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                
                # Calculate cosine similarity
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Return similarity between resume and job description
                return float(similarity_matrix[0][1])
            
            loop = asyncio.get_event_loop()
            similarity = await loop.run_in_executor(self.executor, _calculate)
            
            # Ensure similarity is within valid range
            similarity = max(0.0, min(1.0, similarity))
            
            logger.info(f"Calculated similarity score: {similarity:.3f}")
            return similarity
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            raise ScoringError(f"Failed to calculate similarity: {str(e)}")
    
    async def calculate_skills_similarity(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """
        Calculate similarity based on skills overlap.
        
        Args:
            resume_skills: List of skills from resume
            job_skills: List of skills from job description
            
        Returns:
            float: Skills similarity score between 0 and 1
        """
        if not resume_skills or not job_skills:
            return 0.0
        
        try:
            # Convert to sets for intersection calculation
            resume_set = set(skill.lower().strip() for skill in resume_skills)
            job_set = set(skill.lower().strip() for skill in job_skills)
            
            # Calculate Jaccard similarity
            intersection = len(resume_set.intersection(job_set))
            union = len(resume_set.union(job_set))
            
            if union == 0:
                return 0.0
            
            jaccard_similarity = intersection / union
            
            # Also calculate overlap percentage
            overlap_percentage = intersection / len(job_set) if job_set else 0.0
            
            # Combine both metrics (weighted average)
            combined_score = (jaccard_similarity * 0.4) + (overlap_percentage * 0.6)
            
            return min(1.0, combined_score)
            
        except Exception as e:
            logger.error(f"Skills similarity calculation failed: {str(e)}")
            return 0.0
    
    async def find_matching_skills(self, resume_text: str, job_description: str) -> List[str]:
        """
        Find skills that match between resume and job description.
        
        Args:
            resume_text: Resume text content
            job_description: Job description text
            
        Returns:
            List of matching skills
        """
        try:
            # Extract skills from both texts
            resume_skills = await self.nlp_processor.extract_skills(resume_text)
            job_skills = await self.nlp_processor.extract_skills(job_description)
            
            matching_skills = []
            
            # Find matches across all skill categories
            for skill_type in ["technical_skills", "soft_skills", "certifications"]:
                resume_set = set(skill.lower() for skill in resume_skills.get(skill_type, []))
                job_set = set(skill.lower() for skill in job_skills.get(skill_type, []))
                
                matches = resume_set.intersection(job_set)
                matching_skills.extend([skill.title() for skill in matches])
            
            return list(set(matching_skills))
            
        except Exception as e:
            logger.error(f"Matching skills extraction failed: {str(e)}")
            return []
    
    async def find_missing_skills(self, resume_text: str, job_description: str) -> List[str]:
        """
        Find skills required by job but missing from resume.
        
        Args:
            resume_text: Resume text content
            job_description: Job description text
            
        Returns:
            List of missing skills
        """
        try:
            # Extract skills from both texts
            resume_skills = await self.nlp_processor.extract_skills(resume_text)
            job_skills = await self.nlp_processor.extract_skills(job_description)
            
            missing_skills = []
            
            # Find missing skills across all categories
            for skill_type in ["technical_skills", "soft_skills", "certifications"]:
                resume_set = set(skill.lower() for skill in resume_skills.get(skill_type, []))
                job_set = set(skill.lower() for skill in job_skills.get(skill_type, []))
                
                missing = job_set - resume_set
                missing_skills.extend([skill.title() for skill in missing])
            
            return list(set(missing_skills))
            
        except Exception as e:
            logger.error(f"Missing skills extraction failed: {str(e)}")
            return []
    
    async def get_detailed_analysis(
        self, 
        resume_text: str, 
        job_description: str, 
        similarity_score: float
    ) -> Dict[str, Any]:
        """
        Get detailed analysis of resume vs job description match.
        
        Args:
            resume_text: Resume text content
            job_description: Job description text
            similarity_score: Overall similarity score
            
        Returns:
            Dict containing detailed analysis
        """
        try:
            # Extract information from both texts
            resume_skills = await self.nlp_processor.extract_skills(resume_text)
            job_skills = await self.nlp_processor.extract_skills(job_description)
            resume_experience = await self.nlp_processor.extract_experience(resume_text)
            job_level = await self.nlp_processor.determine_job_level(job_description)
            
            # Calculate skills similarity
            all_resume_skills = []
            all_job_skills = []
            
            for skill_type in ["technical_skills", "soft_skills", "certifications"]:
                all_resume_skills.extend(resume_skills.get(skill_type, []))
                all_job_skills.extend(job_skills.get(skill_type, []))
            
            skills_similarity = await self.calculate_skills_similarity(
                all_resume_skills, all_job_skills
            )
            
            # Analyze strengths and weaknesses
            strengths = []
            weaknesses = []
            
            # Analyze technical skills
            tech_match_ratio = 0
            if job_skills.get("technical_skills"):
                tech_matches = set(skill.lower() for skill in resume_skills.get("technical_skills", [])).intersection(
                    set(skill.lower() for skill in job_skills.get("technical_skills", []))
                )
                tech_match_ratio = len(tech_matches) / len(job_skills["technical_skills"])
                
                if tech_match_ratio > 0.7:
                    strengths.append("Strong technical skills match")
                elif tech_match_ratio < 0.3:
                    weaknesses.append("Limited technical skills match")
            
            # Analyze experience
            if resume_experience.get("total_years", 0) > 0:
                if job_level in ["senior", "lead", "executive"] and resume_experience["total_years"] >= 5:
                    strengths.append("Sufficient experience for senior role")
                elif job_level == "entry" and resume_experience["total_years"] <= 3:
                    strengths.append("Appropriate experience level")
                elif job_level in ["senior", "lead"] and resume_experience["total_years"] < 3:
                    weaknesses.append("May lack experience for senior role")
            
            # Overall assessment
            if similarity_score > 0.8:
                overall_assessment = "Excellent match"
            elif similarity_score > 0.6:
                overall_assessment = "Good match"
            elif similarity_score > 0.4:
                overall_assessment = "Moderate match"
            else:
                overall_assessment = "Poor match"
            
            return {
                "overall_similarity": similarity_score,
                "skills_similarity": skills_similarity,
                "technical_skills_match_ratio": tech_match_ratio,
                "experience_years": resume_experience.get("total_years", 0),
                "job_level": job_level,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "overall_assessment": overall_assessment,
                "resume_skills_count": len(all_resume_skills),
                "job_skills_count": len(all_job_skills)
            }
            
        except Exception as e:
            logger.error(f"Detailed analysis failed: {str(e)}")
            return {
                "overall_similarity": similarity_score,
                "error": str(e)
            }
    
    async def get_recommendation(
        self, 
        similarity_score: float, 
        matching_skills: List[str], 
        missing_skills: List[str]
    ) -> Dict[str, Any]:
        """
        Get recommendation based on scoring results.
        
        Args:
            similarity_score: Overall similarity score
            matching_skills: List of matching skills
            missing_skills: List of missing skills
            
        Returns:
            Dict containing recommendation
        """
        try:
            recommendation = {
                "decision": "",
                "confidence": "",
                "reasons": [],
                "suggestions": []
            }
            
            # Determine decision based on similarity score
            if similarity_score >= 0.8:
                recommendation["decision"] = "Highly Recommended"
                recommendation["confidence"] = "High"
                recommendation["reasons"].append("Excellent overall match")
            elif similarity_score >= 0.6:
                recommendation["decision"] = "Recommended"
                recommendation["confidence"] = "Medium-High"
                recommendation["reasons"].append("Good overall match")
            elif similarity_score >= 0.4:
                recommendation["decision"] = "Consider"
                recommendation["confidence"] = "Medium"
                recommendation["reasons"].append("Moderate match with potential")
            else:
                recommendation["decision"] = "Not Recommended"
                recommendation["confidence"] = "Low"
                recommendation["reasons"].append("Poor match")
            
            # Add specific reasons based on skills
            if len(matching_skills) > 10:
                recommendation["reasons"].append("Strong skills alignment")
            elif len(matching_skills) > 5:
                recommendation["reasons"].append("Decent skills match")
            else:
                recommendation["reasons"].append("Limited skills overlap")
            
            # Add suggestions based on missing skills
            if missing_skills:
                if len(missing_skills) <= 3:
                    recommendation["suggestions"].append(
                        f"Consider developing: {', '.join(missing_skills[:3])}"
                    )
                else:
                    recommendation["suggestions"].append(
                        f"Focus on key missing skills: {', '.join(missing_skills[:5])}"
                    )
            
            # Add general suggestions
            if similarity_score < 0.6:
                recommendation["suggestions"].append(
                    "Consider tailoring resume to better match job requirements"
                )
            
            if len(matching_skills) < 5:
                recommendation["suggestions"].append(
                    "Highlight more relevant skills and experience"
                )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return {
                "decision": "Unable to determine",
                "confidence": "Unknown",
                "reasons": ["Analysis failed"],
                "suggestions": [],
                "error": str(e)
            }
    
    async def batch_score_resumes(
        self, 
        resume_texts: List[str], 
        job_description: str
    ) -> List[Dict[str, Any]]:
        """
        Score multiple resumes against a job description.
        
        Args:
            resume_texts: List of resume texts
            job_description: Job description text
            
        Returns:
            List of scoring results
        """
        results = []
        
        for i, resume_text in enumerate(resume_texts):
            try:
                similarity_score = await self.calculate_similarity(resume_text, job_description)
                matching_skills = await self.find_matching_skills(resume_text, job_description)
                missing_skills = await self.find_missing_skills(resume_text, job_description)
                
                results.append({
                    "resume_index": i,
                    "similarity_score": similarity_score,
                    "matching_skills": matching_skills,
                    "missing_skills": missing_skills,
                    "skills_count": len(matching_skills)
                })
                
            except Exception as e:
                logger.error(f"Batch scoring failed for resume {i}: {str(e)}")
                results.append({
                    "resume_index": i,
                    "error": str(e)
                })
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        return results
    
    async def health_check(self) -> bool:
        """
        Check if resume scorer service is healthy.
        
        Returns:
            bool: True if service is healthy
        """
        try:
            return (
                SKLEARN_AVAILABLE and 
                self.tfidf_vectorizer is not None and
                await self.nlp_processor.health_check()
            )
        except Exception:
            return False
