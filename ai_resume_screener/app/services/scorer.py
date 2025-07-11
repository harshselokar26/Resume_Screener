"""
Resume Scorer Service for AI Resume Screener

This module handles similarity calculation between resumes and job descriptions
using various NLP techniques including TF-IDF and cosine similarity.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import re

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
    Resume scoring service using NLP and ML techniques with enhanced TF-IDF and cosine similarity.
    """
    
    def __init__(self):
        """Initialize resume scorer with NLP processor and vectorizers."""
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.nlp_processor = NLPProcessor()
        self.tfidf_vectorizer = None
        self.skills_vectorizer = None
        self._initialize_vectorizers()
    
    def _initialize_vectorizers(self):
        """Initialize TF-IDF vectorizers with optimized parameters."""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available. Please install: pip install scikit-learn")
            return
        
        try:
            # Enhanced TF-IDF vectorizer for general text similarity
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=8000,           # Increased for better vocabulary coverage
                stop_words='english',
                ngram_range=(1, 3),          # Include trigrams for better context
                min_df=1,                    # Include all terms (important for small corpus)
                max_df=0.95,                 # Remove very common terms
                lowercase=True,
                strip_accents='unicode',
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b',  # Better tokenization
                norm='l2',                   # L2 normalization for cosine similarity
                use_idf=True,                # Use inverse document frequency
                smooth_idf=True,             # Smooth IDF to prevent zero division
                sublinear_tf=True            # Apply sublinear TF scaling
            )
            
            # Skills-specific vectorizer for technical terms
            self.skills_vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 4),          # Capture longer technical terms
                min_df=1,
                lowercase=True,
                strip_accents='unicode',
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9\+\#\.]*\b',  # Include tech symbols
                norm='l2',
                use_idf=True,
                smooth_idf=True
            )
            
            logger.info("Enhanced TF-IDF vectorizers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vectorizers: {str(e)}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better TF-IDF performance.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize common technical terms
        tech_normalizations = {
            r'\bc\+\+\b': 'cplusplus',
            r'\bc#\b': 'csharp',
            r'\bf#\b': 'fsharp',
            r'\bnode\.js\b': 'nodejs',
            r'\breact\.js\b': 'reactjs',
            r'\bvue\.js\b': 'vuejs',
            r'\bangular\.js\b': 'angularjs',
            r'\bmachine learning\b': 'machinelearning',
            r'\bdeep learning\b': 'deeplearning',
            r'\bdata science\b': 'datascience',
            r'\bweb development\b': 'webdevelopment',
            r'\bsoftware development\b': 'softwaredevelopment',
            r'\bproject management\b': 'projectmanagement'
        }
        
        for pattern, replacement in tech_normalizations.items():
            text = re.sub(pattern, replacement, text)
        
        return text.strip()
    
    async def calculate_similarity(self, resume_text: str, job_description: str) -> float:
        """
        Calculate enhanced similarity score using properly trained TF-IDF and cosine similarity.
        
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
                # Preprocess both texts
                resume_processed = self._preprocess_text(resume_text)
                job_processed = self._preprocess_text(job_description)
                
                if not resume_processed or not job_processed:
                    return 0.0
                
                # Create document corpus for training
                documents = [resume_processed, job_processed]
                
                # Fit TF-IDF vectorizer on both documents together
                # This ensures the vocabulary includes terms from both texts
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
                
                # Ensure the matrix is properly normalized for cosine similarity
                tfidf_matrix_normalized = normalize(tfidf_matrix, norm='l2', axis=1)
                
                # Calculate cosine similarity between the two documents
                similarity_matrix = cosine_similarity(tfidf_matrix_normalized)
                
                # Extract similarity between resume (index 0) and job description (index 1)
                similarity_score = float(similarity_matrix[0, 1])
                
                # Additional validation
                if np.isnan(similarity_score) or np.isinf(similarity_score):
                    logger.warning("Invalid similarity score detected, returning 0.0")
                    return 0.0
                
                return similarity_score
            
            loop = asyncio.get_event_loop()
            similarity = await loop.run_in_executor(self.executor, _calculate)
            
            # Ensure similarity is within valid range [0, 1]
            similarity = max(0.0, min(1.0, similarity))
            
            logger.info(f"Calculated TF-IDF cosine similarity: {similarity:.4f}")
            return similarity
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            raise ScoringError(f"Failed to calculate similarity: {str(e)}")
    
    async def calculate_multi_dimensional_similarity(
        self, 
        resume_text: str, 
        job_description: str
    ) -> Dict[str, float]:
        """
        Calculate similarity using multiple approaches for comprehensive analysis.
        
        Args:
            resume_text: Resume text content
            job_description: Job description text
            
        Returns:
            Dict containing different similarity scores
        """
        try:
            def _calculate_multi():
                resume_processed = self._preprocess_text(resume_text)
                job_processed = self._preprocess_text(job_description)
                
                results = {}
                
                # 1. Standard TF-IDF Cosine Similarity
                documents = [resume_processed, job_processed]
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
                tfidf_normalized = normalize(tfidf_matrix, norm='l2', axis=1)
                cosine_sim = cosine_similarity(tfidf_normalized)[0, 1]
                results['tfidf_cosine'] = float(cosine_sim)
                
                # 2. Skills-based TF-IDF Similarity
                if self.skills_vectorizer:
                    skills_matrix = self.skills_vectorizer.fit_transform(documents)
                    skills_normalized = normalize(skills_matrix, norm='l2', axis=1)
                    skills_sim = cosine_similarity(skills_normalized)[0, 1]
                    results['skills_tfidf'] = float(skills_sim)
                else:
                    results['skills_tfidf'] = 0.0
                
                # 3. Jaccard Similarity (token-based)
                resume_tokens = set(resume_processed.split())
                job_tokens = set(job_processed.split())
                
                if resume_tokens and job_tokens:
                    intersection = len(resume_tokens.intersection(job_tokens))
                    union = len(resume_tokens.union(job_tokens))
                    jaccard_sim = intersection / union if union > 0 else 0.0
                    results['jaccard'] = jaccard_sim
                else:
                    results['jaccard'] = 0.0
                
                # 4. Overlap Coefficient
                if resume_tokens and job_tokens:
                    intersection = len(resume_tokens.intersection(job_tokens))
                    min_size = min(len(resume_tokens), len(job_tokens))
                    overlap_coeff = intersection / min_size if min_size > 0 else 0.0
                    results['overlap_coefficient'] = overlap_coeff
                else:
                    results['overlap_coefficient'] = 0.0
                
                return results
            
            loop = asyncio.get_event_loop()
            similarities = await loop.run_in_executor(self.executor, _calculate_multi)
            
            # Validate all scores
            for key, value in similarities.items():
                if np.isnan(value) or np.isinf(value):
                    similarities[key] = 0.0
                else:
                    similarities[key] = max(0.0, min(1.0, value))
            
            return similarities
            
        except Exception as e:
            logger.error(f"Multi-dimensional similarity calculation failed: {str(e)}")
            return {
                'tfidf_cosine': 0.0,
                'skills_tfidf': 0.0,
                'jaccard': 0.0,
                'overlap_coefficient': 0.0
            }
    
    async def calculate_weighted_similarity(
        self, 
        resume_text: str, 
        job_description: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate weighted similarity combining multiple similarity measures.
        
        Args:
            resume_text: Resume text content
            job_description: Job description text
            
        Returns:
            Tuple of (final_score, component_scores)
        """
        try:
            # Get all similarity measures
            similarities = await self.calculate_multi_dimensional_similarity(
                resume_text, job_description
            )
            
            # Define weights for different similarity measures
            weights = {
                'tfidf_cosine': 0.4,        # Primary semantic similarity
                'skills_tfidf': 0.3,        # Technical skills similarity
                'jaccard': 0.2,             # Token overlap
                'overlap_coefficient': 0.1   # Coverage similarity
            }
            
            # Calculate weighted average
            weighted_score = sum(
                similarities.get(metric, 0.0) * weight 
                for metric, weight in weights.items()
            )
            
            # Ensure final score is in valid range
            final_score = max(0.0, min(1.0, weighted_score))
            
            logger.info(f"Weighted similarity score: {final_score:.4f}")
            logger.debug(f"Component scores: {similarities}")
            
            return final_score, similarities
            
        except Exception as e:
            logger.error(f"Weighted similarity calculation failed: {str(e)}")
            return 0.0, {}
    
    async def calculate_skills_similarity(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """
        Calculate enhanced similarity based on skills overlap with TF-IDF weighting.
        
        Args:
            resume_skills: List of skills from resume
            job_skills: List of skills from job description
            
        Returns:
            float: Skills similarity score between 0 and 1
        """
        if not resume_skills or not job_skills:
            return 0.0
        
        try:
            # Normalize skills for comparison
            resume_set = set(skill.lower().strip() for skill in resume_skills if skill.strip())
            job_set = set(skill.lower().strip() for skill in job_skills if skill.strip())
            
            if not resume_set or not job_set:
                return 0.0
            
            # Calculate multiple similarity metrics
            intersection = resume_set.intersection(job_set)
            union = resume_set.union(job_set)
            
            # Jaccard similarity
            jaccard_sim = len(intersection) / len(union) if union else 0.0
            
            # Coverage similarity (how much of job requirements are covered)
            coverage_sim = len(intersection) / len(job_set) if job_set else 0.0
            
            # Precision similarity (how relevant are resume skills)
            precision_sim = len(intersection) / len(resume_set) if resume_set else 0.0
            
            # F1-like score combining coverage and precision
            if coverage_sim + precision_sim > 0:
                f1_sim = 2 * (coverage_sim * precision_sim) / (coverage_sim + precision_sim)
            else:
                f1_sim = 0.0
            
            # Weighted combination emphasizing job requirement coverage
            final_score = (
                jaccard_sim * 0.2 +      # General overlap
                coverage_sim * 0.5 +     # Job requirement coverage (most important)
                precision_sim * 0.1 +    # Resume relevance
                f1_sim * 0.2             # Balanced measure
            )
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.error(f"Skills similarity calculation failed: {str(e)}")
            return 0.0
    
    async def find_matching_skills(self, resume_text: str, job_description: str) -> List[str]:
        """
        Find skills that match between resume and job description using enhanced extraction.
        
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
                resume_set = set(skill.lower().strip() for skill in resume_skills.get(skill_type, []))
                job_set = set(skill.lower().strip() for skill in job_skills.get(skill_type, []))
                
                matches = resume_set.intersection(job_set)
                matching_skills.extend([skill.title() for skill in matches if skill])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_matches = []
            for skill in matching_skills:
                if skill.lower() not in seen:
                    seen.add(skill.lower())
                    unique_matches.append(skill)
            
            logger.info(f"Found {len(unique_matches)} matching skills")
            return unique_matches
            
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
                resume_set = set(skill.lower().strip() for skill in resume_skills.get(skill_type, []))
                job_set = set(skill.lower().strip() for skill in job_skills.get(skill_type, []))
                
                missing = job_set - resume_set
                missing_skills.extend([skill.title() for skill in missing if skill])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_missing = []
            for skill in missing_skills:
                if skill.lower() not in seen:
                    seen.add(skill.lower())
                    unique_missing.append(skill)
            
            logger.info(f"Found {len(unique_missing)} missing skills")
            return unique_missing
            
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
        Get detailed analysis with enhanced similarity metrics.
        
        Args:
            resume_text: Resume text content
            job_description: Job description text
            similarity_score: Overall similarity score
            
        Returns:
            Dict containing detailed analysis
        """
        try:
            # Get multi-dimensional similarity scores
            weighted_score, component_scores = await self.calculate_weighted_similarity(
                resume_text, job_description
            )
            
            # Extract information from both texts
            resume_skills = await self.nlp_processor.extract_skills(resume_text)
            job_skills = await self.nlp_processor.extract_skills(job_description)
            resume_experience = await self.nlp_processor.extract_experience(resume_text)
            job_level = await self.nlp_processor.determine_job_level(job_description)
            
            # Calculate enhanced skills similarity
            all_resume_skills = []
            all_job_skills = []
            
            for skill_type in ["technical_skills", "soft_skills", "certifications"]:
                all_resume_skills.extend(resume_skills.get(skill_type, []))
                all_job_skills.extend(job_skills.get(skill_type, []))
            
            skills_similarity = await self.calculate_skills_similarity(
                all_resume_skills, all_job_skills
            )
            
            # Analyze technical skills match
            tech_match_ratio = 0
            if job_skills.get("technical_skills"):
                tech_matches = set(skill.lower() for skill in resume_skills.get("technical_skills", [])).intersection(
                    set(skill.lower() for skill in job_skills.get("technical_skills", []))
                )
                tech_match_ratio = len(tech_matches) / len(job_skills["technical_skills"])
            
            # Generate insights
            strengths = []
            weaknesses = []
            
            # Technical skills analysis
            if tech_match_ratio > 0.8:
                strengths.append("Excellent technical skills alignment")
            elif tech_match_ratio > 0.6:
                strengths.append("Strong technical skills match")
            elif tech_match_ratio > 0.3:
                strengths.append("Moderate technical skills overlap")
            else:
                weaknesses.append("Limited technical skills match")
            
            # Experience analysis
            experience_years = resume_experience.get("total_years", 0)
            if experience_years > 0:
                if job_level in ["senior", "lead", "executive"]:
                    if experience_years >= 5:
                        strengths.append("Sufficient experience for senior role")
                    elif experience_years >= 3:
                        strengths.append("Adequate experience, may need growth")
                    else:
                        weaknesses.append("May lack experience for senior role")
                elif job_level == "mid":
                    if experience_years >= 2:
                        strengths.append("Good experience for mid-level role")
                    else:
                        weaknesses.append("Limited experience for mid-level role")
                elif job_level == "entry":
                    if experience_years <= 3:
                        strengths.append("Appropriate experience for entry-level")
                    else:
                        strengths.append("Overqualified - brings extra experience")
            
            # Similarity score analysis
            if weighted_score > 0.8:
                overall_assessment = "Excellent match - highly recommended"
            elif weighted_score > 0.6:
                overall_assessment = "Good match - recommended"
            elif weighted_score > 0.4:
                overall_assessment = "Moderate match - consider with reservations"
            else:
                overall_assessment = "Poor match - not recommended"
            
            return {
                "overall_similarity": similarity_score,
                "weighted_similarity": weighted_score,
                "component_similarities": component_scores,
                "skills_similarity": skills_similarity,
                "technical_skills_match_ratio": tech_match_ratio,
                "experience_years": experience_years,
                "job_level": job_level,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "overall_assessment": overall_assessment,
                "resume_skills_count": len(all_resume_skills),
                "job_skills_count": len(all_job_skills),
                "analysis_confidence": "high" if len(all_resume_skills) > 5 and len(all_job_skills) > 3 else "medium"
            }
            
        except Exception as e:
            logger.error(f"Detailed analysis failed: {str(e)}")
            return {
                "overall_similarity": similarity_score,
                "error": str(e),
                "analysis_confidence": "low"
            }
    
    async def get_recommendation(
        self, 
        similarity_score: float, 
        matching_skills: List[str], 
        missing_skills: List[str]
    ) -> Dict[str, Any]:
        """
        Get enhanced recommendation based on comprehensive scoring results.
        
        Args:
            similarity_score: Overall similarity score
            matching_skills: List of matching skills
            missing_skills: List of missing skills
            
        Returns:
            Dict containing detailed recommendation
        """
        try:
            recommendation = {
                "decision": "",
                "confidence": "",
                "reasons": [],
                "suggestions": [],
                "score_breakdown": {
                    "similarity": similarity_score,
                    "skills_match": len(matching_skills),
                    "skills_gap": len(missing_skills)
                }
            }
            
            # Enhanced decision logic
            skills_ratio = len(matching_skills) / (len(matching_skills) + len(missing_skills)) if (len(matching_skills) + len(missing_skills)) > 0 else 0
            
            # Multi-factor decision making
            if similarity_score >= 0.8 and skills_ratio >= 0.7:
                recommendation["decision"] = "Highly Recommended"
                recommendation["confidence"] = "High"
                recommendation["reasons"].extend([
                    "Excellent semantic similarity",
                    "Strong skills alignment",
                    "Comprehensive match across requirements"
                ])
            elif similarity_score >= 0.6 and skills_ratio >= 0.5:
                recommendation["decision"] = "Recommended"
                recommendation["confidence"] = "Medium-High"
                recommendation["reasons"].extend([
                    "Good overall match",
                    "Solid skills foundation"
                ])
            elif similarity_score >= 0.4 or skills_ratio >= 0.4:
                recommendation["decision"] = "Consider"
                recommendation["confidence"] = "Medium"
                recommendation["reasons"].append("Shows potential with some gaps")
            else:
                recommendation["decision"] = "Not Recommended"
                recommendation["confidence"] = "Low"
                recommendation["reasons"].append("Significant gaps in requirements")
            
            # Detailed skill-based reasons
            if len(matching_skills) > 15:
                recommendation["reasons"].append("Exceptional skills breadth")
            elif len(matching_skills) > 10:
                recommendation["reasons"].append("Strong skills portfolio")
            elif len(matching_skills) > 5:
                recommendation["reasons"].append("Adequate skills coverage")
            else:
                recommendation["reasons"].append("Limited skills overlap")
            
            # Targeted suggestions
            if missing_skills:
                critical_missing = missing_skills[:5]  # Focus on top 5
                if len(critical_missing) <= 2:
                    recommendation["suggestions"].append(
                        f"Minor skill gaps: {', '.join(critical_missing)}"
                    )
                elif len(critical_missing) <= 5:
                    recommendation["suggestions"].append(
                        f"Key areas for development: {', '.join(critical_missing)}"
                    )
                else:
                    recommendation["suggestions"].append(
                        f"Significant skill gaps requiring attention: {', '.join(critical_missing)}"
                    )
            
            # Performance-based suggestions
            if similarity_score < 0.6:
                recommendation["suggestions"].append(
                    "Consider resume optimization to better highlight relevant experience"
                )
            
            if len(matching_skills) < 5:
                recommendation["suggestions"].append(
                    "Emphasize transferable skills and relevant projects"
                )
            
            if skills_ratio < 0.3:
                recommendation["suggestions"].append(
                    "Focus on developing core competencies for this role"
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
        Score multiple resumes with enhanced similarity calculation.
        
        Args:
            resume_texts: List of resume texts
            job_description: Job description text
            
        Returns:
            List of scoring results sorted by similarity
        """
        results = []
        
        for i, resume_text in enumerate(resume_texts):
            try:
                # Calculate weighted similarity
                weighted_score, component_scores = await self.calculate_weighted_similarity(
                    resume_text, job_description
                )
                
                matching_skills = await self.find_matching_skills(resume_text, job_description)
                missing_skills = await self.find_missing_skills(resume_text, job_description)
                
                results.append({
                    "resume_index": i,
                    "similarity_score": weighted_score,
                    "component_scores": component_scores,
                    "matching_skills": matching_skills,
                    "missing_skills": missing_skills,
                    "skills_count": len(matching_skills),
                    "skills_gap": len(missing_skills)
                })
                
            except Exception as e:
                logger.error(f"Batch scoring failed for resume {i}: {str(e)}")
                results.append({
                    "resume_index": i,
                    "error": str(e),
                    "similarity_score": 0.0
                })
        
        # Sort by weighted similarity score (descending)
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
                self.skills_vectorizer is not None and
                await self.nlp_processor.health_check()
            )
        except Exception:
            return False
