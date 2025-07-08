"""
NLP Processor Service for AI Resume Screener

This module handles all NLP operations including text cleaning,
skill extraction, keyword identification, and text analysis using spaCy.
"""

import re
import logging
from typing import List, Dict, Set, Any, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

# NLP libraries
try:
    import spacy
    from spacy.matcher import Matcher, PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

import json
from pathlib import Path

from app.config.settings import settings
from app.utils.exceptions import NLPProcessingError

# Setup logging
logger = logging.getLogger(__name__)


class NLPProcessor:
    """
    NLP processor for resume and job description analysis.
    """
    
    def __init__(self):
        """Initialize NLP processor with spaCy model and skill databases."""
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.nlp = None
        self.matcher = None
        self.phrase_matcher = None
        self.skills_db = {}
        self._initialize_nlp()
        self._load_skills_database()
    
    def _initialize_nlp(self):
        """Initialize spaCy NLP model."""
        if not SPACY_AVAILABLE:
            logger.error("spaCy not available. Please install: pip install spacy")
            return
        
        try:
            self.nlp = spacy.load(settings.SPACY_MODEL)
            self.matcher = Matcher(self.nlp.vocab)
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            self._setup_patterns()
            logger.info(f"spaCy model '{settings.SPACY_MODEL}' loaded successfully")
        except OSError:
            logger.error(f"spaCy model '{settings.SPACY_MODEL}' not found. Please install it.")
        except Exception as e:
            logger.error(f"Failed to initialize spaCy: {str(e)}")
    
    def _setup_patterns(self):
        """Setup matching patterns for skills and entities."""
        if not self.matcher:
            return
        
        # Email pattern
        email_pattern = [
            {"LIKE_EMAIL": True}
        ]
        self.matcher.add("EMAIL", [email_pattern])
        
        # Phone pattern
        phone_pattern = [
            {"TEXT": {"REGEX": r"(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"}},
        ]
        self.matcher.add("PHONE", [phone_pattern])
        
        # Years of experience pattern
        experience_patterns = [
            [{"LOWER": {"IN": ["over", "more", "than"]}}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["years", "year", "yrs", "yr"]}}, {"LOWER": {"IN": ["of", "experience", "exp"]}}],
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["years", "year", "yrs", "yr"]}}, {"LOWER": {"IN": ["of", "experience", "exp"]}}],
            [{"LIKE_NUM": True}, {"TEXT": "+"}, {"LOWER": {"IN": ["years", "year", "yrs", "yr"]}}]
        ]
        for i, pattern in enumerate(experience_patterns):
            self.matcher.add(f"EXPERIENCE_{i}", [pattern])
    
    def _load_skills_database(self):
        """Load skills database from JSON file."""
        try:
            skills_file = Path("data/skills_database.json")
            if skills_file.exists():
                with open(skills_file, 'r', encoding='utf-8') as f:
                    self.skills_db = json.load(f)
            else:
                # Create default skills database
                self.skills_db = self._create_default_skills_db()
                self._save_skills_database()
            
            # Setup phrase matcher for skills
            self._setup_skills_matcher()
            logger.info(f"Loaded {len(self.skills_db.get('technical_skills', []))} technical skills")
            
        except Exception as e:
            logger.error(f"Failed to load skills database: {str(e)}")
            self.skills_db = self._create_default_skills_db()
    
    def _create_default_skills_db(self) -> Dict[str, List[str]]:
        """Create default skills database."""
        return {
            "technical_skills": [
                # Programming Languages
                "Python", "Java", "JavaScript", "C++", "C#", "Go", "Rust", "Ruby", "PHP", "Swift",
                "Kotlin", "Scala", "R", "MATLAB", "SQL", "HTML", "CSS", "TypeScript",
                
                # Frameworks & Libraries
                "React", "Angular", "Vue.js", "Node.js", "Express.js", "Django", "Flask", "FastAPI",
                "Spring Boot", "Laravel", "Ruby on Rails", "ASP.NET", "jQuery", "Bootstrap",
                
                # Databases
                "MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite", "Oracle", "SQL Server",
                "Elasticsearch", "Cassandra", "DynamoDB",
                
                # Cloud & DevOps
                "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Jenkins", "Git", "GitHub",
                "GitLab", "CI/CD", "Terraform", "Ansible",
                
                # Data Science & ML
                "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn",
                "Pandas", "NumPy", "Matplotlib", "Seaborn", "Jupyter", "Apache Spark",
                
                # Other Technologies
                "Linux", "Windows", "macOS", "REST API", "GraphQL", "Microservices", "Agile", "Scrum"
            ],
            "soft_skills": [
                "Communication", "Leadership", "Teamwork", "Problem Solving", "Critical Thinking",
                "Time Management", "Project Management", "Adaptability", "Creativity", "Analytical",
                "Attention to Detail", "Customer Service", "Presentation", "Negotiation", "Mentoring"
            ],
            "certifications": [
                "AWS Certified", "Azure Certified", "Google Cloud Certified", "PMP", "Scrum Master",
                "CISSP", "CompTIA", "Cisco Certified", "Microsoft Certified", "Oracle Certified"
            ]
        }
    
    def _save_skills_database(self):
        """Save skills database to JSON file."""
        try:
            skills_file = Path("data/skills_database.json")
            skills_file.parent.mkdir(exist_ok=True)
            with open(skills_file, 'w', encoding='utf-8') as f:
                json.dump(self.skills_db, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save skills database: {str(e)}")
    
    def _setup_skills_matcher(self):
        """Setup phrase matcher for skills."""
        if not self.phrase_matcher or not self.nlp:
            return
        
        try:
            # Add technical skills
            technical_skills = [self.nlp(skill.lower()) for skill in self.skills_db.get("technical_skills", [])]
            if technical_skills:
                self.phrase_matcher.add("TECHNICAL_SKILL", technical_skills)
            
            # Add soft skills
            soft_skills = [self.nlp(skill.lower()) for skill in self.skills_db.get("soft_skills", [])]
            if soft_skills:
                self.phrase_matcher.add("SOFT_SKILL", soft_skills)
            
            # Add certifications
            certifications = [self.nlp(cert.lower()) for cert in self.skills_db.get("certifications", [])]
            if certifications:
                self.phrase_matcher.add("CERTIFICATION", certifications)
                
        except Exception as e:
            logger.error(f"Failed to setup skills matcher: {str(e)}")
    
    async def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep important punctuation
            text = re.sub(r'[^\w\s\.,;:()\-@]', '', text)
            
            # Remove multiple consecutive punctuation
            text = re.sub(r'[.,;:]{2,}', '.', text)
            
            # Strip and return
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text cleaning failed: {str(e)}")
            return text
    
    async def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict containing categorized skills
        """
        if not self.nlp or not text:
            return {"technical_skills": [], "soft_skills": [], "certifications": []}
        
        try:
            def _extract():
                doc = self.nlp(text.lower())
                
                skills = {
                    "technical_skills": set(),
                    "soft_skills": set(),
                    "certifications": set()
                }
                
                # Use phrase matcher
                matches = self.phrase_matcher(doc)
                for match_id, start, end in matches:
                    label = self.nlp.vocab.strings[match_id]
                    span = doc[start:end]
                    
                    if label == "TECHNICAL_SKILL":
                        skills["technical_skills"].add(span.text.title())
                    elif label == "SOFT_SKILL":
                        skills["soft_skills"].add(span.text.title())
                    elif label == "CERTIFICATION":
                        skills["certifications"].add(span.text.title())
                
                # Convert sets to lists
                return {k: list(v) for k, v in skills.items()}
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, _extract)
            
        except Exception as e:
            logger.error(f"Skills extraction failed: {str(e)}")
            raise NLPProcessingError(f"Failed to extract skills: {str(e)}")
    
    async def extract_experience(self, text: str) -> Dict[str, Any]:
        """
        Extract experience information from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict containing experience information
        """
        if not self.nlp or not text:
            return {"total_years": 0, "details": []}
        
        try:
            def _extract():
                doc = self.nlp(text.lower())
                experience_info = {
                    "total_years": 0,
                    "details": []
                }
                
                # Find experience mentions
                matches = self.matcher(doc)
                years_found = []
                
                for match_id, start, end in matches:
                    label = self.nlp.vocab.strings[match_id]
                    if label.startswith("EXPERIENCE"):
                        span = doc[start:end]
                        # Extract numbers from the span
                        numbers = [token.text for token in span if token.like_num]
                        if numbers:
                            try:
                                years = int(numbers[0])
                                years_found.append(years)
                                experience_info["details"].append({
                                    "text": span.text,
                                    "years": years
                                })
                            except ValueError:
                                pass
                
                # Use the maximum years found
                if years_found:
                    experience_info["total_years"] = max(years_found)
                
                return experience_info
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, _extract)
            
        except Exception as e:
            logger.error(f"Experience extraction failed: {str(e)}")
            return {"total_years": 0, "details": []}
    
    async def extract_education(self, text: str) -> List[Dict[str, str]]:
        """
        Extract education information from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of education entries
        """
        if not self.nlp or not text:
            return []
        
        try:
            def _extract():
                doc = self.nlp(text)
                education = []
                
                # Common education keywords
                education_keywords = [
                    "bachelor", "master", "phd", "doctorate", "degree", "diploma",
                    "university", "college", "institute", "school", "education",
                    "b.s.", "b.a.", "m.s.", "m.a.", "mba", "b.tech", "m.tech"
                ]
                
                # Find sentences containing education keywords
                for sent in doc.sents:
                    sent_text = sent.text.lower()
                    if any(keyword in sent_text for keyword in education_keywords):
                        education.append({
                            "text": sent.text.strip(),
                            "type": "education"
                        })
                
                return education
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, _extract)
            
        except Exception as e:
            logger.error(f"Education extraction failed: {str(e)}")
            return []
    
    async def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Text to analyze
            top_n: Number of top keywords to return
            
        Returns:
            List of important keywords
        """
        if not self.nlp or not text:
            return []
        
        try:
            def _extract():
                doc = self.nlp(text.lower())
                
                # Extract meaningful tokens
                keywords = []
                for token in doc:
                    if (not token.is_stop and 
                        not token.is_punct and 
                        not token.is_space and
                        len(token.text) > 2 and
                        token.pos_ in ['NOUN', 'PROPN', 'ADJ']):
                        keywords.append(token.lemma_)
                
                # Count frequency and return top keywords
                from collections import Counter
                keyword_counts = Counter(keywords)
                return [word for word, count in keyword_counts.most_common(top_n)]
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, _extract)
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return []
    
    async def extract_basic_info(self, text: str) -> Dict[str, Any]:
        """
        Extract basic information from resume text.
        
        Args:
            text: Resume text to analyze
            
        Returns:
            Dict containing basic information
        """
        if not self.nlp or not text:
            return {}
        
        try:
            def _extract():
                doc = self.nlp(text)
                info = {
                    "emails": [],
                    "phones": [],
                    "names": [],
                    "organizations": []
                }
                
                # Extract entities
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        info["names"].append(ent.text)
                    elif ent.label_ == "ORG":
                        info["organizations"].append(ent.text)
                
                # Extract emails and phones using matcher
                matches = self.matcher(doc)
                for match_id, start, end in matches:
                    label = self.nlp.vocab.strings[match_id]
                    span = doc[start:end]
                    
                    if label == "EMAIL":
                        info["emails"].append(span.text)
                    elif label == "PHONE":
                        info["phones"].append(span.text)
                
                # Remove duplicates
                for key in info:
                    info[key] = list(set(info[key]))
                
                return info
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, _extract)
            
        except Exception as e:
            logger.error(f"Basic info extraction failed: {str(e)}")
            return {}
    
    async def find_relevant_skills(self, resume_text: str, job_description: str) -> List[str]:
        """
        Find skills from resume that are relevant to job description.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            List of relevant skills
        """
        try:
            # Extract skills from both texts
            resume_skills = await self.extract_skills(resume_text)
            job_skills = await self.extract_skills(job_description)
            
            # Find intersection
            relevant_skills = []
            
            for skill_type in ["technical_skills", "soft_skills", "certifications"]:
                resume_set = set(skill.lower() for skill in resume_skills.get(skill_type, []))
                job_set = set(skill.lower() for skill in job_skills.get(skill_type, []))
                
                common_skills = resume_set.intersection(job_set)
                relevant_skills.extend([skill.title() for skill in common_skills])
            
            return list(set(relevant_skills))
            
        except Exception as e:
            logger.error(f"Relevant skills extraction failed: {str(e)}")
            return []
    
    async def extract_required_skills(self, job_description: str) -> List[str]:
        """
        Extract required skills from job description.
        
        Args:
            job_description: Job description text
            
        Returns:
            List of required skills
        """
        skills_data = await self.extract_skills(job_description)
        
        # Combine all skill types
        all_skills = []
        for skill_type in skills_data:
            all_skills.extend(skills_data[skill_type])
        
        return list(set(all_skills))
    
    async def extract_requirements(self, job_description: str) -> List[str]:
        """
        Extract job requirements from job description.
        
        Args:
            job_description: Job description text
            
        Returns:
            List of requirements
        """
        if not self.nlp or not job_description:
            return []
        
        try:
            def _extract():
                doc = self.nlp(job_description.lower())
                requirements = []
                
                # Look for requirement indicators
                requirement_indicators = [
                    "required", "must have", "should have", "need", "necessary",
                    "essential", "mandatory", "minimum", "preferred", "desired"
                ]
                
                for sent in doc.sents:
                    sent_text = sent.text.lower()
                    if any(indicator in sent_text for indicator in requirement_indicators):
                        requirements.append(sent.text.strip())
                
                return requirements
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, _extract)
            
        except Exception as e:
            logger.error(f"Requirements extraction failed: {str(e)}")
            return []
    
    async def determine_job_level(self, job_description: str) -> str:
        """
        Determine job seniority level from job description.
        
        Args:
            job_description: Job description text
            
        Returns:
            str: Job level (entry, mid, senior, lead, executive)
        """
        if not job_description:
            return "unknown"
        
        text = job_description.lower()
        
        # Define level indicators
        level_indicators = {
            "entry": ["entry", "junior", "associate", "trainee", "intern", "graduate", "0-2 years"],
            "mid": ["mid", "intermediate", "2-5 years", "3-6 years", "experienced"],
            "senior": ["senior", "sr", "5+ years", "6+ years", "lead", "principal"],
            "lead": ["lead", "team lead", "technical lead", "architect", "manager"],
            "executive": ["director", "vp", "vice president", "chief", "head of", "executive"]
        }
        
        # Count matches for each level
        level_scores = {}
        for level, indicators in level_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            level_scores[level] = score
        
        # Return level with highest score
        if max(level_scores.values()) > 0:
            return max(level_scores, key=level_scores.get)
        
        return "unknown"
    
    async def health_check(self) -> bool:
        """
        Check if NLP processor service is healthy.
        
        Returns:
            bool: True if service is healthy
        """
        try:
            return self.nlp is not None and SPACY_AVAILABLE
        except Exception:
            return False
