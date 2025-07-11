"""
Enhanced NLP Processor Service for AI Resume Screener

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
from collections import defaultdict, Counter

from app.config.settings import settings
from app.utils.exceptions import NLPProcessingError

# Setup logging
logger = logging.getLogger(__name__)

# Entity cleanup helpers
_STOPWORDS = {
    "inc", "llc", "ltd", "plc", "co", "corp", "company", "organization",
    "technologies", "technology", "solutions", "group", "services", "systems",
    "software", "consulting", "international", "global", "enterprises",
    "the", "and", "of", "for", "with", "at", "in", "on", "to", "a", "an"
}

_MIN_ENTITY_LEN = 3  # discard very short tokens such as "Mr"

# Job level detection patterns
_JOB_LEVEL_MAP = {
    "entry": [
        "entry level", "graduate", "trainee", r"\bjunior\b", r"\bjr\b", 
        "intern", "associate", "0-2 years", "fresh graduate", "new grad",
        "entry-level", "beginner", "starting"
    ],
    "mid": [
        "mid level", "intermediate", r"\bmid\b", r"\bmid-level\b", 
        "2-5 years", "3-6 years", "experienced", "2+ years", "3+ years",
        "mid-level", "regular", "standard"
    ],
    "senior": [
        "senior", r"\bsr\b", r"\bsen\b", "lead engineer", "principal",
        "5+ years", "6+ years", "expert", "advanced", "senior-level",
        "seasoned", "specialist", "consultant"
    ],
    "lead": [
        "tech lead", "team lead", "staff engineer", "architect", "lead",
        "technical lead", "team leader", "engineering lead", "lead developer",
        "lead engineer", "technical architect", "solution architect"
    ],
    "exec": [
        "director", "vice president", r"\bvp\b", "cto", "chief", "head of",
        "executive", "manager", "senior manager", "engineering manager",
        "department head", "c-level", "ceo", "cfo", "coo"
    ]
}


class EnhancedNLPProcessor:
    """
    Enhanced NLP processor for resume and job description analysis.
    """
    
    def __init__(self):
        """Initialize NLP processor with spaCy model and enhanced skill databases."""
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.nlp = None
        self.matcher = None
        self.phrase_matcher = None
        self.skills_db = {}
        self.skill_patterns = {}
        self.synonym_map = {}
        self.category_skills = defaultdict(list)
        
        self._initialize_nlp()
        self._load_enhanced_skills_database()
        self._build_skill_patterns()
    
    def _initialize_nlp(self):
        """Initialize spaCy NLP model with enhanced configuration."""
        if not SPACY_AVAILABLE:
            logger.error("spaCy not available. Please install: pip install spacy")
            return
        
        try:
            self.nlp = spacy.load(settings.SPACY_MODEL)
            
            # Add custom pipeline components if needed
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
            
            self.matcher = Matcher(self.nlp.vocab)
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            self._setup_enhanced_patterns()
            
            logger.info(f"Enhanced spaCy model '{settings.SPACY_MODEL}' loaded successfully")
        except OSError:
            logger.error(f"spaCy model '{settings.SPACY_MODEL}' not found. Please install it.")
        except Exception as e:
            logger.error(f"Failed to initialize spaCy: {str(e)}")
    
    def _clean_entities(self, ents: List[str]) -> List[str]:
        """
        Post-filter PERSON / ORG entities:
        • lower-case stop-word removal
        • length check
        • de-duplication while preserving order
        """
        seen, cleaned = set(), []
        for ent in ents:
            if not ent or not ent.strip():
                continue
                
            # Split entity into tokens and clean
            tokens = [t.strip() for t in re.split(r"[^\w\s]", ent) if t.strip()]
            
            # Filter out stop words and very short tokens
            filtered_tokens = []
            for token in tokens:
                if (len(token) >= 2 and 
                    token.lower() not in _STOPWORDS and
                    not token.isdigit() and
                    not re.match(r'^[^\w]+$', token)):
                    filtered_tokens.append(token)
            
            if not filtered_tokens:
                continue
                
            # Reconstruct entity
            cleaned_entity = " ".join(filtered_tokens)
            
            # Final length and quality checks
            if (len(cleaned_entity) >= _MIN_ENTITY_LEN and
                not all(t.lower() in _STOPWORDS for t in filtered_tokens) and
                cleaned_entity.lower() not in seen):
                seen.add(cleaned_entity.lower())
                cleaned.append(cleaned_entity)
        
        return cleaned
    
    def _load_enhanced_skills_database(self):
        """Load and process the enhanced skills database."""
        try:
            skills_file = Path("data/skills_database.json")
            if skills_file.exists():
                with open(skills_file, 'r', encoding='utf-8') as f:
                    self.skills_db = json.load(f)
            else:
                logger.warning("Skills database not found, creating default")
                self.skills_db = self._create_default_skills_db()
                self._save_skills_database()
            
            # Process the hierarchical structure
            self._process_skills_structure()
            
            # Setup phrase matcher for skills
            self._setup_enhanced_skills_matcher()
            
            total_skills = sum(len(self.category_skills[cat]) for cat in self.category_skills)
            logger.info(f"Loaded {total_skills} skills across {len(self.category_skills)} categories")
            
        except Exception as e:
            logger.error(f"Failed to load skills database: {str(e)}")
            self.skills_db = self._create_default_skills_db()
    
    def _create_default_skills_db(self) -> Dict[str, Any]:
        """Create default skills database if file doesn't exist."""
        return {
            "technical_skills": [
                {
                    "category": "Programming Languages",
                    "skills": [
                        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
                        "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB", "SQL", "HTML",
                        "CSS", "Dart", "Perl", "Shell Scripting", "PowerShell"
                    ]
                },
                {
                    "category": "Web Frameworks",
                    "skills": [
                        "React", "Angular", "Vue.js", "Node.js", "Express.js", "Django", "Flask",
                        "FastAPI", "Spring Boot", "Laravel", "Ruby on Rails", "ASP.NET", "jQuery",
                        "Bootstrap", "Tailwind CSS", "Next.js", "Nuxt.js", "Svelte"
                    ]
                },
                {
                    "category": "Databases",
                    "skills": [
                        "MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite", "Oracle", "SQL Server",
                        "Elasticsearch", "Cassandra", "DynamoDB", "Neo4j", "InfluxDB"
                    ]
                },
                {
                    "category": "Cloud Platforms",
                    "skills": [
                        "AWS", "Azure", "Google Cloud Platform", "Docker", "Kubernetes", "Jenkins",
                        "Git", "GitHub", "GitLab", "CI/CD", "Terraform", "Ansible"
                    ]
                },
                {
                    "category": "Data Science & ML",
                    "skills": [
                        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn",
                        "Pandas", "NumPy", "Matplotlib", "Seaborn", "Jupyter", "Apache Spark",
                        "spaCy", "NLTK", "Computer Vision", "Natural Language Processing"
                    ]
                }
            ],
            "soft_skills": [
                {
                    "category": "Communication",
                    "skills": [
                        "Communication", "Presentation Skills", "Technical Writing", "Documentation"
                    ]
                },
                {
                    "category": "Leadership",
                    "skills": [
                        "Leadership", "Team Management", "Project Management", "Mentoring", "Coaching"
                    ]
                },
                {
                    "category": "Problem Solving",
                    "skills": [
                        "Problem Solving", "Critical Thinking", "Analytical Skills", "Troubleshooting"
                    ]
                }
            ],
            "certifications": [
                {
                    "category": "Cloud Certifications",
                    "certifications": [
                        "AWS Certified", "Azure Certified", "Google Cloud Certified"
                    ]
                },
                {
                    "category": "Project Management",
                    "certifications": [
                        "PMP", "Scrum Master", "Agile Certified"
                    ]
                }
            ],
            "skill_synonyms": {
                "JavaScript": ["JS", "ECMAScript"],
                "Node.js": ["NodeJS", "Node"],
                "React": ["ReactJS", "React.js"],
                "Python": ["Python3", "Py"],
                "PostgreSQL": ["Postgres"],
                "MongoDB": ["Mongo"],
                "Amazon Web Services": ["AWS"],
                "Machine Learning": ["ML"],
                "Artificial Intelligence": ["AI"]
            }
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
    
    def _process_skills_structure(self):
        """Process the hierarchical skills structure and build lookup maps."""
        # Process technical skills
        if "technical_skills" in self.skills_db:
            for category_data in self.skills_db["technical_skills"]:
                category = category_data.get("category", "General")
                skills = category_data.get("skills", [])
                self.category_skills[f"technical_{category}"] = skills
        
        # Process soft skills
        if "soft_skills" in self.skills_db:
            for category_data in self.skills_db["soft_skills"]:
                category = category_data.get("category", "General")
                skills = category_data.get("skills", [])
                self.category_skills[f"soft_{category}"] = skills
        
        # Process certifications
        if "certifications" in self.skills_db:
            for category_data in self.skills_db["certifications"]:
                category = category_data.get("category", "General")
                certs = category_data.get("certifications", [])
                self.category_skills[f"cert_{category}"] = certs
        
        # Build synonym map
        if "skill_synonyms" in self.skills_db:
            for main_skill, synonyms in self.skills_db["skill_synonyms"].items():
                self.synonym_map[main_skill.lower()] = main_skill
                for synonym in synonyms:
                    self.synonym_map[synonym.lower()] = main_skill
    
    def _setup_enhanced_patterns(self):
        """Setup enhanced matching patterns for various entities."""
        if not self.matcher:
            return
        
        # Enhanced email pattern
        email_patterns = [
            [{"LIKE_EMAIL": True}],
            [{"TEXT": {"REGEX": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}}]
        ]
        for i, pattern in enumerate(email_patterns):
            self.matcher.add(f"EMAIL_{i}", [pattern])
        
        # Enhanced phone patterns
        phone_patterns = [
            [{"TEXT": {"REGEX": r"(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"}}],
            [{"TEXT": {"REGEX": r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}"}}],
            [{"TEXT": {"REGEX": r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}"}}]
        ]
        for i, pattern in enumerate(phone_patterns):
            self.matcher.add(f"PHONE_{i}", [pattern])
        
        # Enhanced experience patterns
        experience_patterns = [
            [{"LOWER": {"IN": ["over", "more", "than"]}}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["years", "year", "yrs", "yr"]}}, {"LOWER": {"IN": ["of", "experience", "exp"]}}],
            [{"LIKE_NUM": True}, {"TEXT": "+"}, {"LOWER": {"IN": ["years", "year", "yrs", "yr"]}}, {"LOWER": {"IN": ["of", "experience", "exp"]}}],
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["years", "year", "yrs", "yr"]}}, {"LOWER": {"IN": ["of", "experience", "exp", "working"]}}],
            [{"LOWER": {"IN": ["experienced", "seasoned"]}}, {"LOWER": {"IN": ["professional", "developer", "engineer"]}}]
        ]
        for i, pattern in enumerate(experience_patterns):
            self.matcher.add(f"EXPERIENCE_{i}", [pattern])
        
        # Salary patterns
        salary_patterns = [
            [{"TEXT": {"REGEX": r"\$\d+[,\d]*"}},
             {"LOWER": {"IN": ["k", "thousand", "per", "annually", "yearly"]}, "OP": "?"}],
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["k", "thousand"]}}, 
             {"LOWER": {"IN": ["salary", "compensation", "per", "year"]}, "OP": "?"}]
        ]
        for i, pattern in enumerate(salary_patterns):
            self.matcher.add(f"SALARY_{i}", [pattern])
    
    def _setup_enhanced_skills_matcher(self):
        """Setup enhanced phrase matcher for skills with categories."""
        if not self.phrase_matcher or not self.nlp:
            return
        
        try:
            # Add all skills to phrase matcher with category labels
            for category, skills in self.category_skills.items():
                if skills:
                    skill_docs = []
                    for skill in skills:
                        # Add main skill
                        skill_docs.append(self.nlp(skill.lower()))
                        
                        # Add synonyms if available
                        if skill in self.skills_db.get("skill_synonyms", {}):
                            for synonym in self.skills_db["skill_synonyms"][skill]:
                                skill_docs.append(self.nlp(synonym.lower()))
                    
                    if skill_docs:
                        self.phrase_matcher.add(category.upper().replace(" ", "_"), skill_docs)
            
            logger.info(f"Setup phrase matcher with {len(self.category_skills)} skill categories")
                
        except Exception as e:
            logger.error(f"Failed to setup enhanced skills matcher: {str(e)}")
    
    def _build_skill_patterns(self):
        """Build skill patterns for enhanced matching."""
        # This method can be expanded for more complex pattern building
        pass
    
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
        Extract skills from text with enhanced categorization.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict containing categorized skills with subcategories
        """
        if not self.nlp or not text:
            return {"technical_skills": [], "soft_skills": [], "certifications": []}
        
        try:
            def _extract():
                doc = self.nlp(text.lower())
                
                # Initialize skill categories
                skills = {
                    "technical_skills": [],
                    "soft_skills": [],
                    "certifications": [],
                    "technical_categories": defaultdict(list),
                    "soft_categories": defaultdict(list),
                    "certification_categories": defaultdict(list)
                }
                
                found_skills = set()
                
                # Use phrase matcher to find skills
                matches = self.phrase_matcher(doc)
                for match_id, start, end in matches:
                    label = self.nlp.vocab.strings[match_id]
                    span = doc[start:end]
                    skill_text = span.text.title()
                    
                    # Normalize skill using synonym map
                    normalized_skill = self.synonym_map.get(span.text.lower(), skill_text)
                    
                    if normalized_skill not in found_skills:
                        found_skills.add(normalized_skill)
                        
                        # Categorize skills
                        if label.startswith("TECHNICAL_"):
                            category = label.replace("TECHNICAL_", "").replace("_", " ").title()
                            skills["technical_skills"].append(normalized_skill)
                            skills["technical_categories"][category].append(normalized_skill)
                        elif label.startswith("SOFT_"):
                            category = label.replace("SOFT_", "").replace("_", " ").title()
                            skills["soft_skills"].append(normalized_skill)
                            skills["soft_categories"][category].append(normalized_skill)
                        elif label.startswith("CERT_"):
                            category = label.replace("CERT_", "").replace("_", " ").title()
                            skills["certifications"].append(normalized_skill)
                            skills["certification_categories"][category].append(normalized_skill)
                
                # Convert defaultdicts to regular dicts
                skills["technical_categories"] = dict(skills["technical_categories"])
                skills["soft_categories"] = dict(skills["soft_categories"])
                skills["certification_categories"] = dict(skills["certification_categories"])
                
                return skills
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, _extract)
            
        except Exception as e:
            logger.error(f"Enhanced skills extraction failed: {str(e)}")
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
    
    async def extract_experience_detailed(self, text: str) -> Dict[str, Any]:
        """
        Extract detailed experience information from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict containing detailed experience information
        """
        if not self.nlp or not text:
            return {"total_years": 0, "details": [], "job_titles": [], "companies": []}
        
        try:
            def _extract():
                doc = self.nlp(text)
                experience_info = {
                    "total_years": 0,
                    "details": [],
                    "job_titles": [],
                    "companies": [],
                    "salary_mentions": []
                }
                
                # Extract years of experience
                matches = self.matcher(doc)
                years_found = []
                
                for match_id, start, end in matches:
                    label = self.nlp.vocab.strings[match_id]
                    span = doc[start:end]
                    
                    if label.startswith("EXPERIENCE"):
                        # Extract numbers from the span
                        numbers = [token.text for token in span if token.like_num]
                        if numbers:
                            try:
                                years = int(numbers[0])
                                years_found.append(years)
                                experience_info["details"].append({
                                    "text": span.text,
                                    "years": years,
                                    "type": "experience"
                                })
                            except ValueError:
                                pass
                    
                    elif label.startswith("SALARY"):
                        experience_info["salary_mentions"].append(span.text)
                
                # Extract job titles and companies using NER with cleanup
                raw_companies = []
                raw_job_titles = []
                
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        raw_companies.append(ent.text)
                    elif ent.label_ == "PERSON" and any(title_word in ent.text.lower() 
                                                       for title_word in ["engineer", "developer", "manager", "analyst", "specialist"]):
                        raw_job_titles.append(ent.text)
                
                # Use the maximum years found
                if years_found:
                    experience_info["total_years"] = max(years_found)
                
                # Apply entity cleanup
                experience_info["companies"] = self._clean_entities(raw_companies)
                experience_info["job_titles"] = self._clean_entities(raw_job_titles)
                
                return experience_info
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, _extract)
            
        except Exception as e:
            logger.error(f"Detailed experience extraction failed: {str(e)}")
            return {"total_years": 0, "details": [], "job_titles": [], "companies": []}
    
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
                keyword_counts = Counter(keywords)
                return [word for word, count in keyword_counts.most_common(top_n)]
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, _extract)
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return []
    
    async def extract_basic_info(self, text: str) -> Dict[str, Any]:
        """
        Extract basic information from resume text with entity cleanup.
        
        Args:
            text: Resume text to analyze
            
        Returns:
            Dict containing cleaned basic information
        """
        if not self.nlp or not text:
            return {"emails": [], "phones": [], "names": [], "organizations": []}
        
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
                raw_names = []
                raw_organizations = []
                
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        raw_names.append(ent.text)
                    elif ent.label_ == "ORG":
                        raw_organizations.append(ent.text)
                
                # Extract emails and phones using matcher
                matches = self.matcher(doc)
                for match_id, start, end in matches:
                    label = self.nlp.vocab.strings[match_id]
                    span = doc[start:end]
                    
                    if label.startswith("EMAIL"):
                        info["emails"].append(span.text)
                    elif label.startswith("PHONE"):
                        info["phones"].append(span.text)
                
                # Apply entity cleanup
                info["names"] = self._clean_entities(raw_names)
                info["organizations"] = self._clean_entities(raw_organizations)
                
                # Remove duplicates from emails and phones
                info["emails"] = list(set(info["emails"]))
                info["phones"] = list(set(info["phones"]))
                
                return info
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, _extract)
            
        except Exception as e:
            logger.error(f"Basic info extraction failed: {str(e)}")
            return {"emails": [], "phones": [], "names": [], "organizations": []}
    
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
            if isinstance(skills_data[skill_type], list):
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
    
    async def determine_job_level(self, text: str) -> str:
        """
        Enhanced job level detection using regex patterns and keyword heuristics.
        
        Args:
            text: Job description or resume text
            
        Returns:
            str: Job level ('entry', 'mid', 'senior', 'lead', 'exec', 'unknown')
        """
        if not text:
            return "unknown"
        
        try:
            text_lower = text.lower()
            
            # Count matches for each level
            scores = {level: 0 for level in _JOB_LEVEL_MAP}
            
            for level, patterns in _JOB_LEVEL_MAP.items():
                for pattern in patterns:
                    # Use regex search for patterns with word boundaries
                    if re.search(pattern, text_lower):
                        scores[level] += 1
                        logger.debug(f"Found {level} indicator: {pattern}")
            
            # Find the level with the highest score
            if max(scores.values()) > 0:
                best_level = max(scores, key=scores.get)
                logger.info(f"Determined job level: {best_level} (score: {scores[best_level]})")
                return best_level
            
            # Fallback: check for years of experience
            experience_patterns = [
                (r"0[-\s]*2\s*years?", "entry"),
                (r"1[-\s]*3\s*years?", "entry"),
                (r"2[-\s]*5\s*years?", "mid"),
                (r"3[-\s]*6\s*years?", "mid"),
                (r"5\+?\s*years?", "senior"),
                (r"6\+?\s*years?", "senior"),
                (r"7\+?\s*years?", "senior"),
                (r"10\+?\s*years?", "lead")
            ]
            
            for pattern, level in experience_patterns:
                if re.search(pattern, text_lower):
                    logger.info(f"Determined job level from experience: {level}")
                    return level
            
            logger.info("Could not determine job level, returning 'unknown'")
            return "unknown"
            
        except Exception as e:
            logger.error(f"Job level determination failed: {str(e)}")
            return "unknown"
    
    async def calculate_skill_match_score(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Calculate detailed skill matching score between resume and job description.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            Dict containing detailed matching analysis
        """
        try:
            # Extract skills from both texts
            resume_skills = await self.extract_skills(resume_text)
            job_skills = await self.extract_skills(job_description)
            
            # Calculate matches by category
            match_analysis = {
                "technical_match": self._calculate_category_match(
                    resume_skills["technical_skills"], 
                    job_skills["technical_skills"]
                ),
                "soft_skills_match": self._calculate_category_match(
                    resume_skills["soft_skills"], 
                    job_skills["soft_skills"]
                ),
                "certifications_match": self._calculate_category_match(
                    resume_skills["certifications"], 
                    job_skills["certifications"]
                ),
                "overall_score": 0.0,
                "detailed_matches": {},
                "missing_skills": {},
                "strength_areas": []
            }
            
            # Calculate overall score
            tech_weight = 0.6
            soft_weight = 0.3
            cert_weight = 0.1
            
            match_analysis["overall_score"] = (
                match_analysis["technical_match"]["score"] * tech_weight +
                match_analysis["soft_skills_match"]["score"] * soft_weight +
                match_analysis["certifications_match"]["score"] * cert_weight
            )
            
            # Identify strength areas
            if match_analysis["technical_match"]["score"] > 0.7:
                match_analysis["strength_areas"].append("Strong technical skills alignment")
            if match_analysis["soft_skills_match"]["score"] > 0.6:
                match_analysis["strength_areas"].append("Good soft skills match")
            if match_analysis["certifications_match"]["score"] > 0.5:
                match_analysis["strength_areas"].append("Relevant certifications")
            
            return match_analysis
            
        except Exception as e:
            logger.error(f"Skill match score calculation failed: {str(e)}")
            return {"overall_score": 0.0, "error": str(e)}
    
    def _calculate_category_match(self, resume_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
        """Calculate matching score for a specific skill category."""
        if not job_skills:
            return {"score": 1.0, "matched": [], "missing": []}
        
        resume_set = set(skill.lower() for skill in resume_skills)
        job_set = set(skill.lower() for skill in job_skills)
        
        matched = list(resume_set.intersection(job_set))
        missing = list(job_set - resume_set)
        
        score = len(matched) / len(job_set) if job_set else 0.0
        
        return {
            "score": score,
            "matched": [skill.title() for skill in matched],
            "missing": [skill.title() for skill in missing],
            "total_required": len(job_skills),
            "total_matched": len(matched)
        }
    
    async def extract_industry_context(self, text: str) -> Dict[str, Any]:
        """
        Extract industry context and domain expertise from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict containing industry information
        """
        try:
            industries = self.skills_db.get("industries", [])
            found_industries = []
            
            text_lower = text.lower()
            for industry in industries:
                if industry.lower() in text_lower:
                    found_industries.append(industry)
            
            return {
                "industries": found_industries,
                "domain_expertise": len(found_industries) > 0,
                "primary_industry": found_industries[0] if found_industries else None
            }
            
        except Exception as e:
            logger.error(f"Industry context extraction failed: {str(e)}")
            return {"industries": [], "domain_expertise": False}
    
    async def find_matching_skills(self, resume_text: str, job_description: str) -> List[str]:
        """
        Find skills that match between resume and job description.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            List of matching skills
        """
        try:
            # Extract skills from both texts
            resume_skills = await self.extract_skills(resume_text)
            job_skills = await self.extract_skills(job_description)
            
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
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            List of missing skills
        """
        try:
            # Extract skills from both texts
            resume_skills = await self.extract_skills(resume_text)
            job_skills = await self.extract_skills(job_description)
            
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
    
    async def health_check(self) -> bool:
        """
        Enhanced health check for NLP processor service.
        
        Returns:
            bool: True if service is healthy
        """
        try:
            health_status = (
                self.nlp is not None and 
                SPACY_AVAILABLE and 
                len(self.category_skills) > 0 and
                self.phrase_matcher is not None
            )
            
            if health_status:
                logger.info("NLP Processor health check: HEALTHY")
            else:
                logger.warning("NLP Processor health check: UNHEALTHY")
            
            return health_status
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False


# For backward compatibility, create an alias
NLPProcessor = EnhancedNLPProcessor
