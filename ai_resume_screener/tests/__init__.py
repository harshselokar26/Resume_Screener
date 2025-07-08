"""
Tests Package for AI Resume Screener

This package contains all test modules including unit tests, integration tests,
and test fixtures for comprehensive application testing.
"""

# Test configuration
TEST_DATABASE_URL = "sqlite:///test_resume_screener.db"
TEST_UPLOAD_DIR = "test_uploads"
TEST_DATA_DIR = "tests/fixtures"

# Test constants
SAMPLE_RESUME_TEXT = """
John Doe
Software Engineer
Email: john.doe@example.com
Phone: (555) 123-4567

EXPERIENCE
Senior Software Engineer at TechCorp (2020-Present)
- Developed web applications using React and Node.js
- Led team of 5 developers
- Implemented CI/CD pipelines

SKILLS
Programming: Python, JavaScript, React, Node.js
Databases: PostgreSQL, MongoDB
Cloud: AWS, Docker, Kubernetes
"""

SAMPLE_JOB_DESCRIPTION = """
Senior Software Engineer Position

We are looking for an experienced software engineer with:
- 5+ years of experience in web development
- Strong skills in React, Node.js, and Python
- Experience with cloud platforms (AWS preferred)
- Knowledge of databases (PostgreSQL, MongoDB)
- Leadership experience preferred

Requirements:
- Bachelor's degree in Computer Science
- Experience with agile methodologies
- Strong communication skills
"""

__all__ = [
    "TEST_DATABASE_URL",
    "TEST_UPLOAD_DIR", 
    "TEST_DATA_DIR",
    "SAMPLE_RESUME_TEXT",
    "SAMPLE_JOB_DESCRIPTION"
]
