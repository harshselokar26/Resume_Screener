# 🤖 AI Resume Screener

[![Build Status](https://github.com/your-org/ai-resume-screener/workflows/CI/badge.svg)](https://github.com/your-org/ai-resume-screener/actions)
[![Coverage Status](https://codecov.io/gh/your-org/ai-resume-screener/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/ai-resume-screener)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An intelligent resume screening and job matching system powered by advanced Natural Language Processing (NLP) and Machine Learning techniques. This application helps recruiters and hiring managers efficiently evaluate resumes against job requirements with high accuracy and detailed insights.

## ✨ Features

### 🔍 **Intelligent Resume Analysis**
- **Multi-format Support**: PDF, DOC, DOCX file processing
- **Advanced Text Extraction**: High-quality text extraction from various document formats
- **Skills Detection**: Automatic identification of technical skills, soft skills, and certifications
- **Experience Analysis**: Years of experience calculation and role progression analysis
- **Education Parsing**: Degree, institution, and graduation year extraction

### 🎯 **Smart Job Matching**
- **Similarity Scoring**: Advanced TF-IDF and cosine similarity algorithms
- **Skills Matching**: Precise skill overlap analysis between resumes and job descriptions
- **Gap Analysis**: Identification of missing skills and requirements
- **Recommendation Engine**: AI-powered hiring recommendations with confidence levels
- **Batch Processing**: Process multiple resumes simultaneously

### 🚀 **Production-Ready API**
- **FastAPI Framework**: High-performance, modern Python web framework
- **Async Support**: Asynchronous processing for better performance
- **Auto Documentation**: Interactive API documentation with Swagger UI
- **Rate Limiting**: Built-in request throttling and abuse prevention
- **Health Monitoring**: Comprehensive health checks and monitoring endpoints

### 🔒 **Security & Scalability**
- **Authentication**: JWT-based authentication and API key support
- **Data Privacy**: Secure file handling with automatic cleanup
- **Containerized**: Docker and Docker Compose for easy deployment
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **CI/CD**: Automated testing, building, and deployment pipelines

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Docker & Docker Compose** (recommended)
- **Git**

### 1. Clone Repository


### 2. Environment Setup

Copy environment template
cp .env.example .env

Edit configuration (update SECRET_KEY and other settings)
nano .env

text

### 3. Quick Start with Docker

Start all services
make up

Or manually
docker-compose up -d

text

### 4. Development Setup

Run setup script
make setup

Install dependencies
make install

Download NLP models
make download-models

Start development server
make dev

text

### 5. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090

## 📖 Usage Examples

### Upload and Score Resume

import requests

Upload resume
with open("resume.pdf", "rb") as f:
response = requests.post(
"http://localhost:8000/api/upload-resume",
files={"file": f}
)
resume_data = response.json()

Score against job description
scoring_data = {
"resume_text": resume_data["extracted_text"],
"job_description": "Senior Software Engineer position requiring Python, React, and 5+ years experience...",
"include_detailed_analysis": True
}

response = requests.post(
"http://localhost:8000/api/score-resume",
json=scoring_data
)

result = response.json()
print(f"Similarity Score: {result['score_percentage']}%")
print(f"Recommendation: {result['recommendation']['decision']}")

text

### Extract Skills

curl -X POST "http://localhost:8000/api/extract-skills"
-H "Content-Type: application/x-www-form-urlencoded"
-d "resume_text=John Doe Software Engineer with Python and React experience..."

text

## 🛠️ Development

### Available Commands

Development
make dev # Start development server
make dev-docker # Start with Docker
make shell # Open container shell

Testing
make test # Run all tests
make test-unit # Run unit tests only
make test-integration # Run integration tests
make test-docker # Run tests in Docker

Code Quality
make lint # Run linting
make format # Format code
make security # Security checks
make pre-commit # Run pre-commit hooks

Docker Operations
make build # Build images
make up # Start services
make down # Stop services
make logs # View logs

Database
make db-migrate # Run migrations
make db-shell # Database shell
make backup # Create backup

Monitoring
make stats # Container stats
make health # Health check
make monitoring # Open dashboards

text

### Project Structure
ai_resume_screener/
├── app/ # Application code
│ ├── api/ # API routes and middleware
│ ├── services/ # Business logic services
│ ├── models/ # Data models and schemas
│ ├── utils/ # Utility functions
│ ├── config/ # Configuration management
│ └── core/ # Core functionality
├── tests/ # Test suite
│ ├── unit/ # Unit tests
│ ├── integration/ # Integration tests
│ └── fixtures/ # Test fixtures
├── data/ # Data files and samples
├── docs/ # Documentation
├── scripts/ # Automation scripts
└── requirements/ # Dependency management

text

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

Application
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=your-secret-key

Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

Database
DATABASE_URL=postgresql://user:pass@localhost/db

NLP
SPACY_MODEL=en_core_web_sm

text

### Docker Configuration

- **Production**: `docker-compose.yml`
- **Development**: `docker-compose.dev.yml`
- **Testing**: Use `--profile testing`

## 📊 API Documentation

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/upload-resume` | POST | Upload resume file |
| `/api/extract-skills` | POST | Extract skills from text |
| `/api/score-resume` | POST | Score resume vs job description |
| `/api/analyze-job-description` | POST | Analyze job requirements |

### Response Format

{
"similarity_score": 0.85,
"score_percentage": 85.0,
"matching_skills": ["Python", "React", "Leadership"],
"missing_skills": ["Docker", "Kubernetes"],
"recommendation": {
"decision": "Highly Recommended",
"confidence": "High",
"reasons": ["Strong technical skills", "Good experience match"]
}
}

text

## 🧪 Testing

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **Performance Tests**: Load and stress testing
- **Security Tests**: Vulnerability scanning

### Running Tests

All tests
make test

Specific test types
pytest tests/unit/ -v
pytest tests/integration/ -v

With coverage
pytest --cov=app --cov-report=html

Performance tests
locust -f tests/performance/locustfile.py

text

## 🚀 Deployment

### Production Deployment

Build production image
docker build -t ai-resume-screener:latest .

Deploy with docker-compose
docker-compose -f docker-compose.yml up -d

Or use deployment script
make deploy-prod

text

### Cloud Deployment

Supports deployment on:
- **AWS ECS/Fargate**
- **Google Cloud Run**
- **Azure Container Instances**
- **Kubernetes**

See [Deployment Guide](docs/deployment_guide.md) for detailed instructions.

## 📈 Monitoring

### Built-in Monitoring

- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Health Checks**: Service status monitoring
- **Logging**: Structured logging with rotation

### Key Metrics

- Request latency and throughput
- Error rates and status codes
- Resource utilization (CPU, memory)
- NLP processing performance
- File upload statistics

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Code Standards

- **Python**: Follow PEP 8 with Black formatting
- **Testing**: Maintain >80% code coverage
- **Documentation**: Update docs for new features
- **Security**: Run security checks before submitting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full Documentation](docs/)
- **API Reference**: [API Docs](docs/api_documentation.md)
- **Issues**: [GitHub Issues](https://github.com/your-org/ai-resume-screener/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/ai-resume-screener/discussions)

## 🙏 Acknowledgments

- **spaCy**: Advanced NLP processing
- **FastAPI**: Modern Python web framework
- **scikit-learn**: Machine learning algorithms
- **Docker**: Containerization platform

---

**Built with ❤️ by [Your Organization](https://your-org.com)**