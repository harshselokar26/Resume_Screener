"""
API Routes for AI Resume Screener

This module defines all API endpoints for resume upload, parsing,
scoring, and job description matching functionality.
"""

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
import logging

# Internal imports
from app.config.settings import settings
from app.models.schemas import (
    ResumeUploadResponse,
    JobDescriptionRequest,
    ScoringRequest,
    ScoringResponse,
    SkillsExtractionResponse,
    ErrorResponse,
    HealthResponse
)
from app.services.pdf_parser import PDFParser
from app.services.nlp_processor import NLPProcessor
from app.services.scorer import ResumeScorer
from app.services.file_handler import FileHandler
from app.api.dependencies import validate_file_upload, get_file_handler
from app.utils.exceptions import (
    FileProcessingError,
    NLPProcessingError,
    ScoringError
)
from app.utils.helpers import generate_unique_filename, cleanup_temp_files

# Setup logging
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Initialize services
pdf_parser = PDFParser()
nlp_processor = NLPProcessor()
resume_scorer = ResumeScorer()


@router.post(
    "/upload-resume",
    response_model=ResumeUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Resume File",
    description="Upload a resume file (PDF, DOC, DOCX) and extract text content"
)
async def upload_resume(
    file: UploadFile = File(..., description="Resume file to upload"),
    file_handler: FileHandler = Depends(get_file_handler)
) -> ResumeUploadResponse:
    """
    Upload and process a resume file.
    
    Args:
        file: Uploaded resume file
        file_handler: File handling service dependency
        
    Returns:
        ResumeUploadResponse: Upload result with extracted text and metadata
        
    Raises:
        HTTPException: If file processing fails
    """
    try:
        # Validate file
        await validate_file_upload(file)
        
        # Generate unique filename
        unique_filename = generate_unique_filename(file.filename)
        file_path = settings.get_upload_path(unique_filename)
        
        # Save uploaded file
        await file_handler.save_file(file, file_path)
        
        # Extract text from file
        if file.filename.lower().endswith('.pdf'):
            extracted_text = await pdf_parser.extract_text_from_pdf(file_path)
        else:
            # Handle DOC/DOCX files
            extracted_text = await pdf_parser.extract_text_from_doc(file_path)
        
        # Get file metadata
        file_stats = os.stat(file_path)
        
        # Process with NLP for basic info
        basic_info = await nlp_processor.extract_basic_info(extracted_text)
        
        logger.info(f"Successfully processed resume: {file.filename}")
        
        return ResumeUploadResponse(
            file_id=str(uuid.uuid4()),
            original_filename=file.filename,
            file_size=file_stats.st_size,
            upload_timestamp=datetime.utcnow(),
            extracted_text=extracted_text,
            text_length=len(extracted_text),
            basic_info=basic_info,
            status="success",
            message="Resume uploaded and processed successfully"
        )
        
    except FileProcessingError as e:
        logger.error(f"File processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File processing failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in upload_resume: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during file upload"
        )
    finally:
        # Cleanup temporary files
        if 'file_path' in locals():
            cleanup_temp_files([file_path])


@router.post(
    "/extract-skills",
    response_model=SkillsExtractionResponse,
    summary="Extract Skills from Resume",
    description="Extract skills, experience, and keywords from resume text"
)
async def extract_skills(
    resume_text: str = Form(..., description="Resume text content"),
    job_description: Optional[str] = Form(None, description="Optional job description for context")
) -> SkillsExtractionResponse:
    """
    Extract skills and relevant information from resume text.
    
    Args:
        resume_text: Text content of the resume
        job_description: Optional job description for context-aware extraction
        
    Returns:
        SkillsExtractionResponse: Extracted skills and information
    """
    try:
        # Extract skills using NLP processor
        skills_data = await nlp_processor.extract_skills(resume_text)
        
        # Extract experience information
        experience_data = await nlp_processor.extract_experience(resume_text)
        
        # Extract education information
        education_data = await nlp_processor.extract_education(resume_text)
        
        # Extract keywords
        keywords = await nlp_processor.extract_keywords(resume_text)
        
        # If job description provided, find relevant skills
        relevant_skills = []
        if job_description:
            relevant_skills = await nlp_processor.find_relevant_skills(
                resume_text, job_description
            )
        
        logger.info("Successfully extracted skills and information from resume")
        
        return SkillsExtractionResponse(
            technical_skills=skills_data.get("technical_skills", []),
            soft_skills=skills_data.get("soft_skills", []),
            experience_years=experience_data.get("total_years", 0),
            experience_details=experience_data.get("details", []),
            education=education_data,
            keywords=keywords,
            relevant_skills=relevant_skills,
            extraction_timestamp=datetime.utcnow(),
            status="success"
        )
        
    except NLPProcessingError as e:
        logger.error(f"NLP processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Skills extraction failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in extract_skills: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during skills extraction"
        )


@router.post(
    "/score-resume",
    response_model=ScoringResponse,
    summary="Score Resume Against Job Description",
    description="Calculate similarity score between resume and job description"
)
async def score_resume(
    request: ScoringRequest
) -> ScoringResponse:
    """
    Score a resume against a job description using NLP similarity.
    
    Args:
        request: Scoring request containing resume text and job description
        
    Returns:
        ScoringResponse: Detailed scoring results and analysis
    """
    try:
        # Validate input
        if not request.resume_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resume text cannot be empty"
            )
        
        if not request.job_description.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job description cannot be empty"
            )
        
        # Calculate similarity score
        similarity_score = await resume_scorer.calculate_similarity(
            request.resume_text,
            request.job_description
        )
        
        # Extract matching skills
        matching_skills = await resume_scorer.find_matching_skills(
            request.resume_text,
            request.job_description
        )
        
        # Find missing skills
        missing_skills = await resume_scorer.find_missing_skills(
            request.resume_text,
            request.job_description
        )
        
        # Get detailed analysis
        analysis = await resume_scorer.get_detailed_analysis(
            request.resume_text,
            request.job_description,
            similarity_score
        )
        
        # Determine recommendation
        recommendation = await resume_scorer.get_recommendation(
            similarity_score,
            matching_skills,
            missing_skills
        )
        
        logger.info(f"Successfully scored resume with similarity: {similarity_score:.3f}")
        
        return ScoringResponse(
            similarity_score=similarity_score,
            score_percentage=round(similarity_score * 100, 2),
            matching_skills=matching_skills,
            missing_skills=missing_skills,
            total_skills_found=len(matching_skills),
            total_skills_required=len(matching_skills) + len(missing_skills),
            analysis=analysis,
            recommendation=recommendation,
            scoring_timestamp=datetime.utcnow(),
            status="success"
        )
        
    except ScoringError as e:
        logger.error(f"Scoring error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Resume scoring failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in score_resume: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during resume scoring"
        )


@router.post(
    "/analyze-job-description",
    summary="Analyze Job Description",
    description="Extract requirements and skills from job description"
)
async def analyze_job_description(
    request: JobDescriptionRequest
) -> Dict[str, Any]:
    """
    Analyze job description to extract requirements and skills.
    
    Args:
        request: Job description analysis request
        
    Returns:
        Dict containing analyzed job description data
    """
    try:
        # Extract required skills
        required_skills = await nlp_processor.extract_required_skills(
            request.job_description
        )
        
        # Extract job requirements
        requirements = await nlp_processor.extract_requirements(
            request.job_description
        )
        
        # Extract job level/seniority
        job_level = await nlp_processor.determine_job_level(
            request.job_description
        )
        
        # Extract keywords
        keywords = await nlp_processor.extract_keywords(
            request.job_description
        )
        
        logger.info("Successfully analyzed job description")
        
        return {
            "required_skills": required_skills,
            "requirements": requirements,
            "job_level": job_level,
            "keywords": keywords,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing job description: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during job description analysis"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API Health Check",
    description="Check API health and service status"
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for monitoring API status.
    
    Returns:
        HealthResponse: Current API health status
    """
    try:
        # Check services
        services_status = {
            "pdf_parser": await pdf_parser.health_check(),
            "nlp_processor": await nlp_processor.health_check(),
            "resume_scorer": await resume_scorer.health_check()
        }
        
        # Check file system
        upload_dir_writable = os.access(settings.UPLOAD_DIR, os.W_OK)
        
        # Overall health
        all_healthy = all(services_status.values()) and upload_dir_writable
        
        return HealthResponse(
            status="healthy" if all_healthy else "unhealthy",
            timestamp=datetime.utcnow(),
            services=services_status,
            upload_dir_writable=upload_dir_writable,
            version=settings.VERSION
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            services={},
            upload_dir_writable=False,
            version=settings.VERSION,
            error=str(e)
        )


@router.delete(
    "/cleanup",
    summary="Cleanup Temporary Files",
    description="Clean up temporary uploaded files"
)
async def cleanup_files() -> Dict[str, Any]:
    """
    Cleanup temporary files and free up storage space.
    
    Returns:
        Dict containing cleanup results
    """
    try:
        # Get list of files to cleanup
        upload_dir = settings.UPLOAD_DIR
        if not os.path.exists(upload_dir):
            return {
                "status": "success",
                "message": "No files to cleanup",
                "files_removed": 0
            }
        
        # Remove old files (older than 1 hour)
        removed_count = 0
        current_time = datetime.utcnow().timestamp()
        
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getctime(file_path)
                if file_age > 3600:  # 1 hour
                    os.remove(file_path)
                    removed_count += 1
        
        logger.info(f"Cleanup completed: {removed_count} files removed")
        
        return {
            "status": "success",
            "message": f"Cleanup completed successfully",
            "files_removed": removed_count,
            "cleanup_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cleanup failed: {str(e)}"
        )
