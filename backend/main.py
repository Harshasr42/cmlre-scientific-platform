"""
CMLRE Scientific Platform Backend
Advanced Marine Data Processing API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import base64

app = FastAPI(
    title="CMLRE Scientific Platform API",
    description="Advanced Marine Data Processing API for CMLRE",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class TaxonomicClassification(BaseModel):
    species_input: str
    confidence_threshold: float = 0.8

class OtolithAnalysis(BaseModel):
    image_data: str  # Base64 encoded image
    analysis_type: str = "2D"
    parameters: Dict[str, Any] = {}

class EDNAResult(BaseModel):
    sequence: str
    species: str
    confidence: float
    database: str

class CorrelationAnalysis(BaseModel):
    parameters: List[str]
    correlation_type: str = "pearson"

# In-memory storage for demo
datasets = {}
analyses = {}
research_projects = []

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "CMLRE Scientific Platform API",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "data_processing": "operational",
            "analytics": "operational",
            "taxonomy": "operational",
            "otolith": "operational",
            "edna": "operational"
        },
        "timestamp": datetime.now().isoformat()
    }

# Data Integration Endpoints
@app.post("/api/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_type: str = Form(...),
    metadata: str = Form("{}")
):
    """Upload and process marine datasets"""
    try:
        # Read file content
        content = await file.read()
        
        # Process based on file type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Store dataset
        dataset_id = f"dataset_{len(datasets) + 1}"
        datasets[dataset_id] = {
            "id": dataset_id,
            "name": file.filename,
            "type": dataset_type,
            "data": df.to_dict(),
            "metadata": json.loads(metadata),
            "upload_time": datetime.now().isoformat(),
            "rows": len(df),
            "columns": list(df.columns)
        }
        
        return {
            "message": "Dataset uploaded successfully",
            "dataset_id": dataset_id,
            "rows": len(df),
            "columns": list(df.columns)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")

@app.get("/api/datasets")
async def get_datasets():
    """Get all uploaded datasets"""
    return {
        "datasets": list(datasets.values()),
        "total": len(datasets)
    }

@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get specific dataset"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return datasets[dataset_id]

# Taxonomic Classification Endpoints
@app.post("/api/taxonomy/classify")
async def classify_species(classification: TaxonomicClassification):
    """Perform taxonomic classification"""
    try:
        # Mock taxonomic classification
        results = []
        
        # Simple keyword-based classification for demo
        species_input = classification.species_input.lower()
        
        if "tuna" in species_input or "thunnus" in species_input:
            results.append({
                "species": "Thunnus albacares",
                "common_name": "Yellowfin Tuna",
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": "Actinopterygii",
                "order": "Perciformes",
                "family": "Scombridae",
                "genus": "Thunnus",
                "confidence": 0.95
            })
        
        if "mackerel" in species_input or "scomberomorus" in species_input:
            results.append({
                "species": "Scomberomorus commerson",
                "common_name": "Narrow-barred Spanish Mackerel",
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": "Actinopterygii",
                "order": "Perciformes",
                "family": "Scombridae",
                "genus": "Scomberomorus",
                "confidence": 0.87
            })
        
        if "shark" in species_input:
            results.append({
                "species": "Carcharhinus limbatus",
                "common_name": "Blacktip Shark",
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": "Chondrichthyes",
                "order": "Carcharhiniformes",
                "family": "Carcharhinidae",
                "genus": "Carcharhinus",
                "confidence": 0.82
            })
        
        # Filter by confidence threshold
        filtered_results = [r for r in results if r["confidence"] >= classification.confidence_threshold]
        
        return {
            "input": classification.species_input,
            "results": filtered_results,
            "total_matches": len(filtered_results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

# Otolith Analysis Endpoints
@app.post("/api/otolith/analyze")
async def analyze_otolith(analysis: OtolithAnalysis):
    """Perform otolith morphology analysis"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(analysis.image_data)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Perform edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate morphometric parameters
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Calculate shape parameters
            aspect_ratio = 1.0  # Simplified calculation
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            roundness = 4 * area / (np.pi * (perimeter / np.pi) ** 2) if perimeter > 0 else 0
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Solidity (area / convex hull area)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            results = {
                "area": round(area, 2),
                "perimeter": round(perimeter, 2),
                "aspect_ratio": round(aspect_ratio, 3),
                "circularity": round(circularity, 3),
                "roundness": round(roundness, 3),
                "solidity": round(solidity, 3),
                "bounding_box": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                },
                "analysis_type": analysis.analysis_type,
                "timestamp": datetime.now().isoformat()
            }
            
            return {
                "message": "Otolith analysis completed successfully",
                "results": results
            }
        else:
            raise HTTPException(status_code=400, detail="No contours found in image")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Otolith analysis error: {str(e)}")

# eDNA Analysis Endpoints
@app.post("/api/edna/analyze")
async def analyze_edna(
    sequence: str = Form(...),
    database: str = Form("NCBI"),
    min_confidence: float = Form(0.8)
):
    """Perform eDNA sequence analysis"""
    try:
        # Mock eDNA analysis
        results = []
        
        # Simple sequence-based matching for demo
        sequence = sequence.upper()
        
        if "ATCG" in sequence or len(sequence) > 100:
            # Mock species matches
            if "TUNNA" in sequence or "THUNNUS" in sequence:
                results.append({
                    "species": "Thunnus albacares",
                    "common_name": "Yellowfin Tuna",
                    "confidence": 0.92,
                    "sequence_length": len(sequence),
                    "database": database,
                    "match_region": "COI gene"
                })
            
            if "MACKEREL" in sequence or "SCOMBER" in sequence:
                results.append({
                    "species": "Scomberomorus commerson",
                    "common_name": "Narrow-barred Spanish Mackerel",
                    "confidence": 0.85,
                    "sequence_length": len(sequence),
                    "database": database,
                    "match_region": "COI gene"
                })
            
            if "SHARK" in sequence or "CARCHARHINUS" in sequence:
                results.append({
                    "species": "Carcharhinus limbatus",
                    "common_name": "Blacktip Shark",
                    "confidence": 0.78,
                    "sequence_length": len(sequence),
                    "database": database,
                    "match_region": "COI gene"
                })
        
        # Filter by confidence threshold
        filtered_results = [r for r in results if r["confidence"] >= min_confidence]
        
        return {
            "input_sequence_length": len(sequence),
            "database": database,
            "results": filtered_results,
            "total_matches": len(filtered_results),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"eDNA analysis error: {str(e)}")

# Analytics Endpoints
@app.post("/api/analytics/correlation")
async def correlation_analysis(analysis: CorrelationAnalysis):
    """Perform cross-domain correlation analysis"""
    try:
        # Generate mock correlation data
        np.random.seed(42)
        n_samples = 100
        
        # Create mock oceanographic and biological data
        data = {
            'sst': np.random.normal(28, 2, n_samples),
            'salinity': np.random.normal(35, 1, n_samples),
            'oxygen': np.random.normal(7, 1, n_samples),
            'fish_abundance': np.random.normal(50, 15, n_samples),
            'species_diversity': np.random.normal(20, 5, n_samples),
            'biomass': np.random.normal(100, 30, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate correlation matrix
        if analysis.correlation_type == "pearson":
            corr_matrix = df.corr()
        elif analysis.correlation_type == "spearman":
            corr_matrix = df.corr(method='spearman')
        else:
            corr_matrix = df.corr()
        
        # Convert to JSON serializable format
        corr_dict = corr_matrix.to_dict()
        
        return {
            "correlation_type": analysis.correlation_type,
            "correlation_matrix": corr_dict,
            "parameters": analysis.parameters,
            "sample_size": n_samples,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correlation analysis error: {str(e)}")

@app.post("/api/analytics/pca")
async def pca_analysis(parameters: List[str] = Form(...)):
    """Perform Principal Component Analysis"""
    try:
        # Generate mock data for PCA
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'sst': np.random.normal(28, 2, n_samples),
            'salinity': np.random.normal(35, 1, n_samples),
            'oxygen': np.random.normal(7, 1, n_samples),
            'fish_abundance': np.random.normal(50, 15, n_samples),
            'species_diversity': np.random.normal(20, 5, n_samples),
            'biomass': np.random.normal(100, 30, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        return {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "components": pca.components_.tolist(),
            "pca_result": pca_result.tolist(),
            "parameters": parameters,
            "sample_size": n_samples,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PCA analysis error: {str(e)}")

# Research Project Endpoints
@app.post("/api/projects")
async def create_project(
    name: str = Form(...),
    description: str = Form(...),
    project_type: str = Form(...),
    created_by: str = Form(...)
):
    """Create a new research project"""
    try:
        project = {
            "id": f"project_{len(research_projects) + 1}",
            "name": name,
            "description": description,
            "type": project_type,
            "created_by": created_by,
            "status": "Active",
            "created_at": datetime.now().isoformat(),
            "datasets": [],
            "analyses": []
        }
        
        research_projects.append(project)
        
        return {
            "message": "Project created successfully",
            "project": project
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Project creation error: {str(e)}")

@app.get("/api/projects")
async def get_projects():
    """Get all research projects"""
    return {
        "projects": research_projects,
        "total": len(research_projects)
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
