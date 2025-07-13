# Medical RAG System Setup Guide

## Prerequisites
- Python 3.10 or higher
- Virtual environment (recommended)

## Installation Steps

### 1. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install spaCy language models
```bash
python setup_spacy.py
```

Or manually:
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_sci_sm
```

### 4. Set up environment variables
Create a `.env` file with the following variables:
```env
# API Keys (optional but recommended for full functionality)
TOGETHER_API_KEY=your_together_api_key
ORPHANET_API_KEY=your_orphanet_api_key
PUBMED_API_KEY=your_pubmed_api_key

# Model Configuration
MODEL_NAME=deepseek-chat
TEMPERATURE=0.1
EMBEDDING_MODEL=dmis-lab/biobert-base-cased-v1.2

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# PubMed Configuration
PUBMED_EMAIL=your_email@example.com
```

### 5. Run the application
```bash
python -m app.main
```

## Troubleshooting

### Common Issues

1. **spaCy model not found**
   - Run: `python -m spacy download en_core_web_sm`
   - Run: `python -m spacy download en_core_sci_sm`

2. **Version conflicts**
   - Delete `.venv` and recreate it
   - Install requirements in order: `pip install -r requirements.txt`

3. **Port already in use**
   - Kill existing process: `lsof -ti:8000 | xargs kill -9`
   - Or change port in `.env`: `API_PORT=8001`

4. **Memory issues**
   - Use `faiss-cpu` instead of `faiss-gpu`
   - Reduce batch sizes in configuration

### Dependencies Fixed

The updated `requirements.txt` includes:
- ✅ Fixed version conflicts (transformers, numpy, pydantic)
- ✅ Added missing dependencies (faiss-cpu, tiktoken, scikit-learn)
- ✅ Updated to compatible versions
- ✅ Added proper version constraints

### Key Changes Made

1. **requirements.txt**: Updated with proper versions and missing dependencies
2. **entity_extractor.py**: Fixed spaCy model initialization
3. **setup_spacy.py**: Added script to install required spaCy models

## Accessing the Application

Once running, access:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Features Available

- Medical question answering using RAG
- Entity extraction from medical text
- Knowledge graph building
- Disease and drug searches
- RAG-ready graph functionality
- Interactive API documentation 