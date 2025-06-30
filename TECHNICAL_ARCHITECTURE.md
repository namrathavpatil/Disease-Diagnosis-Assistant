# Technical Architecture Documentation

## System Overview

The Medical RAG System is designed to provide accurate, evidence-based medical information by combining multiple authoritative sources through advanced retrieval and generation techniques.

## 🏗 Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   FastAPI       │    │   RAG Engine    │
│   (HTML/JS)     │◄──►│   Backend       │◄──►│   (Core Logic)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Knowledge     │    │   FAISS Index   │
│   • PubMed      │    │   Graph         │    │   (Semantic     │
│   • Orphanet    │    │   Builder       │    │   Search)       │
│   • FDA         │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Core Components

### 1. Data Collection Layer
**Purpose**: Gather medical information from authoritative sources

#### PubMed Collector
- **Function**: Fetches scientific articles and abstracts
- **API**: NCBI E-utilities
- **Output**: Structured article data with metadata
- **Features**: Full-text extraction, batching, keyword extraction

#### Orphanet Collector
- **Function**: Retrieves rare disease information
- **API**: Orphanet REST API
- **Output**: Disease profiles, symptoms, treatments
- **Features**: Entity extraction, relationship mapping

#### FDA Collector
- **Function**: Collects drug safety and treatment data
- **API**: FDA Open Data API
- **Output**: Drug information, safety alerts, guidelines
- **Features**: Search term extraction, document processing

### 2. Knowledge Graph Builder
**Purpose**: Create structured knowledge representation

#### Entity Extraction
- **Method**: Rule-based + LLM-based extraction
- **Output**: Medical entities (diseases, drugs, symptoms)
- **Storage**: Graph nodes with properties

#### Relationship Mapping
- **Method**: Co-occurrence analysis + semantic similarity
- **Output**: Entity relationships (treats, causes, associated_with)
- **Storage**: Graph edges with weights

### 3. RAG-Ready Graph Builder
**Purpose**: Prepare data for efficient retrieval

#### Text Chunking
- **Method**: Sliding window with overlap
- **Size**: Variable (500-1000 tokens)
- **Overlap**: 100-200 tokens
- **Output**: Semantic chunks with metadata

#### Embedding Generation
- **Model**: all-MiniLM-L6-v2
- **Dimensions**: 384
- **Method**: Sentence transformers
- **Output**: Vector representations

#### FAISS Index
- **Type**: IndexFlatIP (Inner Product)
- **Purpose**: Fast similarity search
- **Features**: Cosine similarity, top-k retrieval

### 4. RAG Engine
**Purpose**: Generate accurate answers from retrieved context

#### Retrieval System
- **Method**: Semantic search + keyword fallback
- **Top-k**: 5 chunks per query
- **Scoring**: Similarity scores + confidence metrics

#### Generation System
- **Model**: Mixtral-8x7B-Instruct-v0.1
- **Provider**: Together AI
- **Features**: Context-aware prompting, source citation

#### Confidence Assessment
- **Method**: LLM self-assessment
- **Scale**: 0-1 confidence score
- **Fallback**: Follow-up question generation

### 5. Web Interface
**Purpose**: User-friendly interaction with the system

#### Frontend
- **Technology**: HTML, JavaScript, CSS
- **Features**: Real-time querying, result display, source links

#### Backend API
- **Framework**: FastAPI
- **Endpoints**: Query, build, stats, health
- **Features**: Async processing, error handling

## 📊 Data Flow

### 1. Query Processing
```
User Query → FastAPI → RAG Engine → Retrieval → Generation → Response
```

### 2. Data Collection
```
Search Terms → Collectors → Raw Data → Processing → Structured Data
```

### 3. Knowledge Building
```
Structured Data → Entity Extraction → Graph Building → RAG Preparation
```

### 4. RAG Preparation
```
Text Data → Chunking → Embedding → FAISS Index → Ready for Retrieval
```

## 🗄 Data Storage

### File-based Storage
- **`rag_ready_graph.json`**: Complete RAG structure
- **`rag_chunks.json`**: All text chunks with metadata
- **`rag_entity_nodes.json`**: Entity information
- **`knowledge_graph.json`**: Graph structure

### In-Memory Storage
- **FAISS Index**: Vector embeddings for fast search
- **Chunk Cache**: Frequently accessed chunks
- **Entity Cache**: Entity relationships

## 🔄 Processing Pipeline

### Phase 1: Data Collection
1. **Query Analysis**: Extract search terms and entities
2. **Source Selection**: Choose relevant data sources
3. **Data Fetching**: Retrieve information from APIs
4. **Data Cleaning**: Remove duplicates, format data

### Phase 2: Knowledge Building
1. **Entity Extraction**: Identify medical entities
2. **Relationship Mapping**: Connect related entities
3. **Graph Construction**: Build knowledge graph
4. **Validation**: Verify data quality

### Phase 3: RAG Preparation
1. **Text Chunking**: Split documents into chunks
2. **Embedding Generation**: Create vector representations
3. **Index Building**: Construct FAISS index
4. **Metadata Creation**: Add retrieval metadata

### Phase 4: Query Processing
1. **Query Encoding**: Convert query to vector
2. **Semantic Search**: Find relevant chunks
3. **Context Assembly**: Prepare context for LLM
4. **Answer Generation**: Generate response with citations

## 🛡 Security & Performance

### Security Measures
- **API Rate Limiting**: Prevent abuse
- **Input Validation**: Sanitize user queries
- **Error Handling**: Graceful failure modes
- **Logging**: Audit trail for queries

### Performance Optimizations
- **Chunk Deduplication**: Reduce storage and processing
- **Embedding Caching**: Avoid recomputation
- **Async Processing**: Non-blocking operations
- **Memory Management**: Efficient data structures

## 🔍 Monitoring & Analytics

### System Metrics
- **Response Time**: Query processing duration
- **Accuracy**: Answer quality assessment
- **Throughput**: Queries per second
- **Memory Usage**: Resource consumption

### User Analytics
- **Query Patterns**: Common question types
- **Source Usage**: Which sources are most relevant
- **Confidence Distribution**: Answer confidence levels
- **Follow-up Questions**: User engagement patterns

## 🚀 Deployment Architecture

### Development Environment
- **Local Server**: uvicorn on localhost:8000
- **File Storage**: Local JSON files
- **Memory**: In-memory FAISS index

### Production Considerations
- **Containerization**: Docker for consistency
- **Database**: PostgreSQL for persistent storage
- **Caching**: Redis for performance
- **Load Balancing**: Multiple instances
- **Monitoring**: Prometheus + Grafana

## 📈 Scalability Considerations

### Horizontal Scaling
- **Stateless Design**: Easy instance replication
- **Database Sharding**: Distribute data across nodes
- **Load Balancing**: Route requests efficiently

### Vertical Scaling
- **Memory Optimization**: Efficient data structures
- **CPU Optimization**: Parallel processing
- **Storage Optimization**: Compression and indexing

---

**Version**: 1.0  
**Last Updated**: December 2024  
**Maintained By**: Development Team 