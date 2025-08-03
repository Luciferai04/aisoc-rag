# Live Camera Enhanced Translator with Advanced Translation Engine

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Flask](https://img.shields.io/badge/Flask-2.0+-red.svg)](https://flask.palletsprojects.com)
[![Gradio](https://img.shields.io/badge/Gradio-5.35+-orange.svg)](https://gradio.app)
[![WhisperLive](https://img.shields.io/badge/WhisperLive-Real--Time-green.svg)](https://github.com/collabora/WhisperLive)
[![Advanced AI](https://img.shields.io/badge/Advanced_AI-Translation_Engine-brightgreen.svg)](#advanced-translation-engine)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Supported-326ce5.svg)](https://kubernetes.io)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-green.svg)](https://github.com/features/actions)
[![Load Testing](https://img.shields.io/badge/Load_Testing-Locust-red.svg)](https://locust.io)
[![HTTPS](https://img.shields.io/badge/HTTPS-Enabled-green.svg)](https://letsencrypt.org)

## Overview

This project is a **Live Camera Enhanced Translator** featuring a state-of-the-art **Advanced Translation Engine** that provides real-time speech-to-text transcription and context-aware translation with live camera feed integration. The system combines WhisperLive for real-time audio processing with an intelligent multi-stage translation pipeline, featuring both a user-friendly Gradio web interface and a robust Flask API for advanced schema checking and educational content analysis.

### Key Highlights

- **Advanced Translation Engine**: Next-generation AI translation with context awareness, quality assessment, and cultural adaptation
- **Live Camera Feed**: Real-time webcam video stream integrated with live transcription and translation
- **WhisperLive Integration**: Real-time speech-to-text transcription using WhisperLive for instant audio processing
- **Multi-Language Support**: English to Bengali and Hindi translation with cultural context preservation
- **AI-Powered Analysis**: Reinforcement learning optimization with PPO, DQN, and A3C agents
- **Schema Checker Pipeline**: Educational content analysis against predefined schemas
- **Dual Interface**: Gradio frontend + Flask API backend for comprehensive functionality
- **Production Ready**: Waitress WSGI server, proper error handling, and comprehensive logging
- **Real-Time Processing**: Live audio transcription with synchronized translation output

## Architecture

### Main Components

- ** Gradio Interface**: User-friendly web interface for video upload and translation
- ** Flask API**: RESTful backend with schema management and analysis endpoints
- ** Reinforcement Learning Coordinator**: Multi-agent optimization system (PPO, DQN, A3C)
- ** Schema Checker Pipeline**: Advanced educational content analysis and topic coverage evaluation
- ** Admin Interface**: Comprehensive schema management and batch processing capabilities
- ** WebSocket Support**: Real-time communication for live translation updates

### Key Features

- **Multi-agent RL Optimization**: Enhances translation accuracy by optimizing model parameters in real-time
- **Comprehensive Schema Analysis**: Analyzes translated content against expected educational topics with detailed reporting
- **Batch Processing**: Handle multiple sessions and schemas efficiently
- **RESTful API**: Complete CRUD operations for schemas, reports, and transcripts
- **Production Deployment**: Waitress WSGI server with proper error handling and logging
- **Format Support**: JSON, YAML, CSV schema formats with automatic normalization

## Quick Start

### Prerequisites

- **Python 3.8+** (Required)
- **Google API Key** for translation services
- **Redis server** (optional, for session management)
- **FFmpeg** for video processing
- **Sufficient disk space** for video processing

### Installation

1. **Clone and Navigate**
 ```bash
 git clone <repository-url>
 cd real-time-translator
 ```

2. **Create Virtual Environment**
 ```bash
 python -m venv flask_env
 source flask_env/bin/activate # On Windows: flask_env\\Scripts\\activate
 ```

3. **Install Dependencies**
 ```bash
 pip install -r requirements.txt
 ```

 **Note**: If you encounter missing dependencies, install them manually:
 ```bash
 pip install flask flask-cors gradio watchdog nltk rake-nltk spacy waitress
 ```

4. **Install spaCy Model**
 ```bash
 python -m spacy download en_core_web_sm
 ```

5. **Set Environment Variables**
 ```bash
 export GOOGLE_API_KEY=your_google_api_key_here
 ```

### Usage Options

#### Option A: Gradio Interface Only (Basic Use)

```bash
python3 live_camera_enhanced_ui.py
```

- **Access**: `http://localhost:7860` or `http://localhost:7861`
- **Features**: Video translation, webcam input, basic schema checking

#### Option B: Full System (Recommended)

1. **Start Flask API:**
 ```bash
 python run_flask_api.py
 ```
 - **API Available**: `http://localhost:5001`

2. **Start Gradio Interface** (new terminal):
 ```bash
 source flask_env/bin/activate
 python live_camera_enhanced_ui.py
 ```

### Core Features

#### Live Translation with Camera Feed
- **Live Camera Stream**: Real-time 320x240 webcam video feed in the interface
- **WhisperLive Integration**: Real-time speech-to-text transcription with instant audio processing
- **Synchronized Output**: Live transcript and translation displayed simultaneously
- **Multi-language Support**: English to Bengali/Hindi translation in real-time
- **Cultural Context**: Maintains cultural nuances and context in translations

#### Traditional Video Translation
- Upload video files for processing
- Batch translation with synchronized subtitles
- High-quality audio extraction and processing

#### Schema Checker Admin Panel
- Upload educational schemas (JSON, YAML, CSV)
- Process session transcripts against schemas
- Generate detailed analysis reports
- Batch processing capabilities
- Topic coverage analysis

#### Flask API Endpoints
- **System Info**: `GET /`
- **Schema Management**: `GET/POST/DELETE /api/schemas`
- **Report Generation**: `GET /api/reports`
- **Session Processing**: `POST /api/process`
- **Batch Operations**: `POST /api/batch-process`
- **Health Check**: `GET /api/health`

## System Architecture

The Live Camera Enhanced Translator is built as a **distributed microservices architecture** with real-time capabilities, AI optimization, and educational analytics.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       ADVANCED TRANSLATION ENGINE LAYER                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Context Awareness      │  Quality Assessment       │  Cultural Adaptation     │
│  - Conversation History │  - Real-time Metrics      │  - Regional Variations   │
│  - Domain Detection     │  - Adaptive Feedback      │  - Honorifics Management │
│  - Register Detection   │  - Iterative Improvement  │  - Context-Sensitive     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Gradio UI (7860)     │  Admin Interface (7861)  │  WebRTC Handler           │
│  - Live Camera Feed   │  - Schema Management      │  - Real-time Video        │
│  - Video Upload       │  - Batch Processing       │  - WebSocket Streams      │
│  - Translation UI     │  - Report Generation      │  - Audio Streaming        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            API LAYER                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Flask API (5001/8000)    │  WebSocket Service     │  Load Balancer (Nginx)   │
│  - REST Endpoints         │  - Real-time Updates    │  - SSL/TLS Termination   │
│  - Authentication         │  - Live Transcription   │  - Request Routing       │
│  - File Upload            │  - Translation Stream   │  - Rate Limiting         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PROCESSING LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WhisperLive Client      │  RL Coordinator         │  EgoSchema Integration   │
│  - Real-time ASR         │  - PPO Agent            │  - Video Understanding   │
│  - Speech-to-Text        │  - DQN Agent            │  - Educational Analysis  │
│  - Voice Activity        │  - A3C Agent            │  - Benchmark Testing     │
│  - Multiple Backends     │  - Parameter Optimization│  - Content Evaluation    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         AI/ML LAYER                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Google Gemini API       │  OpenAI Whisper        │  Custom Models           │
│  - Text Translation      │  - Speech Recognition   │  - Classroom-tuned       │
│  - Contextual Analysis   │  - Multiple Model Sizes │  - Fine-tuned Whisper   │
│  - Cultural Adaptation   │  - Real-time Processing │  - Domain Adaptation     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ANALYTICS LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Schema Checker Pipeline │  Performance Monitor   │  Metacognitive Controller │
│  - Topic Extraction      │  - Latency Tracking    │  - Strategy Selection     │
│  - Keyword Analysis      │  - Quality Metrics     │  - Adaptive Learning      │
│  - Educational Alignment │  - Resource Monitoring │  - Context Awareness      │
│  - Report Generation     │  - Real-time Dashboard │  - Decision Making        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Redis Cache (6379)      │  PostgreSQL DB         │  File Storage            │
│  - Session Management    │  - User Data            │  - Video Files           │
│  - Experience Replay     │  - Analytics Data       │  - Audio Files           │
│  - Real-time Queues      │  - Schema Repository    │  - Report Files          │
│  - WebSocket State       │  - Audit Logs          │  - Model Artifacts       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      MONITORING LAYER                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Prometheus (9090)       │  Grafana (3000)        │  Health Checks           │
│  - Metrics Collection    │  - Visualization        │  - Service Discovery     │
│  - Alert Manager         │  - Real-time Dashboards │  - Load Balancing        │
│  - Time Series Data      │  - Performance Analytics│  - Auto-scaling          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **Frontend Services**
- **Gradio UI (Port 7860)**: Main user interface for video translation
- **Admin Interface (Port 7861)**: Schema management and batch processing
- **WebRTC Handler**: Real-time video/audio streaming capabilities

#### 2. **API Gateway**
- **Flask API (Port 5001/8000)**: RESTful API with comprehensive endpoints
- **WebSocket Service**: Real-time communication for live features
- **Nginx Load Balancer**: SSL termination, request routing, and scaling

#### 3. **AI Processing Pipeline**
- **WhisperLive Integration**: Real-time speech-to-text with multiple backend support
- **Reinforcement Learning Coordinator**: Multi-agent optimization (PPO, DQN, A3C)
- **EgoSchema Integration**: Advanced video understanding and educational benchmarking
- **Metacognitive Controller**: Adaptive strategy selection and context-aware processing

#### 4. **Data Processing**
- **Schema Checker Pipeline**: Educational content analysis and topic alignment
- **Performance Monitor**: Real-time metrics and quality assessment
- **Report Generator**: Comprehensive analytics and insights

#### 5. **Storage & Caching**
- **Redis**: Session management, caching, and real-time queues
- **PostgreSQL**: Persistent data storage for users, analytics, and audit logs
- **File System**: Video/audio files, reports, and model artifacts

#### 6. **Monitoring & Observability**
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Real-time dashboards and visualization
- **Health Checks**: Service monitoring and auto-recovery

### Data Flow Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│   Video/    │───▶│   WhisperLive│───▶│  RL Coordinator │───▶│    Gemini    │
│   Audio     │    │   (Real-time │    │  (Optimization) │    │ (Translation)│
│   Input     │    │     ASR)     │    │                 │    │              │
└─────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
       │                   │                      │                    │
       ▼                   ▼                      ▼                    ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  EgoSchema  │    │ Metacognitive│    │Schema Checker   │    │   Report     │
│ Integration │    │  Controller  │    │   Pipeline      │    │ Generation   │
│ (Analysis)  │    │ (Strategy)   │    │ (Educational)   │    │ (Insights)   │
└─────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
       │                   │                      │                    │
       └───────────────────┼──────────────────────┼────────────────────┘
                           ▼                      ▼
                  ┌─────────────────┐    ┌─────────────────┐
                  │   Redis Cache   │    │  Performance    │
                  │  (Sessions &    │    │   Monitor       │
                  │   Real-time)    │    │ (Metrics)       │
                  └─────────────────┘    └─────────────────┘
```

### Detailed Processing Flow

1. **Input Capture**
   - Live camera feed or video upload
   - Real-time audio stream processing
   - WebRTC for browser-based capture

2. **Speech Recognition**
   - WhisperLive real-time transcription
   - Multiple backend support (Faster-Whisper, OpenVINO, TensorRT)
   - Voice Activity Detection (VAD)
   - Custom classroom-trained models

3. **AI Optimization**
   - RL Coordinator optimizes translation parameters
   - Multi-agent approach (PPO for quality, DQN for latency, A3C for schemas)
   - Metacognitive controller selects optimal strategies
   - Context-aware parameter adjustment

4. **Translation Processing**
   - Google Gemini API with optimized parameters
   - Cultural context preservation
   - Educational terminology handling
   - Real-time streaming translation

5. **Educational Analysis**
   - Schema checker validates content alignment
   - Topic extraction and keyword analysis
   - EgoSchema integration for video understanding
   - Educational benchmark evaluation

6. **Analytics & Reporting**
   - Real-time performance monitoring
   - Comprehensive analytics dashboard
   - Educational effectiveness metrics
   - Automated report generation

7. **Data Persistence**
   - Session data in Redis for real-time access
   - Long-term storage in PostgreSQL
   - File storage for media and reports
   - Audit logging for compliance

## Technologies Used

The Live Camera Enhanced Translator leverages a comprehensive technology stack designed for scalability, performance, and educational effectiveness.

### **Core Technologies**

#### **Programming Languages**
- **Python 3.8+**: Primary language for backend services and AI processing
- **JavaScript**: Frontend interactions and WebRTC implementations
- **HTML/CSS**: User interface components and styling

#### **Web Frameworks**
- **Gradio 5.35+**: Interactive web interface for video translation
- **Flask 2.0+**: RESTful API backend with microservices architecture
- **FastAPI 0.115+**: High-performance async API endpoints
- **Uvicorn 0.35+**: ASGI server for FastAPI applications
- **Starlette 0.46+**: Lightweight ASGI framework components

### **AI/ML Technologies**

#### **Speech Recognition & Processing**
- **OpenAI Whisper 20250625+**: State-of-the-art speech-to-text recognition
- **Faster-Whisper 1.1+**: Optimized Whisper implementation for real-time processing
- **WhisperLive**: Real-time streaming speech recognition
- **CTranslate2 4.0+**: Efficient inference engine for Transformer models
- **WebRTCVAD 2.0+**: Voice Activity Detection for audio preprocessing

#### **Machine Learning Frameworks**
- **PyTorch 2.7+**: Deep learning framework for model training and inference
- **Transformers 4.53+**: Hugging Face library for transformer models
- **Stable-Baselines3**: Reinforcement learning algorithms (PPO, DQN, A3C)
- **Accelerate 1.8+**: Distributed training and inference optimization
- **ONNX Runtime 1.20+**: Cross-platform model inference

#### **Translation & Language Processing**
- **Google Generative AI 0.8.5+**: Gemini API for contextual translation
- **spaCy 3.8+**: Advanced natural language processing
- **NLTK 3.9+**: Natural language toolkit for text analysis
- **RAKE-NLTK 1.0+**: Keyword extraction algorithms
- **Tiktoken 0.9+**: Tokenization for language models

### **Data Processing & Analytics**

#### **Data Science Libraries**
- **NumPy 2.2+**: Numerical computing and array operations
- **Pandas 2.3+**: Data manipulation and analysis
- **SciPy 1.16+**: Scientific computing and statistics
- **Scikit-learn 1.5+**: Machine learning algorithms and metrics

#### **Audio/Video Processing**
- **PyAudio 0.2.14+**: Real-time audio input/output
- **OpenCV 4.11+**: Computer vision and video processing
- **FFmpeg**: Video/audio encoding and format conversion
- **Librosa 0.10+**: Audio analysis and feature extraction
- **PyDub 0.25+**: Audio file manipulation
- **SoundDevice 0.5+**: Audio device interface
- **SoundFile 0.13+**: Audio file I/O operations

### **Database & Storage**

#### **Databases**
- **Redis 7**: In-memory data structure store for caching and session management
- **PostgreSQL 15**: Relational database for persistent data storage
- **HiRedis 2.0+**: High-performance Redis client

#### **File Processing**
- **Pillow 11.3+**: Image processing and manipulation
- **PyPDF2 3.0+**: PDF document processing
- **OpenPyXL 3.1+**: Excel file processing
- **Python-DOCX 0.8+**: Word document processing
- **Markdown 3.5+**: Markdown processing and conversion

### **Web Technologies & Communication**

#### **HTTP & Networking**
- **Requests 2.32+**: HTTP library for API calls
- **HTTPX 0.28+**: Async HTTP client
- **aioHTTP 3.12+**: Asynchronous HTTP client/server
- **WebSockets 15.0+**: Real-time bidirectional communication
- **WebSocket-Client 1.8+**: WebSocket client library

#### **Security & Authentication**
- **Cryptography 45.0+**: Cryptographic recipes and primitives
- **PyOpenSSL 25.1+**: SSL/TLS wrapper for secure communications
- **Certifi 2025.6+**: Certificate verification
- **Flask-CORS**: Cross-Origin Resource Sharing support

### **DevOps & Infrastructure**

#### **Containerization & Orchestration**
- **Docker**: Application containerization
- **Docker Compose**: Multi-container application orchestration
- **Kubernetes**: Container orchestration and scaling
- **Nginx**: Load balancing, reverse proxy, and SSL termination

#### **Cloud Platforms**
- **AWS**: Elastic Container Service, EC2, S3, CloudWatch
- **Google Cloud Platform**: Kubernetes Engine, Cloud Storage, AI Platform
- **Microsoft Azure**: Container Instances, Kubernetes Service, Storage

#### **CI/CD & Automation**
- **GitHub Actions**: Continuous integration and deployment
- **Gunicorn**: Python WSGI HTTP Server
- **Waitress**: Production-ready WSGI server

### **Monitoring & Observability**

#### **Metrics & Monitoring**
- **Prometheus**: Time-series database and monitoring system
- **Grafana**: Visualization and analytics platform
- **Prometheus Client**: Python metrics exposition library
- **AlertManager**: Alert handling and routing

#### **Logging & Debugging**
- **Python Logging**: Built-in logging framework
- **Rich 14.0+**: Rich text and beautiful formatting
- **Click 8.2+**: Command-line interface creation
- **Typer 0.16+**: Modern CLI applications

### **Development & Testing**

#### **Testing Frameworks**
- **pytest 8.4+**: Testing framework
- **pytest-asyncio 0.25+**: Async testing support
- **pytest-cov 6.0+**: Coverage reporting
- **pytest-mock 3.14+**: Mock object library

#### **Code Quality & Formatting**
- **Black 24.10+**: Code formatting
- **Flake8 7.1+**: Code linting and style checking
- **isort 5.13+**: Import sorting
- **MyPy 1.13+**: Static type checking

#### **Performance Testing**
- **Locust**: Load testing framework
- **tqdm 4.67+**: Progress bars and performance monitoring

### **Utility Libraries**

#### **File & System Operations**
- **pathlib**: Object-oriented filesystem paths
- **tempfile**: Temporary file and directory creation
- **shutil**: High-level file operations
- **zipfile**: ZIP archive processing
- **json/orjson 3.10+**: JSON processing (optimized)
- **PyYAML 6.0+**: YAML processing
- **python-dateutil 2.9+**: Date/time utilities

#### **Configuration & Environment**
- **python-dotenv 1.0+**: Environment variable loading
- **configparser**: Configuration file processing
- **argparse**: Command-line argument parsing
- **os/pathlib**: Operating system interface

### **Educational & Specialized Technologies**

#### **Educational AI**
- **EgoSchema**: Video understanding benchmark for educational content
- **Custom Whisper Models**: Fine-tuned for classroom environments
- **Educational Schema Validation**: Custom validation for curriculum alignment

#### **Reinforcement Learning**
- **PPO (Proximal Policy Optimization)**: Translation quality optimization
- **DQN (Deep Q-Network)**: Latency minimization
- **A3C (Asynchronous Actor-Critic)**: Schema generation optimization
- **Experience Replay**: Learning from historical interactions

### **Hardware & Performance**

#### **GPU Acceleration**
- **CUDA Support**: NVIDIA GPU acceleration (optional)
- **TensorRT**: High-performance deep learning inference
- **OpenVINO**: Intel hardware optimization

#### **Performance Optimization**
- **Multiprocessing**: Parallel processing capabilities
- **Threading**: Concurrent execution
- **Asyncio**: Asynchronous programming
- **Queue**: Thread-safe data structures

### **Integration Technologies**

#### **External APIs**
- **Google Gemini API**: Advanced language understanding and generation
- **Google Cloud Translation**: Backup translation service
- **Hugging Face Hub**: Model repository and inference

#### **Real-time Communication**
- **WebRTC**: Real-time peer-to-peer communication
- **WebSocket**: Bidirectional real-time communication
- **Server-Sent Events**: Real-time server-to-client updates

### **Deployment Architectures**

#### **Local Development**
- **Virtual Environments**: Isolated Python environments
- **Development Servers**: Built-in development servers
- **Hot Reloading**: Automatic code reloading during development

#### **Production Deployment**
- **Microservices Architecture**: Loosely coupled, independently deployable services
- **Load Balancing**: Distribute requests across multiple instances
- **Auto-scaling**: Dynamic resource allocation based on demand
- **Health Checks**: Automated service monitoring and recovery
- **SSL/TLS**: Secure communication protocols

#### **Monitoring & Alerting**
- **Real-time Dashboards**: Live system performance visualization
- **Alerting Rules**: Automated notification system
- **Log Aggregation**: Centralized logging and analysis
- **Performance Metrics**: System resource utilization tracking

This comprehensive technology stack ensures the Live Camera Enhanced Translator is robust, scalable, and suitable for both educational and production environments.

## Module Descriptions

### 1. RL Coordinator (`src/rl_coordinator.py`)
Manages multiple reinforcement learning agents:
- **PPO Agent**: Optimizes translation quality parameters
- **DQN Agent**: Minimizes latency in processing
- **A3C Agent**: Generates optimal schemas
- **Experience Replay**: Redis-backed experience storage

### 2. Schema Checker Pipeline (`schema_checker/`)
Comprehensive analysis system:
- **Schema Parser**: Normalizes JSON/YAML/CSV schema files
- **Keyword Extractor**: Extracts topics using RAKE/TF-IDF/Hybrid methods
- **Topic Comparator**: Compares expected vs. actual topics
- **Report Generator**: Creates detailed analysis reports

### 3. Flask API (`src/flask_api.py`)
RESTful API with endpoints:
- `/api/translate`: Synchronous translation
- `/api/schema/<session_id>`: Session schema retrieval
- `/api/sessions`: Session management
- `/api/topics/compare`: Topic comparison
- `/metrics`: Prometheus metrics

### 4. Performance Monitor (`src/performance_monitor.py`)
Monitors system performance:
- Translation latency tracking
- Resource utilization metrics
- Quality score calculation
- Real-time dashboard updates

## Configuration

### Environment Variables

Environment variables are crucial for the application's configuration. Below is a complete list of variables used for different deployment scenarios:

```bash
# Core Configuration
GOOGLE_API_KEY=your_google_api_key_here
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@postgres:5432/translator
SECRET_KEY=your_secret_key_here
PROMETHEUS_PORT=9090

# Server Configuration
FLASK_PORT=5001
GRADIO_PORT=7860
ENVIRONMENT=production
LOG_LEVEL=info

# AI Model Configuration
WHISPER_MODEL=base
TRANSLATION_MODEL=gemini
RL_LEARNING_RATE=0.001
EXPERIENCE_REPLAY=True
REINFORCEMENT_STRATEGY=adaptive

# WebSocket Configuration
WEBSOCKET_PING_INTERVAL=30
WEBSOCKET_PING_TIMEOUT=10
WEBSOCKET_MAX_SIZE=1048576 # 1MB
WEBSOCKET_COMPRESSION=deflate

# Language Support
DEFAULT_TARGET_LANGUAGE=bn # Default is Bengali
SUPPORTED_LANGUAGES=bn,en,hi

# Redis Configuration
REDIS_MAX_CONNECTIONS=20
REDIS_SSL=False
SESSION_TTL=3600 # Session timeout in seconds

# GPU  Deep Learning Configuration
CUDA_ENABLED=True
TENSORRT_ACCELERATION=True

# Monitoring and Analytics
METRICS_ENABLED=True
ALERTS_ENABLED=True

# Security
ENABLE_SECURE_SSL_REDIRECT=True
ACCESS_LOG=True
CORS_ALLOWED_ORIGINS=* # Enable cors for specific domains

# Development Settings
FLASK_ENV=development
FLASK_DEBUG=1
LOG_FILE=app.log
LOG_ROTATION=daily
```

### Configuration Files

The system relies on several configuration files to set up environments.

- **`.env`**: Environment variable definitions for local development.
- **`config.json`**: Application-specific configurations, can be adapted for different environments.

Example `config.json` structure:

```json
{
  "logging": {
    "level": "info",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "handlers": ["file", "console"]
  },
  "database": {
    "engine": "postgresql",
    "host": "localhost",
    "port": 5432,
    "database": "translator",
    "username": "user",
    "password": "pass"
  },
  "websocket": {
    "enabled": true,
    "pingInterval": 30,
    "pingTimeout": 10,
    "maxReconnectAttempts": 10
  }
}
```

### Environment Files

**Production Environment**

Ensure secure handling of sensitive data using environment variables. All sensitive information such as API keys and database passwords should be stored securely.

- **AWS**: Use AWS Secrets Manager or Parameter Store.
- **GCP**: Use Secret Manager.
- **Azure**: Use Key Vault.

**Development Environment**

For local development, ensure that environment variables are set up in the `.env` file at the root of the project directory:

```ini
DEBUG=True
GOOGLE_API_KEY=your_development_key_here
SECRET_KEY=your_development_secret_key
```

### Redis Configuration

Redis is used for caching and session management. Here are the configurable parameters:

- **Host**: localhost
- **Port**: 6379
- **Session TTL**: 3600 seconds
- **Max Connections**: 20
- **SSL Enabled**: False

### Monitoring Configuration

Ensure that all monitoring systems are properly configured:

- **Prometheus**: Metrics collection and visualisation.
- **Grafana**: Dashboards for real-time monitoring.
- **AlertManager**: Configure alerts for key performance indicators.

## Deployment

### Production Deployment

The Live Camera Enhanced Translator is designed for scalable and robust production deployment using Docker Compose, Kubernetes, and cloud platforms.

#### **Docker Compose (Recommended)**

This setup is suitable for local development and small-scale production environments.

1. **Environment Configuration:**
   - Create a `.env` file in the project root to define your environment variables:
   ```ini
   GOOGLE_API_KEY=your_google_api_key_here
   SECRET_KEY=your_secret_key_here
   REDIS_URL=redis://redis:6379
   DATABASE_URL=postgresql://user:pass@postgres:5432/translator
   ````

2. **Launch Services:**
   ```bash
   docker-compose up -d
   ```

3. **Access Services:**
   - Gradio UI: `http://localhost`
   - Flask API: `http://localhost:5001`
   - Prometheus: `http://localhost:9090`
   - Grafana: `http://localhost:3000`

#### **Kubernetes Deployment**

Kubernetes provides scalable and resilient deployment for cloud environments.

1. **Configure Kubernetes Cluster:**
   - Ensure your cloud provider's CLI is installed (e.g., `gcloud`, `aws`, `az`).
   - Set up a Kubernetes cluster and configure `kubectl`.

2. **Apply Kubernetes Manifests:**
   ```bash
   kubectl apply -f cloud-deploy/kubernetes/deployment.yaml
   ```

3. **Monitor and Scale:**
   - Use Kubernetes Dashboard or CLI for monitoring.
   - Scale automatically using Horizontal Pod Autoscalers.

#### **Cloud Deployment**

Deploy on AWS, GCP, or Azure using native services for the best integration.

**AWS Deployment:**
- Use Elastic Compute Service (ECS) or Kubernetes Service (EKS).
- Utilize CloudFormation or Terraform for infrastructure as code.

**Google Cloud Deployment:**
- Deploy using Google Kubernetes Engine (GKE) or App Engine.
- Use Cloud Deployment Manager for resource management.

**Azure Deployment:**
- Use Azure Kubernetes Service (AKS) or Azure App Services.
- Manage infrastructure with Azure Resource Manager templates.

### Classroom Deployment

**For educational environments**, local deployment must be straightforward and accessible.

1. **Hardware Requirements:**
   - Minimum: 8GB RAM, dual-core processor
   - Recommended: 16GB RAM, quad-core processor

2. **Network Considerations:**
   - Ensure sufficient bandwidth for video/audio streaming.
   - Local network firewall settings should allow HTTP/HTTPS traffic.

3. **Local Environment Setup:**
   - Set up Python virtual environments for isolation.
   - Use Docker for local service isolation and simplified deployment.

4. **Deployment Instructions:**
   - Configure environment variables for classroom network settings.
   - Start services using Docker or Python scripts from the terminal.

5. **Educational Use Cases:**
   - Support for live classroom translation and transcription.
   - Schema checks against predefined educational content.

### Monitoring & Maintenance

1. **Logs & Alerts:**
   - Use Grafana for real-time dashboarding and alerts.
   - Store logs centrally for compliance and troubleshooting.

2. **Performance Monitoring:**
   - Track key metrics with Prometheus (e.g., latency, usage stats).
   - Set up alert rules for threshold breaches.

3. **Maintenance & Updates:**
   - Regularly update Docker images and Kubernetes deployments.
   - Apply patches and updates to comply with security standards.

These deployment strategies ensure that the Live Camera Enhanced Translator is adaptable across diverse settings, providing optimal functionality whether in a classroom, local server, or cloud data center.

## API Endpoints

### Translation API
```bash
POST /api/translate
Content-Type: application/json

{
 "text": "Hello, how are you?",
 "target_language": "Bengali",
 "session_id": "optional-session-id"
}
```

### Session Management
```bash
GET /api/sessions # List all sessions
GET /api/schema/{session_id} # Get session schema
DELETE /api/sessions/{session_id} # Close session
```

### Topic Analysis
```bash
POST /api/topics/compare
Content-Type: application/json

{
 "session_id": "session-uuid",
 "topic": "mathematics"
}
```

## Monitoring and Metrics

### Prometheus Metrics
- `translation_requests_total`: Total translation requests by language
- `translation_latency_seconds`: Translation request latency histogram
- `active_sessions_count`: Number of active sessions
- `websocket_connections_count`: Active WebSocket connections

### Health Check
```bash
GET /health
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Structure
```
live-camera-enhanced-translator/
 live_camera_enhanced_ui.py # Main Live Camera Enhanced UI
 app.py # Basic Gradio application
 requirements.txt # Python dependencies
 docker-compose.yml # Docker configuration
 src/ # Source code modules
 rl_coordinator.py # RL optimization system
 flask_api.py # REST API server
 performance_monitor.py # Performance tracking
 ...
 schema_checker/ # Schema analysis pipeline
 main.py # Pipeline coordinator
 schema_parser.py # Schema normalization
 keyword_extractor.py # Topic extraction
 ...
 schemas/ # Educational schemas
 normalized/ # Processed schema files
 transcripts/ # Session transcripts
 reports/ # Analysis reports
 tests/ # Test suite
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License
MIT License - see LICENSE file for details

## Troubleshooting

### Common Issues

#### [WARNING] Port Already in Use
```bash
# Error: Port 5000 is already in use
# Solution: Use a different port
python flask_api.py --port 5001
```

#### [WARNING] Missing Dependencies
```bash
# Error: ModuleNotFoundError: No module named 'spacy'
# Solution: Install missing packages
pip install spacy
python -m spacy download en_core_web_sm
```

#### [WARNING] API Key Issues
```bash
# Error: Authentication failed
# Solution: Check your Google API key
echo $GOOGLE_API_KEY # Should show your key
export GOOGLE_API_KEY="your_actual_key_here"
```

#### [WARNING] WebSocket Connection Failed
```bash
# Error: WebSocket connection failed
# Solution: Check if Flask API is running
curl http://localhost:5001/api/health
```

#### [WARNING] Video Processing Issues
```bash
# Error: FFmpeg not found
# Solution: Install FFmpeg
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from https://ffmpeg.org/
```

#### [WARNING] Schema Upload Issues
```bash
# Error: Schema validation failed
# Solution: Check schema format
# Supported formats: JSON, YAML, CSV
# Example valid schema structure:
{
 "topics": ["mathematics", "science", "history"],
 "keywords": ["algebra", "physics", "world war"]
}
```

### Performance Issues

#### Memory Usage
- **Issue**: High memory usage during video processing
- **Solution**: Process videos in chunks, reduce concurrent sessions

#### Translation Latency
- **Issue**: Slow translation response times
- **Solution**: Enable Redis caching, optimize API calls

#### Disk Space
- **Issue**: Running out of disk space
- **Solution**: Clean up temporary files, implement file rotation

### Debug Mode

```bash
# Enable debug logging
export FLASK_ENV=development
export FLASK_DEBUG=1
export LOG_LEVEL=DEBUG

# Run with verbose output
python flask_api.py --verbose
```

## Advanced Configuration

### Environment Variables

```bash
# Core Configuration
GOOGLE_API_KEY=your_google_api_key_here
REDIS_URL=redis://localhost:6379
SECRET_KEY=your_secret_key_here

# Server Configuration
FLASK_PORT=5001
GRADIO_PORT=7860
PROMETHEUS_PORT=9090

# Processing Configuration
MAX_CONCURRENT_SESSIONS=10
SESSION_TIMEOUT=3600
MAX_FILE_SIZE=100MB

# AI Model Configuration
WHISPER_MODEL=base
TRANSLATION_MODEL=gemini-pro
RL_LEARNING_RATE=0.001

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=app.log
LOG_ROTATION=daily
```

### Custom Schemas

#### JSON Schema Format
```json
{
 "name": "Mathematics Course",
 "description": "Basic mathematics curriculum",
 "topics": [
 "algebra",
 "geometry",
 "statistics"
 ],
 "keywords": [
 "equation",
 "triangle",
 "probability"
 ],
 "metadata": {
 "level": "beginner",
 "duration": "1 hour"
 }
}
```

#### YAML Schema Format
```yaml
name: Mathematics Course
description: Basic mathematics curriculum
topics:
 - algebra
 - geometry
 - statistics
keywords:
 - equation
 - triangle
 - probability
metadata:
 level: beginner
 duration: 1 hour
```

### WebSocket Configuration

```python
# Custom WebSocket settings
WEBSOCKET_CONFIG = {
 "ping_interval": 30,
 "ping_timeout": 10,
 "max_size": 1024 * 1024, # 1MB
 "compression": "deflate"
}
```

### Batch Processing

```bash
# Process multiple sessions
curl -X POST http://localhost:5001/api/batch-process \
 -H "Content-Type: application/json" \
 -d '{
 "sessions": ["session1", "session2", "session3"],
 "schema_id": "math-schema",
 "format": "json"
 }'
```

### Custom Metrics

```python
# Add custom Prometheus metrics
from prometheus_client import Counter, Histogram

custom_counter = Counter('custom_translations_total', 'Total custom translations')
custom_histogram = Histogram('custom_latency_seconds', 'Custom operation latency')
```

## API Reference

### Core Endpoints

#### System Information
```bash
GET /
# Returns API metadata and system info
```

#### Health Check
```bash
GET /api/health
# Returns: {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}
```

#### Schema Management
```bash
# List all schemas
GET /api/schemas

# Upload new schema
POST /api/schemas
Content-Type: multipart/form-data
# Form field: schema_file

# Get specific schema
GET /api/schemas/{schema_id}

# Delete schema
DELETE /api/schemas/{schema_id}
```

#### Session Processing
```bash
# Process single session
POST /api/process
Content-Type: application/json
{
 "session_id": "session-uuid",
 "schema_id": "schema-uuid",
 "transcript": "session transcript text"
}

# Batch process sessions
POST /api/batch-process
Content-Type: application/json
{
 "sessions": ["session1", "session2"],
 "schema_id": "schema-uuid"
}
```

#### Report Generation
```bash
# List all reports
GET /api/reports

# Get specific report
GET /api/reports/{report_id}

# Export report
GET /api/reports/{report_id}/export?format=json
```

#### Data Export
```bash
# Export all data
GET /api/export?format=json

# Export specific data types
GET /api/export?format=csv&type=sessions
GET /api/export?format=yaml&type=schemas
```

### Response Formats

#### Success Response
```json
{
 "status": "success",
 "data": {
 "id": "resource-id",
 "message": "Operation completed successfully"
 },
 "timestamp": "2024-01-01T00:00:00Z"
}
```

#### Error Response
```json
{
 "status": "error",
 "error": {
 "code": "VALIDATION_ERROR",
 "message": "Invalid request data",
 "details": {
 "field": "schema_id",
 "reason": "Required field missing"
 }
 },
 "timestamp": "2024-01-01T00:00:00Z"
}
```

## Production Deployment

### Docker Deployment

```dockerfile
# Dockerfile example
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5001 7860

CMD ["python", "run_flask_api.py"]
```

### Load Balancing

```nginx
# Nginx configuration
upstream flask_api {
 server 127.0.0.1:5001;
 server 127.0.0.1:5002;
 server 127.0.0.1:5003;
}

server {
 listen 80;
 location /api/ {
 proxy_pass http://flask_api;
 proxy_set_header Host $host;
 proxy_set_header X-Real-IP $remote_addr;
 }
}
```

### Monitoring Setup

```yaml
# docker-compose.yml monitoring stack
version: '3.8'
services:
 prometheus:
 image: prom/prometheus
 ports:
 - "9090:9090"
 volumes:
 - ./prometheus.yml:/etc/prometheus/prometheus.yml
 
 grafana:
 image: grafana/grafana
 ports:
 - "3000:3000"
 environment:
 - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Resources

### Documentation
- [Flask API Documentation](docs/api.md)
- [Schema Format Guide](docs/schemas.md)
- [Deployment Guide](docs/deployment.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

### External Resources
- [Gradio Documentation](https://gradio.app/docs)
- [Flask Documentation](https://flask.palletsprojects.com)
- [Whisper Documentation](https://github.com/openai/whisper)
- [Google AI Documentation](https://ai.google.dev)

## Contributing

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/real-time-translator.git
cd real-time-translator

# Create development environment
python -m venv dev_env
source dev_env/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Code formatting
black .
flake8 .
```

### Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes with tests
4. **Commit** your changes (`git commit -m 'Add amazing feature'`)
5. **Push** to the branch (`git push origin feature/amazing-feature`)
6. **Open** a Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Write tests for new features
- Update documentation as needed

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

### Bug Reports
For bugs and issues, please create an issue in the repository with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version)

### [TIP] Feature Requests
For new features, please create an issue with:
- Clear description of the feature
- Use case and benefits
- Implementation suggestions (if any)

### Community
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and community support
- **Wiki**: Additional documentation and guides

---

** Star this repository if you find it useful!**

** Last Updated**: July 2025
