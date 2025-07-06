# 🤖 News Recommendation Engine with ML Stack

A complete news recommendation system powered by Elasticsearch, ELSER semantic search, multilingual embeddings, Claude AI, and intelligent reranking.

## 🚀 Key Features

- **🧠 Advanced ML Stack**: ELSER, Multilingual Embeddings, Claude AI, Reranking
- **📡 RSS Crawling**: Automatic news ingestion with proper content extraction
- **🔄 Real Vectorization**: Proper ML ingest pipeline with sparse and dense vectors
- **👥 Dynamic User Profiles**: Real-time interest modification and personalization
- **🎯 AI Recommendations**: Claude-powered intelligent content selection
- **🌐 Interactive Demo UI**: Web interface for testing and demonstration

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RSS Crawler   │────│  Elasticsearch  │────│   Claude AI     │
│                 │    │                 │    │                 │
│ • Content Prep  │    │ • ELSER Vectors │    │ • Intelligence  │
│ • Category Inf. │    │ • Dense Vectors │    │ • Reasoning     │
│ • Rich Metadata │    │ • Hybrid Search │    │ • Selection     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Recommendation  │
                       │     Engine      │
                       │                 │
                       │ • Personalization│
                       │ • Reranking     │
                       │ • Analytics     │
                       └─────────────────┘
```

## 📋 Prerequisites

- **Python 3.8+**
- **Elasticsearch Cloud Instance** with inference endpoints configured
- **Internet connection** for RSS feed access

## 🛠️ Setup Instructions

### 1. Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd news-recommendation-engine

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file with your Elasticsearch credentials:

```bash
# Copy the template
cp .env.template .env

# Edit with your actual credentials
nano .env
```

Required configuration:
```env
ES_CLOUD_ID=your_cloud_id_here
ES_API_KEY=your_api_key_here  
ES_BASE_URL=your_elasticsearch_url_here
RSS_URL=https://www.abs-cbn.com/rss/mobile/latest-news
LOG_LEVEL=INFO
```

### 3. Verify Inference Endpoints

Ensure these inference endpoints exist in your Elasticsearch cluster:

- **ELSER**: `.elser-2-elasticsearch` (sparse embedding)
- **Multilingual**: `.multilingual-e5-small-elasticsearch` (dense embedding) 
- **Reranking**: `.rerank-v1-elasticsearch` (result reranking)
- **Claude AI**: `claude-completions` (AI reasoning)

### 4. Initialize System

```bash
# Setup indices and pipeline
python main.py --setup

# Verify system status
python main.py --status
```

### 5. Run Demo

```bash
# Complete demonstration
python main.py --demo

# Or start web UI
python main.py --web
```

## 🌐 Web UI Demo

Start the interactive web interface:

```bash
python main.py --web
# Open http://localhost:5000 in your browser
```

### Web UI Features:

- **📊 System Status**: Real-time monitoring of ML stack
- **👥 User Management**: Dynamic interest modification
- **🎯 AI Recommendations**: Claude-powered content selection  
- **📰 Content Crawling**: Real-time news ingestion
- **🔧 System Controls**: Full system management

## 📚 Usage Guide

### Command Line Interface

```bash
# System setup and status
python main.py --setup           # Initialize system
python main.py --status          # Show system status
python main.py --demo            # Run complete demo

# Content management
python main.py --crawl           # Crawl and index news
python main.py --articles        # List recent articles

# User management  
python main.py --users           # List all users
python main.py --recommend       # Generate recommendations for all
python main.py --recommend --user-id <id>  # User-specific recommendations

# Testing and debugging
python main.py --test-search "basketball"   # Test hybrid search
python main.py --web             # Start web UI
```

### API Endpoints

The web application exposes these REST endpoints:

- `GET /api/status` - System status and health
- `GET /api/users` - List all users
- `POST /api/users/{id}/interests/add` - Add user interest
- `POST /api/users/{id}/interests/remove` - Remove user interest
- `GET /api/recommendations/{id}` - Get user recommendations
- `POST /api/crawl` - Trigger news crawling
- `POST /api/initialize` - Initialize system

## 🧠 ML Technology Stack

### 1. ELSER Semantic Search
- **Purpose**: Deep semantic understanding beyond keywords
- **Technology**: Elasticsearch Learned Sparse EncodeR
- **Usage**: Primary semantic matching for content discovery

### 2. Multilingual Dense Embeddings  
- **Purpose**: 384-dimensional vector similarity matching
- **Technology**: multilingual-e5-small model
- **Usage**: Cross-lingual content understanding and similarity

### 3. Claude AI Intelligence
- **Purpose**: Advanced reasoning for content selection
- **Technology**: Claude Sonnet 4 via Anthropic API
- **Usage**: Intelligent article ranking and reasoning

### 4. Intelligent Reranking
- **Purpose**: ML-powered relevance optimization
- **Technology**: Elasticsearch rerank-v1 model
- **Usage**: Final result optimization for relevance

## 📊 System Components

### Elasticsearch Indices

#### News Articles (`news_recommendation_hybrid`)
```json
{
  "id": "unique_article_id",
  "title": "Article title",
  "description": "Rich extracted content",
  "content": "Full article text",
  "semantic_content": "Combined content for ELSER",
  "categories": ["inferred", "categories"],
  "published_date": "2025-01-15T10:00:00",
  "source": "ABS-CBN",
  "ml": {
    "inference": {
      "semantic_content": "sparse_vector",
      "title_embedding": "dense_vector[384]",
      "description_embedding": "dense_vector[384]"
    }
  }
}
```

#### User Profiles (`user_profiles_dynamic`)
```json
{
  "user_id": "unique_user_id",
  "name": "User Name", 
  "interests": ["sports", "technology"],
  "preferences": {
    "categories": ["sports", "tech"],
    "keywords": ["basketball", "ai", "innovation"],
    "sources": ["ABS-CBN"]
  },
  "reading_history": [
    {
      "article_id": "article_id",
      "timestamp": "2025-01-15T10:00:00",
      "engagement_score": 0.85
    }
  ]
}
```

### ML Ingest Pipeline (`news-vectorization-pipeline`)

Automatically processes articles during indexing:

1. **Content Preparation**: Combines title, description, content
2. **ELSER Processing**: Generates sparse semantic vectors
3. **Dense Embeddings**: Creates multilingual vector representations
4. **Error Handling**: Graceful fallback for processing failures

## 🎯 Recommendation Process

### 1. User Context Analysis
- Extract user interests and reading patterns
- Generate contextual search queries
- Identify preference patterns from history

### 2. Hybrid Content Discovery
- **ELSER Semantic Search**: Deep content understanding
- **Vector Similarity**: Multilingual embedding matching  
- **Keyword Matching**: Traditional text search fallback
- **User Filtering**: Interest-based content filtering

### 3. Personalization Scoring
- Reading history pattern analysis
- Interest relevance weighting
- Recency and engagement bonuses
- Combined ML and behavioral scoring

### 4. Claude AI Intelligence
- Advanced reasoning for content selection
- Personalized article ranking with explanations
- Diverse perspective recommendation
- Quality and relevance assessment

### 5. Final Reranking
- ML-powered relevance optimization
- User engagement prediction
- Final result ordering and presentation

## 🔧 Configuration Options

### Environment Variables

```env
# Elasticsearch Configuration
ES_CLOUD_ID=your_cluster_cloud_id
ES_API_KEY=your_api_key
ES_BASE_URL=your_elasticsearch_base_url

# RSS Feed Settings
RSS_URL=https://www.abs-cbn.com/rss/mobile/latest-news
CRAWL_INTERVAL=300

# Application Settings  
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Inference Endpoint Configuration

The system expects these pre-configured inference endpoints:

```python
ELSER_INFERENCE_ID = '.elser-2-elasticsearch'
MULTILINGUAL_INFERENCE_ID = '.multilingual-e5-small-elasticsearch'  
RERANK_INFERENCE_ID = '.rerank-v1-elasticsearch'
CLAUDE_INFERENCE_ID = 'claude-completions'
```

## 📈 Performance Monitoring

### System Status Monitoring
- **Vectorization Coverage**: Percentage of articles with ML vectors
- **Response Times**: Average recommendation generation time
- **User Engagement**: Reading history and interaction analytics
- **ML Stack Health**: Status of all inference endpoints

### Analytics Available
- User-specific recommendation performance
- Content discovery effectiveness  
- ML feature utilization rates
- System throughput and latency metrics

## 🛠️ Troubleshooting

### Common Issues

#### "System not initialized" 
```bash
python main.py --setup
```

#### "No ML vectors found"
- Verify inference endpoints are running
- Check ingest pipeline configuration
- Re-crawl content: `python main.py --crawl`

#### "Claude AI failed"
- Verify Claude inference endpoint exists
- Check API rate limits and quotas
- System falls back to ML-only recommendations

#### "No articles found"
- Check RSS feed accessibility
- Verify network connectivity
- RSS will fall back to enhanced mock articles

### Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
python main.py --status
```

### Verification Commands

```bash
# Check system components
python main.py --status

# Test search functionality
python main.py --test-search "sports"

# Verify user profiles
python main.py --users

# Check indexed content
python main.py --articles
```

## 🚀 Advanced Usage

### Custom User Creation

```python
from user_profiles import UserProfileManager
from elasticsearch_client import ElasticsearchClient

es_client = ElasticsearchClient()
user_manager = UserProfileManager(es_client)

user_data = {
    "name": "Custom User",
    "interests": ["technology", "ai", "science"],
    "additional_keywords": ["machine learning", "research"],
    "preferred_sources": ["ABS-CBN"]
}

user_id = user_manager.create_user_profile(user_data)
```

### Direct API Usage

```python
from recommendation_engine import NewsRecommendationEngine

engine = NewsRecommendationEngine(es_client, user_manager)
recommendations = engine.generate_personalized_recommendations(user_id)
```

### Search Testing

```python
# Test hybrid search
articles = es_client.hybrid_search(
    query="artificial intelligence", 
    user_interests=["technology", "innovation"],
    size=10
)
```

## 📝 Development Notes

### File Structure
```
news-recommendation-engine/
├── config.py                 # Configuration management
├── elasticsearch_client.py   # ML-powered Elasticsearch client
├── rss_crawler.py           # RSS crawling and content extraction
├── user_profiles.py         # Dynamic user profile management
├── recommendation_engine.py # AI-powered recommendation engine
├── web_app.py              # Flask web interface
├── main.py                 # Main application orchestrator
├── requirements.txt        # Python dependencies
├── .env                   # Environment configuration
└── README.md             # This file
```

### Key Design Principles
- **ML-First Approach**: Proper vectorization at ingestion time
- **Dynamic Personalization**: Real-time user interest modification
- **Fallback Strategies**: Graceful degradation when ML components fail
- **Performance Focus**: Optimized for real-world usage patterns
- **Demo-Ready**: Interactive web UI for easy demonstration

## 🤝 Contributing

This is a demonstration system showcasing modern ML-powered search and recommendation capabilities. Key areas for enhancement:

- Additional news sources and content types
- Advanced user behavior modeling
- Real-time learning and adaptation
- A/B testing framework for recommendation strategies
- Extended analytics and monitoring capabilities

## 📄 License

This project is intended for demonstration and educational purposes.

---

**🚀 Ready to explore AI-powered news recommendations!**

Start with: `python main.py --demo` or `python main.py --web`