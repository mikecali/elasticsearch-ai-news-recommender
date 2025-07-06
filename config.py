"""
Configuration module for News Recommendation Engine
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class Config:
    """Configuration class to manage all settings"""
    
    # Elasticsearch Configuration
    ES_CLOUD_ID = os.getenv('ES_CLOUD_ID')
    ES_API_KEY = os.getenv('ES_API_KEY')
    ES_BASE_URL = os.getenv('ES_BASE_URL')
    
    # RSS Feed Configuration
    RSS_URL = os.getenv('RSS_URL', 'https://www.abs-cbn.com/rss/mobile/latest-news')
    CRAWL_INTERVAL = int(os.getenv('CRAWL_INTERVAL', 300))
    
    # Index Names
    NEWS_INDEX = 'news_recommendation_hybrid'
    USER_PROFILE_INDEX = 'user_profiles_dynamic'
    SEARCH_HISTORY_INDEX = 'search_history'
    
    # Inference Configuration
    ELSER_INFERENCE_ID = '.elser-2-elasticsearch'
    MULTILINGUAL_INFERENCE_ID = '.multilingual-e5-small-elasticsearch'
    RERANK_INFERENCE_ID = '.rerank-v1-elasticsearch'
    CLAUDE_INFERENCE_ID = 'claude-completions'
    
    # Ingest Pipeline Name
    NEWS_INGEST_PIPELINE = 'news-vectorization-pipeline'
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        required_vars = ['ES_CLOUD_ID', 'ES_API_KEY', 'ES_BASE_URL']
        missing_vars = []
        
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        return True
    
    @classmethod
    def get_elasticsearch_config(cls) -> Dict[str, Any]:
        """Get Elasticsearch connection configuration"""
        return {
            'cloud_id': cls.ES_CLOUD_ID,
            'api_key': cls.ES_API_KEY,
            'request_timeout': 30,
            'retry_on_timeout': True,
            'max_retries': 3
        }
