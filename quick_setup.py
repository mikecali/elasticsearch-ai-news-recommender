#!/usr/bin/env python3
"""
Quick setup script to create essential files for News Recommendation Engine
"""

import os

def create_file(filename, content):
    """Create a file with given content"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to create {filename}: {e}")
        return False

def main():
    print("🚀 Creating essential files for News Recommendation Engine...")
    
    # 1. Create .env file
    env_content = """# Elasticsearch Configuration
ES_CLOUD_ID=Westpac_ESQL_Demo:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyQ5NzM1NDFhMWRjYWE0NmEzODEwOGQ2OTM3ZjY5MDhjNyRjYjg2NGZiMDM4MDc0YjE1YmFkMTBlZDcxZDhjMDUwMw==
ES_API_KEY=VmlyWno1Y0JST2J2NjV5UFl3eVQ6eUpyZXJRSldFR2tUdk1laFgxbTlkUQ==
ES_BASE_URL=https://westpac-es-ql-demo.es.us-central1.gcp.cloud.es.io

# RSS Feed Configuration  
RSS_URL=https://www.abs-cbn.com/rss/mobile/latest-news
CRAWL_INTERVAL=300

# Logging Configuration
LOG_LEVEL=INFO
"""
    create_file('.env', env_content)
    
    # 2. Create requirements.txt
    requirements_content = """elasticsearch>=8.11.0
feedparser>=6.0.10
requests>=2.31.0
beautifulsoup4>=4.12.2
urllib3>=2.0.0
flask>=3.0.0
flask-cors>=4.0.0
python-dotenv>=1.0.0
python-dateutil>=2.8.2
schedule>=1.2.0
"""
    create_file('requirements.txt', requirements_content)
    
    # 3. Create config.py
    config_content = '''"""
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
'''
    create_file('config.py', config_content)
    
    # 4. Create elasticsearch_client.py (simplified version)
    es_client_content = '''"""
Elasticsearch client with ML vectorization support
"""

from elasticsearch import Elasticsearch
import logging
from typing import Dict, Any, List
from datetime import datetime
from config import Config

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class ElasticsearchClient:
    """Elasticsearch client with ML vectorization"""
    
    def __init__(self):
        """Initialize Elasticsearch client"""
        try:
            Config.validate_config()
            self.client = Elasticsearch(**Config.get_elasticsearch_config())
            
            if self.client.ping():
                logger.info("✅ Successfully connected to Elasticsearch")
            else:
                raise ConnectionError("Failed to connect to Elasticsearch")
                
        except Exception as e:
            logger.error(f"Error initializing Elasticsearch client: {str(e)}")
            raise
    
    def create_news_ingest_pipeline(self) -> bool:
        """Create ingest pipeline for vectorization"""
        try:
            pipeline = {
                "description": "News articles ML vectorization pipeline",
                "processors": [
                    {
                        "set": {
                            "field": "semantic_content",
                            "value": "{{title}} {{description}} {{content}}"
                        }
                    },
                    {
                        "inference": {
                            "input_output": {
                                "input_field": "semantic_content",
                                "output_field": "ml.inference.semantic_content"
                            },
                            "model_id": Config.ELSER_INFERENCE_ID,
                            "on_failure": [
                                {
                                    "append": {
                                        "field": "_ingest_errors",
                                        "value": "Failed to generate ELSER embedding: {{ _ingest.on_failure_message }}"
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "inference": {
                            "input_output": {
                                "input_field": "title",
                                "output_field": "ml.inference.title_embedding"
                            },
                            "model_id": Config.MULTILINGUAL_INFERENCE_ID,
                            "on_failure": [
                                {
                                    "append": {
                                        "field": "_ingest_errors",
                                        "value": "Failed to generate title embedding: {{ _ingest.on_failure_message }}"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
            
            # Delete existing pipeline if exists
            try:
                self.client.ingest.delete_pipeline(id=Config.NEWS_INGEST_PIPELINE)
            except:
                pass
            
            # Create new pipeline
            self.client.ingest.put_pipeline(
                id=Config.NEWS_INGEST_PIPELINE,
                body=pipeline
            )
            logger.info(f"✅ Created ingest pipeline: {Config.NEWS_INGEST_PIPELINE}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating ingest pipeline: {str(e)}")
            return False
    
    def create_news_index(self) -> bool:
        """Create news index with ML mapping"""
        try:
            mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "title": {"type": "text", "analyzer": "standard"},
                        "url": {"type": "keyword"},
                        "description": {"type": "text", "analyzer": "standard"},
                        "content": {"type": "text", "analyzer": "standard"},
                        "semantic_content": {"type": "text"},
                        "published_date": {"type": "date"},
                        "categories": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "crawled_at": {"type": "date"},
                        "language": {"type": "keyword"},
                        "ml": {
                            "properties": {
                                "inference": {
                                    "properties": {
                                        "semantic_content": {"type": "sparse_vector"},
                                        "title_embedding": {
                                            "type": "dense_vector",
                                            "dims": 384,
                                            "index": True,
                                            "similarity": "cosine"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            # Delete index if exists
            if self.client.indices.exists(index=Config.NEWS_INDEX):
                self.client.indices.delete(index=Config.NEWS_INDEX)
            
            # Create fresh index
            self.client.indices.create(index=Config.NEWS_INDEX, body=mapping)
            logger.info(f"✅ Created news index: {Config.NEWS_INDEX}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating news index: {str(e)}")
            return False
    
    def create_user_profile_index(self) -> bool:
        """Create user profiles index"""
        try:
            mapping = {
                "mappings": {
                    "properties": {
                        "user_id": {"type": "keyword"},
                        "name": {"type": "text"},
                        "interests": {"type": "keyword"},
                        "reading_history": {
                            "type": "nested",
                            "properties": {
                                "article_id": {"type": "keyword"},
                                "timestamp": {"type": "date"},
                                "engagement_score": {"type": "float"}
                            }
                        },
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"}
                    }
                }
            }
            
            if self.client.indices.exists(index=Config.USER_PROFILE_INDEX):
                self.client.indices.delete(index=Config.USER_PROFILE_INDEX)
            
            self.client.indices.create(index=Config.USER_PROFILE_INDEX, body=mapping)
            logger.info(f"✅ Created user profile index: {Config.USER_PROFILE_INDEX}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating user profile index: {str(e)}")
            return False
    
    def setup_indices_and_pipeline(self) -> bool:
        """Setup all indices and ingest pipeline"""
        try:
            logger.info("🚀 Setting up Elasticsearch indices and ML pipeline...")
            
            if not self.create_news_ingest_pipeline():
                return False
            
            if not self.create_news_index():
                return False
            
            if not self.create_user_profile_index():
                return False
            
            logger.info("✅ Successfully setup all indices and ML pipeline")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up indices: {str(e)}")
            return False
    
    def index_articles_with_vectorization(self, articles: List[Dict[str, Any]]) -> bool:
        """Index articles with ML vectorization"""
        try:
            if not articles:
                return True
            
            logger.info(f"🔄 Indexing {len(articles)} articles with ML vectorization...")
            successful = 0
            
            for article in articles:
                try:
                    cleaned_article = {
                        "id": str(article.get("id", "")),
                        "title": str(article.get("title", ""))[:1000],
                        "url": str(article.get("url", "")),
                        "description": str(article.get("description", ""))[:2000],
                        "content": str(article.get("content", ""))[:5000],
                        "published_date": article.get("published_date", datetime.now().isoformat()),
                        "categories": article.get("categories", []),
                        "source": str(article.get("source", "ABS-CBN")),
                        "crawled_at": datetime.now().isoformat(),
                        "language": str(article.get("language", "en"))
                    }
                    
                    if not cleaned_article["description"]:
                        cleaned_article["description"] = cleaned_article["title"]
                    
                    response = self.client.index(
                        index=Config.NEWS_INDEX,
                        id=cleaned_article["id"],
                        body=cleaned_article,
                        pipeline=Config.NEWS_INGEST_PIPELINE
                    )
                    
                    if response.get("result") in ["created", "updated"]:
                        successful += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to index article: {str(e)}")
            
            logger.info(f"✅ Successfully indexed {successful} articles")
            return successful > 0
            
        except Exception as e:
            logger.error(f"Error indexing articles: {str(e)}")
            return False
    
    def get_vectorization_stats(self) -> Dict[str, Any]:
        """Get vectorization statistics"""
        try:
            total_docs = self.client.count(index=Config.NEWS_INDEX)["count"]
            return {
                "total_documents": total_docs,
                "index_name": Config.NEWS_INDEX,
                "pipeline_name": Config.NEWS_INGEST_PIPELINE
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
'''
    create_file('elasticsearch_client.py', es_client_content)
    
    # 5. Create a simple test script
    test_content = '''#!/usr/bin/env python3
"""
Simple test script to verify setup
"""

def test_import():
    """Test that we can import the main components"""
    try:
        from config import Config
        print("✅ Config imported successfully")
        
        from elasticsearch_client import ElasticsearchClient
        print("✅ ElasticsearchClient imported successfully")
        
        # Test connection
        client = ElasticsearchClient()
        print("✅ Elasticsearch connection successful")
        
        print("\\n🎉 All imports and connections working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_import()
'''
    create_file('test_setup.py', test_content)
    
    print("\n✅ Essential files created!")
    print("\n🚀 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Test setup: python test_setup.py")
    print("3. If test passes, you can proceed with the full system")

if __name__ == "__main__":
    main()