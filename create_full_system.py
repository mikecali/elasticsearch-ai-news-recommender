#!/usr/bin/env python3
"""
Complete system setup script for News Recommendation Engine
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
    print("🚀 Creating complete News Recommendation Engine system...")
    
    # 1. Enhanced elasticsearch_client.py with full features
    es_client_content = '''"""
Enhanced Elasticsearch client with complete ML vectorization support
"""

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError
import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from config import Config

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class ElasticsearchClient:
    """Enhanced Elasticsearch client with full ML stack"""
    
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
        """Create enhanced ingest pipeline for complete vectorization"""
        try:
            pipeline = {
                "description": "Complete News articles ML vectorization pipeline",
                "processors": [
                    # Create semantic content field for ELSER
                    {
                        "set": {
                            "field": "semantic_content",
                            "value": "{{title}} {{description}} {{content}}"
                        }
                    },
                    # ELSER sparse embedding
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
                    # Multilingual dense embedding for title
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
                    },
                    # Multilingual dense embedding for description
                    {
                        "inference": {
                            "input_output": {
                                "input_field": "description",
                                "output_field": "ml.inference.description_embedding"
                            },
                            "model_id": Config.MULTILINGUAL_INFERENCE_ID,
                            "on_failure": [
                                {
                                    "append": {
                                        "field": "_ingest_errors",
                                        "value": "Failed to generate description embedding: {{ _ingest.on_failure_message }}"
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
                logger.info("Deleted existing ingest pipeline")
            except:
                pass
            
            # Create new pipeline
            self.client.ingest.put_pipeline(
                id=Config.NEWS_INGEST_PIPELINE,
                body=pipeline
            )
            logger.info(f"✅ Created enhanced ingest pipeline: {Config.NEWS_INGEST_PIPELINE}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating ingest pipeline: {str(e)}")
            return False
    
    def create_news_index(self) -> bool:
        """Create news index with complete ML mapping"""
        try:
            mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "url": {"type": "keyword"},
                        "description": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "semantic_content": {
                            "type": "text"
                        },
                        "published_date": {"type": "date"},
                        "categories": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "crawled_at": {"type": "date"},
                        "language": {"type": "keyword"},
                        
                        # ML inference results
                        "ml": {
                            "properties": {
                                "inference": {
                                    "properties": {
                                        # ELSER sparse embedding
                                        "semantic_content": {
                                            "type": "sparse_vector"
                                        },
                                        # Multilingual dense embeddings
                                        "title_embedding": {
                                            "type": "dense_vector",
                                            "dims": 384,
                                            "index": True,
                                            "similarity": "cosine",
                                            "index_options": {
                                                "type": "int8_hnsw",
                                                "m": 16,
                                                "ef_construction": 100
                                            }
                                        },
                                        "description_embedding": {
                                            "type": "dense_vector",
                                            "dims": 384,
                                            "index": True,
                                            "similarity": "cosine",
                                            "index_options": {
                                                "type": "int8_hnsw",
                                                "m": 16,
                                                "ef_construction": 100
                                            }
                                        },
                                        "model_id": {
                                            "type": "text",
                                            "fields": {
                                                "keyword": {
                                                    "type": "keyword",
                                                    "ignore_above": 256
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "default": {"type": "standard"}
                        }
                    }
                }
            }
            
            # Update existing index mapping (don't delete if it has data)
            try:
                if self.client.indices.exists(index=Config.NEWS_INDEX):
                    # Just update the mapping
                    self.client.indices.put_mapping(
                        index=Config.NEWS_INDEX,
                        body=mapping["mappings"]
                    )
                    logger.info(f"✅ Updated mapping for existing index: {Config.NEWS_INDEX}")
                else:
                    # Create fresh index
                    self.client.indices.create(index=Config.NEWS_INDEX, body=mapping)
                    logger.info(f"✅ Created news index with ML mapping: {Config.NEWS_INDEX}")
                
                return True
                
            except Exception as e:
                logger.warning(f"Mapping update failed, recreating index: {str(e)}")
                # If mapping update fails, recreate
                if self.client.indices.exists(index=Config.NEWS_INDEX):
                    self.client.indices.delete(index=Config.NEWS_INDEX)
                self.client.indices.create(index=Config.NEWS_INDEX, body=mapping)
                logger.info(f"✅ Recreated news index: {Config.NEWS_INDEX}")
                return True
            
        except Exception as e:
            logger.error(f"Error creating news index: {str(e)}")
            return False
    
    def create_user_profile_index(self) -> bool:
        """Create enhanced user profiles index"""
        try:
            mapping = {
                "mappings": {
                    "properties": {
                        "user_id": {"type": "keyword"},
                        "name": {"type": "text"},
                        "interests": {"type": "keyword"},
                        "preferences": {
                            "type": "object",
                            "properties": {
                                "categories": {"type": "keyword"},
                                "keywords": {"type": "keyword"},
                                "sources": {"type": "keyword"}
                            }
                        },
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
            
            if not self.client.indices.exists(index=Config.USER_PROFILE_INDEX):
                self.client.indices.create(index=Config.USER_PROFILE_INDEX, body=mapping)
                logger.info(f"✅ Created user profile index: {Config.USER_PROFILE_INDEX}")
            else:
                logger.info(f"✅ User profile index already exists: {Config.USER_PROFILE_INDEX}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating user profile index: {str(e)}")
            return False
    
    def setup_indices_and_pipeline(self) -> bool:
        """Setup all indices and enhanced ingest pipeline"""
        try:
            logger.info("🚀 Setting up complete Elasticsearch indices and ML pipeline...")
            
            # Create enhanced ingest pipeline first
            if not self.create_news_ingest_pipeline():
                logger.error("Failed to create ingest pipeline")
                return False
            
            # Create/update indices
            if not self.create_news_index():
                logger.error("Failed to create news index")
                return False
            
            if not self.create_user_profile_index():
                logger.error("Failed to create user profile index")
                return False
            
            logger.info("✅ Successfully setup all indices and enhanced ML pipeline")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up indices: {str(e)}")
            return False
    
    def index_articles_with_vectorization(self, articles: List[Dict[str, Any]]) -> bool:
        """Index articles with complete ML vectorization"""
        try:
            if not articles:
                logger.warning("No articles to index")
                return True
            
            logger.info(f"🔄 Indexing {len(articles)} articles with complete ML vectorization...")
            
            successful = 0
            failed_articles = []
            
            for article in articles:
                try:
                    # Clean and prepare article
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
                    
                    # Ensure we have content for vectorization
                    if not cleaned_article["description"]:
                        cleaned_article["description"] = cleaned_article["title"]
                    
                    if not cleaned_article["content"]:
                        cleaned_article["content"] = cleaned_article["description"]
                    
                    # Index with ML pipeline
                    response = self.client.index(
                        index=Config.NEWS_INDEX,
                        id=cleaned_article["id"],
                        body=cleaned_article,
                        pipeline=Config.NEWS_INGEST_PIPELINE
                    )
                    
                    if response.get("result") in ["created", "updated"]:
                        successful += 1
                        logger.debug(f"✅ Indexed with ML: {cleaned_article['id']}")
                    else:
                        failed_articles.append((cleaned_article["id"], f"Unexpected result: {response.get('result')}"))
                        
                except Exception as e:
                    failed_articles.append((article.get("id", "unknown"), str(e)))
                    logger.warning(f"❌ Failed to index article {article.get('id', 'unknown')}: {str(e)}")
            
            logger.info(f"✅ Successfully indexed {successful} articles with complete ML vectorization")
            
            if failed_articles:
                logger.error(f"❌ Failed to index {len(failed_articles)} articles")
                for article_id, error in failed_articles[:3]:
                    logger.error(f"   - {article_id}: {error}")
            
            # Verify vectorization
            if successful > 0:
                self._verify_vectorization()
            
            return len(failed_articles) == 0
            
        except Exception as e:
            logger.error(f"Error indexing articles: {str(e)}")
            return False
    
    def _verify_vectorization(self) -> None:
        """Verify that complete ML vectorization worked"""
        try:
            import time
            time.sleep(3)  # Wait for indexing to complete
            
            # Check for documents with ML vectors
            response = self.client.search(
                index=Config.NEWS_INDEX,
                body={
                    "size": 1,
                    "query": {"match_all": {}},
                    "_source": ["title", "ml.inference"]
                }
            )
            
            if response["hits"]["total"]["value"] > 0:
                hit = response["hits"]["hits"][0]["_source"]
                ml_inference = hit.get("ml", {}).get("inference", {})
                
                logger.info("🔍 Complete ML Vectorization verification:")
                logger.info(f"   - ELSER sparse embedding: {'✅' if 'semantic_content' in ml_inference else '❌'}")
                logger.info(f"   - Title dense embedding: {'✅' if 'title_embedding' in ml_inference else '❌'}")
                logger.info(f"   - Description dense embedding: {'✅' if 'description_embedding' in ml_inference else '❌'}")
                
                # Show vector details
                if 'title_embedding' in ml_inference:
                    vector_size = len(ml_inference['title_embedding'])
                    logger.info(f"   - Vector dimensions: {vector_size}")
                
            else:
                logger.warning("⚠️  No documents found after indexing")
                
        except Exception as e:
            logger.warning(f"Error verifying vectorization: {str(e)}")
    
    def hybrid_search(self, query: str, user_interests: List[str] = None, size: int = 10) -> List[Dict[str, Any]]:
        """Perform advanced hybrid search using complete ML stack"""
        try:
            logger.info(f"🔍 Performing advanced hybrid search for: '{query}'")
            
            # Build comprehensive hybrid search query
            search_body = {
                "size": size * 2,  # Get more for reranking
                "query": {
                    "bool": {
                        "should": [
                            # ELSER semantic search (highest weight)
                            {
                                "sparse_vector": {
                                    "field": "ml.inference.semantic_content",
                                    "inference_id": Config.ELSER_INFERENCE_ID,
                                    "query": query,
                                    "boost": 3.0
                                }
                            },
                            # Dense vector search on title
                            {
                                "knn": {
                                    "field": "ml.inference.title_embedding",
                                    "query_vector_builder": {
                                        "text_embedding": {
                                            "model_id": Config.MULTILINGUAL_INFERENCE_ID,
                                            "model_text": query
                                        }
                                    },
                                    "k": 50,
                                    "num_candidates": 100,
                                    "boost": 2.0
                                }
                            },
                            # Dense vector search on description
                            {
                                "knn": {
                                    "field": "ml.inference.description_embedding",
                                    "query_vector_builder": {
                                        "text_embedding": {
                                            "model_id": Config.MULTILINGUAL_INFERENCE_ID,
                                            "model_text": query
                                        }
                                    },
                                    "k": 50,
                                    "num_candidates": 100,
                                    "boost": 1.5
                                }
                            },
                            # Keyword search fallback
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title^3", "description^2", "content"],
                                    "type": "best_fields",
                                    "boost": 1.0
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"published_date": {"order": "desc"}}
                ]
            }
            
            # Add user interest filtering if provided
            if user_interests:
                search_body["query"]["bool"]["filter"] = [
                    {"terms": {"categories": user_interests}}
                ]
            
            # Execute search
            response = self.client.search(index=Config.NEWS_INDEX, body=search_body)
            
            # Prepare articles for reranking
            articles = []
            for hit in response["hits"]["hits"]:
                article = hit["_source"]
                article["_score"] = hit["_score"]
                articles.append(article)
            
            # Apply reranking if we have results
            if articles:
                reranked_articles = self._rerank_articles(query, articles[:size])
                logger.info(f"✅ Found {len(reranked_articles)} articles via complete hybrid search + reranking")
                return reranked_articles
            else:
                logger.info("No articles found in hybrid search")
                return []
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return self._fallback_search(query, size)
    
    def _rerank_articles(self, query: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply intelligent reranking to improve relevance"""
        try:
            # Prepare documents for reranking
            docs = []
            for article in articles:
                doc_text = f"{article.get('title', '')} {article.get('description', '')}"
                docs.append(doc_text)
            
            # Call rerank API
            rerank_response = self.client.inference.inference(
                inference_id=Config.RERANK_INFERENCE_ID,
                body={
                    "query": query,
                    "input": docs
                }
            )
            
            # Process rerank results
            rerank_results = rerank_response.get("rerank", [])
            
            # Apply rerank scores
            for i, result in enumerate(rerank_results):
                if i < len(articles):
                    articles[i]["_rerank_score"] = result.get("relevance_score", 0.0)
            
            # Sort by rerank score
            articles.sort(key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
            
            logger.info("✅ Applied intelligent reranking to results")
            return articles
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {str(e)}")
            return articles
    
    def _fallback_search(self, query: str, size: int = 10) -> List[Dict[str, Any]]:
        """Enhanced fallback to keyword search"""
        try:
            logger.info("Using enhanced fallback keyword search")
            
            search_body = {
                "size": size,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match_phrase": {
                                    "title": {
                                        "query": query,
                                        "boost": 5.0
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title^3", "description^2", "content"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO",
                                    "boost": 2.0
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"published_date": {"order": "desc"}}
                ]
            }
            
            response = self.client.search(index=Config.NEWS_INDEX, body=search_body)
            
            articles = []
            for hit in response["hits"]["hits"]:
                article = hit["_source"]
                article["_score"] = hit["_score"]
                article["_fallback_used"] = True
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Even fallback search failed: {str(e)}")
            return []
    
    def get_vectorization_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about vectorization"""
        try:
            total_docs = self.client.count(index=Config.NEWS_INDEX)["count"]
            
            # Check how many have ML vectors
            ml_docs = self.client.count(
                index=Config.NEWS_INDEX,
                body={
                    "query": {
                        "exists": {"field": "ml.inference.semantic_content"}
                    }
                }
            )["count"]
            
            # Check dense vector coverage
            dense_docs = self.client.count(
                index=Config.NEWS_INDEX,
                body={
                    "query": {
                        "exists": {"field": "ml.inference.title_embedding"}
                    }
                }
            )["count"]
            
            return {
                "total_documents": total_docs,
                "documents_with_ml_vectors": ml_docs,
                "documents_with_dense_vectors": dense_docs,
                "vectorization_coverage": (ml_docs / max(total_docs, 1)) * 100,
                "dense_vector_coverage": (dense_docs / max(total_docs, 1)) * 100,
                "index_name": Config.NEWS_INDEX,
                "pipeline_name": Config.NEWS_INGEST_PIPELINE
            }
            
        except Exception as e:
            logger.error(f"Error getting vectorization stats: {str(e)}")
            return {}
'''
    
    # Save the enhanced elasticsearch_client.py
    if create_file('elasticsearch_client.py', es_client_content):
        print("✅ Enhanced elasticsearch_client.py with complete ML features")
    
    # 2. Create rss_crawler.py
    rss_crawler_content = '''"""
RSS Crawler with enhanced content extraction for ML vectorization
"""

import feedparser
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import ssl
import urllib3
from urllib.request import urlopen, Request

from config import Config

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class RSSCrawler:
    """Enhanced RSS Crawler optimized for ML vectorization"""
    
    def __init__(self, rss_url: str = None):
        """Initialize RSS Crawler"""
        self.rss_url = rss_url or Config.RSS_URL
        self.session = requests.Session()
        
        # Browser-like headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        self.session.verify = False
    
    def fetch_rss_feed(self) -> Optional[feedparser.FeedParserDict]:
        """Fetch and parse RSS feed with multiple fallback methods"""
        try:
            logger.info(f"📡 Fetching RSS feed from: {self.rss_url}")
            
            methods = [
                self._fetch_with_requests,
                self._fetch_with_urllib,
                self._fetch_with_feedparser
            ]
            
            for method in methods:
                try:
                    feed = method()
                    if feed and feed.entries:
                        logger.info(f"✅ Successfully fetched {len(feed.entries)} entries using {method.__name__}")
                        return feed
                except Exception as e:
                    logger.warning(f"{method.__name__} failed: {str(e)}")
                    continue
            
            logger.error("❌ All RSS fetching methods failed")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed: {str(e)}")
            return None
    
    def _fetch_with_requests(self):
        """Fetch using requests session"""
        response = self.session.get(self.rss_url, timeout=30)
        response.raise_for_status()
        return feedparser.parse(response.content)
    
    def _fetch_with_urllib(self):
        """Fetch using urllib with SSL context"""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        req = Request(self.rss_url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
        
        with urlopen(req, context=ssl_context, timeout=30) as response:
            content = response.read()
        
        return feedparser.parse(content)
    
    def _fetch_with_feedparser(self):
        """Fetch using feedparser directly"""
        feedparser.USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        return feedparser.parse(self.rss_url)
    
    def extract_rich_content(self, description: str) -> str:
        """Extract clean, rich text content optimized for ML vectorization"""
        if not description:
            return ""
        
        try:
            soup = BeautifulSoup(description, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content with proper spacing
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace and normalize
            lines = [line.strip() for line in text.split('\\n') if line.strip()]
            cleaned_text = ' '.join(lines)
            
            # Remove excessive whitespace
            import re
            cleaned_text = re.sub(r'\\s+', ' ', cleaned_text)
            
            return cleaned_text[:2000]  # Optimal length for vectorization
            
        except Exception as e:
            logger.warning(f"Error extracting content: {str(e)}")
            return description[:2000] if description else ""
    
    def generate_article_id(self, title: str, url: str) -> str:
        """Generate unique ID for article"""
        content = f"{title}{url}".encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def infer_categories(self, title: str, description: str) -> List[str]:
        """Enhanced category inference for better search relevance"""
        content = f"{title} {description}".lower()
        categories = []
        
        # Enhanced category detection with more keywords
        category_keywords = {
            'sports': ['basketball', 'football', 'volleyball', 'sports', 'game', 'player', 'team', 'pba', 'nba', 'uaap', 'ncaa', 'match', 'tournament', 'championship', 'league', 'athletics', 'olympics'],
            'politics': ['government', 'president', 'senate', 'congress', 'mayor', 'election', 'policy', 'duterte', 'marcos', 'robredo', 'political', 'vote', 'campaign', 'official', 'cabinet', 'minister', 'governance'],
            'entertainment': ['actor', 'actress', 'movie', 'film', 'concert', 'music', 'celebrity', 'showbiz', 'entertainment', 'artista', 'singer', 'dancing', 'talent', 'drama', 'comedy', 'theater'],
            'food': ['restaurant', 'food', 'cooking', 'chef', 'cuisine', 'recipe', 'dining', 'meal', 'eat', 'taste', 'flavor', 'kitchen', 'culinary', 'gastronomy'],
            'technology': ['tech', 'digital', 'app', 'software', 'computer', 'internet', 'smartphone', 'gadget', 'innovation', 'ai', 'online', 'cyber', 'data', 'programming'],
            'health': ['health', 'medical', 'doctor', 'hospital', 'medicine', 'virus', 'covid', 'vaccine', 'disease', 'treatment', 'wellness', 'fitness', 'nutrition'],
            'business': ['business', 'company', 'economy', 'financial', 'market', 'trade', 'industry', 'corporate', 'investment', 'profit', 'stock', 'startup', 'entrepreneur'],
            'education': ['school', 'university', 'student', 'teacher', 'education', 'learning', 'academic', 'graduation', 'scholarship', 'college', 'research'],
            'crime': ['crime', 'police', 'arrest', 'investigation', 'murder', 'robbery', 'drugs', 'illegal', 'court', 'trial', 'justice', 'law enforcement'],
            'weather': ['weather', 'storm', 'typhoon', 'rain', 'flood', 'climate', 'temperature', 'forecast', 'pagasa', 'meteorology', 'hurricane'],
            'international': ['international', 'global', 'world', 'foreign', 'abroad', 'overseas', 'diplomatic', 'embassy', 'summit', 'treaty'],
            'local': ['local', 'community', 'barangay', 'city', 'town', 'municipal', 'regional', 'provincial', 'neighborhood']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in content for keyword in keywords):
                categories.append(category)
        
        # Default category if none found
        if not categories:
            categories = ['news']
        
        return categories[:4]  # Limit to 4 categories max
    
    def parse_article(self, entry: feedparser.FeedParserDict) -> Optional[Dict[str, Any]]:
        """Parse RSS entry with enhanced content extraction"""
        try:
            # Extract basic information
            title = entry.get('title', '').strip()
            link = entry.get('link', '').strip()
            
            if not title or not link:
                logger.warning("Article missing title or link, skipping")
                return None
            
            # Get description/summary with rich extraction
            raw_description = entry.get('description', '').strip()
            raw_summary = entry.get('summary', raw_description).strip()
            
            # Extract rich content
            description = self.extract_rich_content(raw_description)
            summary = self.extract_rich_content(raw_summary)
            
            # Ensure we have meaningful content
            if not description and not summary:
                description = f"News article about {title}"
                summary = description
            elif not description:
                description = summary
            elif not summary:
                summary = description[:500]  # Shorter summary
            
            # Use description as content for now (can be enhanced to fetch full article)
            content = description
            
            # Parse publication date
            pub_date = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6])
            else:
                pub_date = datetime.now()
            
            # Enhanced category inference
            categories = self.infer_categories(title, description)
            
            # Generate unique ID
            article_id = self.generate_article_id(title, link)
            
            article = {
                'id': article_id,
                'title': title,
                'url': link,
                'description': description,
                'content': content,
                'published_date': pub_date.isoformat(),
                'categories': categories,
                'source': 'ABS-CBN',
                'crawled_at': datetime.now().isoformat(),
                'language': 'en'
            }
            
            return article
            
        except Exception as e:
            logger.error(f"Error parsing article entry: {str(e)}")
            return None
    
    def crawl_news(self) -> List[Dict[str, Any]]:
        """Crawl news articles optimized for complete ML vectorization"""
        try:
            feed = self.fetch_rss_feed()
            if not feed:
                logger.warning("⚠️  No feed data received, using enhanced mock articles...")
                return self.create_enhanced_mock_articles()
            
            articles = []
            logger.info(f"🔄 Processing {len(feed.entries)} entries from RSS feed")
            
            for i, entry in enumerate(feed.entries):
                logger.debug(f"Processing entry {i+1}/{len(feed.entries)}")
                
                article = self.parse_article(entry)
                if article:
                    articles.append(article)
                    logger.debug(f"✅ Parsed: {article['title'][:50]}...")
                else:
                    logger.warning(f"❌ Failed to parse entry {i+1}")
            
            logger.info(f"✅ Successfully parsed {len(articles)} articles")
            
            # Log content quality sample
            if articles:
                sample = articles[0]
                logger.info(f"📊 Content quality sample:")
                logger.info(f"   Title: {len(sample['title'])} chars")
                logger.info(f"   Description: {len(sample['description'])} chars")
                logger.info(f"   Categories: {sample['categories']}")
            
            return articles
            
        except Exception as e:
            logger.error(f"Error during news crawling: {str(e)}")
            return self.create_enhanced_mock_articles()
    
    def create_enhanced_mock_articles(self) -> List[Dict[str, Any]]:
        """Create comprehensive mock articles for demonstration"""
        logger.info("🎭 Creating enhanced mock articles for complete ML demonstration...")
        
        mock_articles = [
            {
                'id': 'demo_sports_basketball_1',
                'title': 'Ginebra Dominates TNT in PBA Finals Game 6, Captures Championship Crown',
                'url': 'https://news.abs-cbn.com/sports/2025/01/15/ginebra-dominates-tnt-pba-finals',
                'description': 'Barangay Ginebra San Miguel delivered a masterful performance against TNT Tropang Giga in Game 6 of the PBA Finals, winning 92-78 to claim their first championship in three years. Scottie Thompson led the Gin Kings with an outstanding 28 points, 14 rebounds, and 8 assists, while Japeth Aguilar contributed 22 points and 10 rebounds. Coach Tim Cone praised his team\\'s exceptional teamwork, defensive intensity, and clutch performances throughout the playoffs. The victory marks Ginebra\\'s ninth PBA championship title and establishes them as one of the most successful franchises in league history.',
                'content': 'The PBA Finals concluded with a spectacular display of Philippine basketball excellence. Ginebra\\'s championship run included victories over formidable opponents like Magnolia Hotshots and San Miguel Beermen. The team\\'s balanced scoring attack, featuring contributions from both veteran stars and emerging talents, proved decisive in the championship series. Fans celebrated throughout Metro Manila as the Gin Kings secured their place in PBA history.',
                'published_date': datetime.now().isoformat(),
                'categories': ['sports', 'basketball', 'pba', 'championship'],
                'source': 'ABS-CBN',
                'crawled_at': datetime.now().isoformat(),
                'language': 'en'
            },
            {
                'id': 'demo_politics_senate_1',
                'title': 'Senate Unanimously Passes Landmark Universal Healthcare Reform Act',
                'url': 'https://news.abs-cbn.com/news/2025/01/15/senate-healthcare-reform-unanimous',
                'description': 'The Philippine Senate achieved a historic milestone by unanimously passing the Universal Healthcare Reform Act, comprehensive legislation that will revolutionize medical services for millions of Filipino families. The groundbreaking bill allocates 250 billion pesos over five years for healthcare infrastructure development, including construction of rural hospitals, establishment of telemedicine networks, and procurement of advanced medical equipment. Senator Grace Poe, the bill\\'s principal author, emphasized that this legislation will guarantee free basic healthcare services to all Filipinos regardless of economic status.',
                'content': 'This transformative healthcare reform represents the most significant advancement in Philippine healthcare policy in decades. The legislation addresses critical gaps in medical services, particularly in remote and underserved areas. Healthcare advocacy groups, medical professionals, and patient rights organizations have praised the bill as a major step toward achieving universal healthcare coverage and improving health outcomes nationwide.',
                'published_date': datetime.now().isoformat(),
                'categories': ['politics', 'government', 'health', 'legislation'],
                'source': 'ABS-CBN',
                'crawled_at': datetime.now().isoformat(),
                'language': 'en'
            },
            {
                'id': 'demo_food_fusion_1',
                'title': 'Celebrity Chef Launches Revolutionary Filipino-Korean Fusion Concept in BGC',
                'url': 'https://news.abs-cbn.com/lifestyle/2025/01/15/filipino-korean-fusion-restaurant',
                'description': 'Award-winning celebrity chef Carlos Villaflor has unveiled "Seoul Manila Kitchen," an innovative Filipino-Korean fusion restaurant in Bonifacio Global City that promises to redefine cross-cultural dining experiences. The restaurant features an extraordinary menu combining traditional Filipino ingredients like ube, coconut, and longganisa with authentic Korean cooking techniques including ramen preparation, kimchi fermentation, and bulgogi grilling. Chef Villaflor, who completed intensive culinary training in Seoul for two years, aims to create a bridge between Filipino and Korean culinary traditions.',
                'content': 'The fusion concept represents a growing trend in Philippine dining where international flavors meet indigenous ingredients. The restaurant\\'s interior design seamlessly blends modern Korean aesthetics with traditional Filipino elements, creating an atmosphere that celebrates both cultures. Opening week features include live cooking demonstrations, Korean-Filipino cultural performances, and special tasting menus designed to showcase the unique flavor combinations.',
                'published_date': datetime.now().isoformat(),
                'categories': ['food', 'lifestyle', 'business', 'culture'],
                'source': 'ABS-CBN',
                'crawled_at': datetime.now().isoformat(),
                'language': 'en'
            },
            {
                'id': 'demo_technology_digital_1',
                'title': 'Philippines Launches Comprehensive National Digital Identity System',
                'url': 'https://news.abs-cbn.com/news/2025/01/15/philippines-digital-identity-launch',
                'description': 'The Department of Information and Communications Technology officially launched the Philippine Digital Identity (PhilID) system, a groundbreaking platform enabling citizens to access government services online through secure digital authentication. The sophisticated system integrates birth certificates, tax records, social security numbers, driver\\'s licenses, and other official documents into a unified digital identity framework. Citizens can now apply for permits, licenses, certificates, and various government services online without visiting physical government offices.',
                'content': 'This comprehensive digital transformation initiative represents a major leap forward in modernizing government service delivery. The PhilID system utilizes advanced blockchain technology and biometric authentication to ensure security and prevent identity fraud. The platform will be gradually implemented across all government agencies over the next two years, with privacy advocates praising the robust security measures and data protection protocols.',
                'published_date': datetime.now().isoformat(),
                'categories': ['technology', 'government', 'digital', 'innovation'],
                'source': 'ABS-CBN',
                'crawled_at': datetime.now().isoformat(),
                'language': 'en'
            }
        ]
        
        logger.info(f"✅ Created {len(mock_articles)} enhanced mock articles")
        
        # Log content statistics
        total_desc_chars = sum(len(article['description']) for article in mock_articles)
        avg_desc_length = total_desc_chars // len(mock_articles)
        logger.info(f"📊 Average description length: {avg_desc_length} characters")
        
        return mock_articles
    
    def get_feed_info(self) -> Dict[str, Any]:
        """Get comprehensive RSS feed metadata"""
        try:
            feed = self.fetch_rss_feed()
            if not feed:
                return {
                    'title': 'ABS-CBN News (Enhanced Demo Mode)',
                    'description': 'Latest news with enhanced content extraction for complete ML vectorization',
                    'link': self.rss_url,
                    'language': 'en',
                    'updated': datetime.now().isoformat(),
                    'total_entries': 4
                }
            
            return {
                'title': getattr(feed.feed, 'title', 'ABS-CBN News'),
                'description': getattr(feed.feed, 'description', 'Latest news'),
                'link': getattr(feed.feed, 'link', self.rss_url),
                'language': getattr(feed.feed, 'language', 'en'),
                'updated': getattr(feed.feed, 'updated', datetime.now().isoformat()),
                'total_entries': len(feed.entries)
            }
            
        except Exception as e:
            logger.error(f"Error getting feed info: {str(e)}")
            return {
                'title': 'ABS-CBN News (Error)',
                'description': 'RSS feed currently unavailable',
                'link': self.rss_url,
                'language': 'en',
                'updated': datetime.now().isoformat(),
                'total_entries': 0
            }
'''
    
    if create_file('rss_crawler.py', rss_crawler_content):
        print("✅ Complete RSS crawler with enhanced content extraction")
    
    print(f"\n🎉 Complete system files created successfully!")
    print("\n🚀 Next steps:")
    print("1. Update your system: python main.py --setup")
    print("2. Crawl enhanced content: python main.py --crawl")
    print("3. Create remaining components (user_profiles.py, recommendation_engine.py, web_app.py)")

if __name__ == "__main__":
    main()