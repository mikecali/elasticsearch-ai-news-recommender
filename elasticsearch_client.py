"""
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
