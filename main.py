#!/usr/bin/env python3
"""
Complete main application for News Recommendation Engine with full ML stack
"""

import logging
import argparse
import time
from datetime import datetime
from typing import Dict, Any

from config import Config
from elasticsearch_client import ElasticsearchClient
from rss_crawler import RSSCrawler
from user_profiles import UserProfileManager
from recommendation_engine import NewsRecommendationEngine

# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsRecommendationApp:
    """Complete News Recommendation Application"""
    
    def __init__(self):
        """Initialize the complete application"""
        try:
            logger.info("🚀 Initializing Complete News Recommendation Engine...")
            
            # Initialize components
            self.es_client = ElasticsearchClient()
            self.crawler = RSSCrawler()
            self.user_manager = UserProfileManager(self.es_client)
            self.recommendation_engine = NewsRecommendationEngine(self.es_client, self.user_manager)
            
            logger.info("✅ Complete application initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing application: {str(e)}")
            raise
    
    def setup_system(self) -> bool:
        """Setup complete system with ML vectorization"""
        try:
            logger.info("🔧 Setting up complete system with ML Stack...")
            
            # Setup Elasticsearch indices and enhanced ingest pipeline
            if not self.es_client.setup_indices_and_pipeline():
                logger.error("❌ Failed to setup Elasticsearch indices and pipeline")
                return False
            
            # Create demo users if they don't exist
            existing_users = self.user_manager.get_all_users()
            if len(existing_users) < 3:
                logger.info("👥 Creating demo users...")
                user_ids = self.user_manager.create_demo_users()
                if not user_ids:
                    logger.error("❌ Failed to create demo users")
                    return False
                logger.info(f"✅ Created {len(user_ids)} demo users")
            else:
                logger.info(f"✅ Found {len(existing_users)} existing users")
            
            logger.info("✅ Complete system setup successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up system: {str(e)}")
            return False
    
    def crawl_and_index_news(self) -> int:
        """Crawl news and index with complete ML vectorization"""
        try:
            logger.info("📰 Starting enhanced news crawling and ML indexing...")
            
            # Crawl articles
            articles = self.crawler.crawl_news()
            if not articles:
                logger.warning("⚠️  No articles crawled")
                return 0
            
            logger.info(f"📄 Crawled {len(articles)} articles with rich content")
            
            # Index with complete ML vectorization
            success = self.es_client.index_articles_with_vectorization(articles)
            if success:
                logger.info(f"✅ Successfully indexed {len(articles)} articles with complete ML vectors")
                return len(articles)
            else:
                logger.error("❌ Failed to index articles")
                return 0
                
        except Exception as e:
            logger.error(f"❌ Error crawling and indexing news: {str(e)}")
            return 0
    
    def generate_recommendations_for_user(self, user_id: str) -> Dict[str, Any]:
        """Generate recommendations for a specific user"""
        try:
            logger.info(f"🎯 Generating complete ML recommendations for user {user_id}")
            
            recommendations = self.recommendation_engine.generate_personalized_recommendations(user_id)
            
            if "error" not in recommendations:
                num_recs = len(recommendations.get("recommendations", []))
                performance = recommendations.get("performance", {})
                logger.info(f"✅ Generated {num_recs} recommendations in {performance.get('total_time', 0):.2f}s")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {str(e)}")
            return {"error": str(e)}
    
    def generate_recommendations_for_all_users(self) -> Dict[str, Any]:
        """Generate recommendations for all users using complete ML stack"""
        try:
            logger.info("🎯 Generating complete ML recommendations for all users...")
            
            users = self.user_manager.get_all_users()
            all_recommendations = {}
            successful = 0
            
            for user in users:
                user_id = user["user_id"]
                user_name = user.get("name", "Unknown")
                
                try:
                    logger.info(f"Processing user: {user_name}")
                    recommendations = self.recommendation_engine.generate_personalized_recommendations(user_id)
                    all_recommendations[user_id] = recommendations
                    
                    if "error" not in recommendations:
                        successful += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to generate recommendations for {user_name}: {str(e)}")
                    all_recommendations[user_id] = {"error": str(e)}
            
            logger.info(f"✅ Generated recommendations for {successful}/{len(users)} users")
            return all_recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations for all users: {str(e)}")
            return {}
    
    def simulate_user_activity(self) -> bool:
        """Simulate comprehensive user reading activity"""
        try:
            logger.info("🎭 Simulating comprehensive user reading activity...")
            
            # Get all users
            users = self.user_manager.get_all_users()
            if not users:
                logger.warning("No users found for simulation")
                return False
            
            user_ids = [user["user_id"] for user in users]
            
            # Get recent articles
            try:
                response = self.es_client.client.search(
                    index=Config.NEWS_INDEX,
                    body={
                        "size": 20,
                        "query": {"match_all": {}},
                        "sort": [{"published_date": {"order": "desc"}}]
                    }
                )
                
                articles = [hit["_source"] for hit in response["hits"]["hits"]]
                
                if not articles:
                    logger.warning("No articles found for simulation")
                    return False
                
                # Simulate comprehensive interactions
                success = self.user_manager.simulate_reading_activity(user_ids, articles)
                
                if success:
                    logger.info("✅ Successfully simulated comprehensive user reading activity")
                
                return success
                
            except Exception as e:
                logger.warning(f"Error getting articles for simulation: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error simulating user activity: {str(e)}")
            return False
    
    def display_system_status(self) -> None:
        """Display comprehensive system status"""
        try:
            print("\n" + "="*80)
            print("🤖 COMPLETE NEWS RECOMMENDATION ENGINE - ML-POWERED SYSTEM STATUS")
            print("="*80)
            
            # Elasticsearch status
            if self.es_client.client.ping():
                print("✅ Elasticsearch: Connected")
            else:
                print("❌ Elasticsearch: Disconnected")
                return
            
            # Enhanced vectorization stats
            vectorization_stats = self.es_client.get_vectorization_stats()
            total_docs = vectorization_stats.get("total_documents", 0)
            ml_docs = vectorization_stats.get("documents_with_ml_vectors", 0)
            dense_docs = vectorization_stats.get("documents_with_dense_vectors", 0)
            coverage = vectorization_stats.get("vectorization_coverage", 0)
            dense_coverage = vectorization_stats.get("dense_vector_coverage", 0)
            
            print(f"📊 Articles Index: {total_docs} documents")
            print(f"🧠 ELSER Vectorization: {ml_docs} documents ({coverage:.1f}% coverage)")
            print(f"🔢 Dense Vectorization: {dense_docs} documents ({dense_coverage:.1f}% coverage)")
            
            # Index status
            indices = [Config.NEWS_INDEX, Config.USER_PROFILE_INDEX]
            for index in indices:
                try:
                    count = self.es_client.client.count(index=index)["count"]
                    print(f"✅ {index}: {count} documents")
                except Exception as e:
                    print(f"❌ {index}: Error - {str(e)}")
            
            # Enhanced ingest pipeline status
            try:
                pipeline_response = self.es_client.client.ingest.get_pipeline(id=Config.NEWS_INGEST_PIPELINE)
                print(f"✅ Enhanced ML Pipeline: {Config.NEWS_INGEST_PIPELINE} active")
            except:
                print(f"❌ Enhanced ML Pipeline: {Config.NEWS_INGEST_PIPELINE} not found")
            
            # User status
            users = self.user_manager.get_all_users()
            print(f"👥 Active Users: {len(users)} profiles")
            for user in users:
                reading_count = len(user.get("reading_history", []))
                print(f"   - {user['name']}: {reading_count} articles read")
            
            # Complete ML Stack status
            print("\n🧠 Complete ML Technology Stack:")
            print(f"   - ELSER Semantic Search: {Config.ELSER_INFERENCE_ID}")
            print(f"   - Multilingual Embeddings: {Config.MULTILINGUAL_INFERENCE_ID}")
            print(f"   - Intelligent Reranking: {Config.RERANK_INFERENCE_ID}")
            print(f"   - Claude AI Intelligence: {Config.CLAUDE_INFERENCE_ID}")
            
            # RSS Feed status
            try:
                feed_info = self.crawler.get_feed_info()
                print(f"📡 RSS Feed: {feed_info.get('title', 'Unknown')} ({feed_info.get('total_entries', 0)} entries)")
            except:
                print("❌ RSS Feed: Not available")
            
            print("="*80)
            
        except Exception as e:
            logger.error(f"❌ Error displaying system status: {str(e)}")
    
    def run_complete_demo(self) -> None:
        """Run complete system demonstration"""
        try:
            print("\n🚀 Starting Complete News Recommendation Engine ML Demo...")
            
            # Setup system
            print("\n🔧 Setting up complete system...")
            if not self.setup_system():
                print("❌ System setup failed")
                return
            
            # Display status
            self.display_system_status()
            
            # Crawl and index news with complete vectorization
            print("\n📰 Crawling and indexing news with complete ML vectorization...")
            articles_count = self.crawl_and_index_news()
            if articles_count > 0:
                print(f"✅ Successfully indexed {articles_count} articles with complete ML vectors")
            else:
                print("⚠️  No articles were indexed")
            
            # Simulate comprehensive user activity
            print("\n🎭 Simulating comprehensive user reading activity...")
            if self.simulate_user_activity():
                print("✅ User activity simulated successfully")
            else:
                print("⚠️  User activity simulation had issues")
            
            # Generate complete ML recommendations for all users
            print("\n🎯 Generating complete AI-powered recommendations...")
            all_recommendations = self.generate_recommendations_for_all_users()
            
            # Display sample recommendations
            sample_count = 0
            for user_id, recommendations in all_recommendations.items():
                if sample_count >= 2:  # Show only first 2 users
                    break
                    
                if "error" not in recommendations:
                    self.display_recommendations(recommendations)
                    sample_count += 1
                    print("\n" + "-"*60)
            
            print(f"\n✅ Complete demo finished successfully!")
            print(f"🌐 Web UI available at: http://localhost:5000")
            print(f"📊 Generated recommendations for {len(all_recommendations)} users")
            print(f"🧠 ML Features: ELSER + Dense Vectors + Claude AI + Reranking")
            
        except Exception as e:
            logger.error(f"❌ Error running complete demo: {str(e)}")
            print(f"❌ Demo failed: {str(e)}")
    
    def display_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """Display recommendations with enhanced formatting"""
        try:
            if "error" in recommendations:
                print(f"❌ Error: {recommendations['error']}")
                return
            
            user_name = recommendations.get('user_name', 'Unknown')
            user_interests = recommendations.get('user_interests', [])
            recs = recommendations.get('recommendations', [])
            performance = recommendations.get('performance', {})
            
            print(f"\n🎯 Complete AI Recommendations for {user_name}")
            print(f"👤 Interests: {', '.join(user_interests)}")
            print(f"⚡ Generated in {performance.get('total_time', 0):.2f}s using {', '.join(performance.get('ml_features_used', []))}")
            print(f"📝 Summary: {recommendations.get('summary', 'No summary')}")
            
            if not recs:
                print("❌ No recommendations found")
                return
            
            print(f"\n📰 Top {len(recs)} AI-Recommended Articles:")
            for i, article in enumerate(recs, 1):
                print(f"\n{i}. {article.get('title', 'No title')}")
                print(f"   📅 Published: {article.get('published_date', 'Unknown')}")
                print(f"   🏷️  Categories: {', '.join(article.get('categories', []))}")
                print(f"   🎯 Relevance: {article.get('claude_relevance', 0.0):.0%}")
                print(f"   💡 AI Reasoning: {article.get('claude_reasoning', 'ML-based selection')}")
                if article.get('ai_selected'):
                    print(f"   ✨ Claude AI Selected")
                else:
                    print(f"   🔍 ML Ranked")
                
        except Exception as e:
            logger.error(f"❌ Error displaying recommendations: {str(e)}")
    
    def test_hybrid_search(self, query: str, user_id: str = None) -> None:
        """Test complete hybrid search functionality"""
        try:
            print(f"\n🔍 Testing complete ML-powered search for: '{query}'")
            print("-" * 60)
            
            user_interests = []
            if user_id:
                user_preferences = self.user_manager.get_user_preferences(user_id)
                user_interests = user_preferences.get('interests', [])
                print(f"👤 User interests: {user_interests}")
            
            start_time = time.time()
            articles = self.es_client.hybrid_search(query, user_interests, size=5)
            search_time = time.time() - start_time
            
            print(f"⚡ Complete hybrid search completed in {search_time:.2f}s")
            
            if articles:
                print(f"📊 Found {len(articles)} articles:")
                for i, article in enumerate(articles, 1):
                    score = article.get('_score', 0)
                    rerank_score = article.get('_rerank_score', 0)
                    print(f"{i}. {article.get('title', 'No title')}")
                    print(f"   🎯 Score: {score:.2f} | Rerank: {rerank_score:.2f}")
                    print(f"   🏷️  Categories: {', '.join(article.get('categories', []))}")
                    print()
            else:
                print("❌ No articles found")
                
        except Exception as e:
            logger.error(f"❌ Error testing search: {str(e)}")

def main():
    """Main function with complete CLI"""
    parser = argparse.ArgumentParser(description="Complete News Recommendation Engine with ML Stack")
    parser.add_argument("--setup", action="store_true", help="Setup complete system")
    parser.add_argument("--demo", action="store_true", help="Run complete ML demo")
    parser.add_argument("--crawl", action="store_true", help="Crawl and index news")
    parser.add_argument("--recommend", action="store_true", help="Generate recommendations")
    parser.add_argument("--user-id", help="Specific user ID for recommendations")
    parser.add_argument("--status", action="store_true", help="Show complete system status")
    parser.add_argument("--users", action="store_true", help="List all users")
    parser.add_argument("--test-search", help="Test hybrid search with a query")
    parser.add_argument("--web", action="store_true", help="Start web UI")
    
    args = parser.parse_args()
    
    try:
        app = NewsRecommendationApp()
        
        if args.setup:
            success = app.setup_system()
            if success:
                print("✅ Complete system setup finished successfully")
                print("💡 Next: python main.py --demo (for full demonstration)")
            else:
                print("❌ System setup failed")
                
        elif args.demo:
            app.run_complete_demo()
            
        elif args.crawl:
            count = app.crawl_and_index_news()
            print(f"✅ Crawled and indexed {count} articles with complete ML vectors")
            
        elif args.recommend:
            if args.user_id:
                recommendations = app.generate_recommendations_for_user(args.user_id)
                app.display_recommendations(recommendations)
            else:
                all_recommendations = app.generate_recommendations_for_all_users()
                for user_id, recs in all_recommendations.items():
                    app.display_recommendations(recs)
                    print("\n" + "-"*60)
                    
        elif args.status:
            app.display_system_status()
            
        elif args.users:
            users = app.user_manager.get_all_users()
            print("\n👥 System Users:")
            print("-" * 50)
            for i, user in enumerate(users, 1):
                print(f"{i}. {user['name']} (ID: {user['user_id'][:8]}...)")
                print(f"   🎯 Interests: {', '.join(user['interests'])}")
                print(f"   📚 Reading History: {len(user.get('reading_history', []))} articles")
                print()
            
        elif args.test_search:
            app.test_hybrid_search(args.test_search, args.user_id)
            
        elif args.web:
            print("🌐 Starting complete web UI...")
            from web_app import app as web_app
            web_app.run(debug=True, host='0.0.0.0', port=5000)
            
        else:
            # Default: run demo or show help
            print("🤖 Complete News Recommendation Engine with ML Stack")
            print("Available commands:")
            print("  --setup      Setup complete system")
            print("  --demo       Run complete ML demo")  
            print("  --crawl      Crawl and index news")
            print("  --recommend  Generate AI recommendations")
            print("  --status     Show complete system status")
            print("  --web        Start interactive web UI")
            print("  --test-search <query>  Test hybrid search")
            
    except Exception as e:
        logger.error(f"❌ Application error: {str(e)}")
        print(f"❌ Error: {str(e)}")
        print("\n💡 Make sure your Elasticsearch credentials are correct in the .env file")

if __name__ == "__main__":
    main()
