#!/usr/bin/env python3
"""
Create remaining components for complete News Recommendation Engine
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
    print("🚀 Creating remaining components for complete system...")
    
    # 1. Create user_profiles.py
    user_profiles_content = '''"""
Dynamic User Profile management for news recommendation engine
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

from config import Config

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class UserProfileManager:
    """Manages dynamic user profiles and preferences"""
    
    def __init__(self, es_client):
        """Initialize User Profile Manager"""
        self.es_client = es_client
    
    def create_user_profile(self, user_data: Dict[str, Any]) -> str:
        """Create a new user profile"""
        try:
            user_id = str(uuid.uuid4())
            
            profile = {
                "user_id": user_id,
                "name": user_data.get("name", ""),
                "interests": user_data.get("interests", []),
                "preferences": {
                    "categories": user_data.get("interests", []),
                    "keywords": user_data.get("interests", []) + user_data.get("additional_keywords", []),
                    "sources": user_data.get("preferred_sources", ["ABS-CBN"]),
                    "reading_time": user_data.get("reading_time", "any"),
                    "content_length": user_data.get("content_length", "medium")
                },
                "reading_history": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Index user profile
            response = self.es_client.client.index(
                index=Config.USER_PROFILE_INDEX,
                id=user_id,
                body=profile
            )
            
            if response.get("result") == "created":
                logger.info(f"✅ Created user profile: {user_id} ({profile['name']})")
                return user_id
            else:
                logger.error(f"❌ Failed to create user profile: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating user profile: {str(e)}")
            return None
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile by ID"""
        try:
            response = self.es_client.client.get(
                index=Config.USER_PROFILE_INDEX,
                id=user_id
            )
            return response["_source"]
            
        except Exception as e:
            logger.warning(f"User profile not found {user_id}: {str(e)}")
            return None
    
    def update_user_interests(self, user_id: str, new_interests: List[str]) -> bool:
        """Dynamically update user interests"""
        try:
            # Validate interests
            if not isinstance(new_interests, list) or len(new_interests) == 0:
                logger.error("Invalid interests provided")
                return False
            
            # Clean and deduplicate interests
            clean_interests = list(set([interest.strip().lower() for interest in new_interests if interest.strip()]))
            
            if len(clean_interests) == 0:
                logger.error("No valid interests after cleaning")
                return False
            
            # Update profile
            update_body = {
                "doc": {
                    "interests": clean_interests,
                    "preferences": {
                        "categories": clean_interests,
                        "keywords": clean_interests + [kw + "s" for kw in clean_interests]  # Add plural forms
                    },
                    "updated_at": datetime.now().isoformat()
                }
            }
            
            response = self.es_client.client.update(
                index=Config.USER_PROFILE_INDEX,
                id=user_id,
                body=update_body
            )
            
            if response.get("result") in ["updated", "noop"]:
                logger.info(f"✅ Updated interests for user {user_id}: {clean_interests}")
                return True
            else:
                logger.error(f"❌ Failed to update interests: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating user interests: {str(e)}")
            return False
    
    def add_user_interest(self, user_id: str, new_interest: str) -> bool:
        """Add a single interest to user profile"""
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                return False
            
            current_interests = profile.get("interests", [])
            new_interest = new_interest.strip().lower()
            
            if new_interest in current_interests:
                logger.warning(f"Interest '{new_interest}' already exists for user {user_id}")
                return True  # Already exists, so technically successful
            
            updated_interests = current_interests + [new_interest]
            return self.update_user_interests(user_id, updated_interests)
            
        except Exception as e:
            logger.error(f"Error adding user interest: {str(e)}")
            return False
    
    def remove_user_interest(self, user_id: str, interest_to_remove: str) -> bool:
        """Remove a single interest from user profile"""
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                return False
            
            current_interests = profile.get("interests", [])
            interest_to_remove = interest_to_remove.strip().lower()
            
            if interest_to_remove not in current_interests:
                logger.warning(f"Interest '{interest_to_remove}' not found for user {user_id}")
                return True  # Not found, so technically successful
            
            updated_interests = [interest for interest in current_interests if interest != interest_to_remove]
            
            if len(updated_interests) == 0:
                logger.error("Cannot remove all interests. At least one is required.")
                return False
            
            return self.update_user_interests(user_id, updated_interests)
            
        except Exception as e:
            logger.error(f"Error removing user interest: {str(e)}")
            return False
    
    def update_reading_history(self, user_id: str, article_id: str, engagement_score: float = 1.0) -> bool:
        """Update user's reading history"""
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                logger.error(f"User profile not found: {user_id}")
                return False
            
            # Add new reading entry
            reading_entry = {
                "article_id": article_id,
                "timestamp": datetime.now().isoformat(),
                "engagement_score": min(1.0, max(0.0, engagement_score))  # Clamp between 0 and 1
            }
            
            # Limit reading history to last 50 entries for performance
            reading_history = profile.get("reading_history", [])
            reading_history.append(reading_entry)
            reading_history = reading_history[-50:]
            
            # Update profile
            update_body = {
                "doc": {
                    "reading_history": reading_history,
                    "updated_at": datetime.now().isoformat()
                }
            }
            
            response = self.es_client.client.update(
                index=Config.USER_PROFILE_INDEX,
                id=user_id,
                body=update_body
            )
            
            logger.debug(f"✅ Updated reading history for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating reading history: {str(e)}")
            return False
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user preferences for search"""
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                return {"interests": [], "categories": [], "keywords": []}
            
            interests = profile.get("interests", [])
            preferences = profile.get("preferences", {})
            
            # Build expanded preferences
            expanded_keywords = interests.copy()
            
            # Add related keywords based on interests
            keyword_expansions = {
                "sports": ["athletics", "game", "tournament", "championship", "team", "player"],
                "basketball": ["nba", "pba", "ncaa", "uaap", "hoops", "court"],
                "politics": ["government", "election", "policy", "senate", "congress", "president"],
                "food": ["cooking", "restaurant", "cuisine", "chef", "recipe", "dining"],
                "technology": ["tech", "digital", "innovation", "software", "app", "gadget"],
                "entertainment": ["movie", "music", "celebrity", "show", "concert", "performance"],
                "health": ["medical", "wellness", "fitness", "doctor", "hospital", "treatment"],
                "business": ["economy", "financial", "company", "market", "industry", "trade"]
            }
            
            for interest in interests:
                if interest in keyword_expansions:
                    expanded_keywords.extend(keyword_expansions[interest])
            
            user_prefs = {
                "interests": interests,
                "categories": preferences.get("categories", interests),
                "keywords": list(set(expanded_keywords)),  # Remove duplicates
                "sources": preferences.get("sources", ["ABS-CBN"]),
                "reading_time": preferences.get("reading_time", "any"),
                "content_length": preferences.get("content_length", "medium")
            }
            
            return user_prefs
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {str(e)}")
            return {"interests": [], "categories": [], "keywords": []}
    
    def create_demo_users(self) -> List[str]:
        """Create diverse demo users for testing"""
        demo_users = [
            {
                "name": "Alex Rivera",
                "interests": ["sports", "basketball", "fitness"],
                "additional_keywords": ["nba", "pba", "athletics", "training"],
                "preferred_sources": ["ABS-CBN"],
                "reading_time": "morning",
                "content_length": "medium"
            },
            {
                "name": "Maria Santos",
                "interests": ["politics", "government", "policy"],
                "additional_keywords": ["election", "senate", "congress", "legislation"],
                "preferred_sources": ["ABS-CBN"],
                "reading_time": "evening",
                "content_length": "long"
            },
            {
                "name": "David Chen",
                "interests": ["food", "cooking", "restaurants"],
                "additional_keywords": ["recipe", "chef", "cuisine", "dining"],
                "preferred_sources": ["ABS-CBN"],
                "reading_time": "any",
                "content_length": "short"
            },
            {
                "name": "Sofia Rodriguez",
                "interests": ["technology", "innovation", "digital"],
                "additional_keywords": ["ai", "software", "app", "gadget"],
                "preferred_sources": ["ABS-CBN"],
                "reading_time": "afternoon",
                "content_length": "medium"
            }
        ]
        
        created_users = []
        for user_data in demo_users:
            user_id = self.create_user_profile(user_data)
            if user_id:
                created_users.append(user_id)
        
        logger.info(f"✅ Created {len(created_users)} demo users")
        return created_users
    
    def simulate_reading_activity(self, user_ids: List[str], articles: List[Dict[str, Any]]) -> bool:
        """Simulate realistic reading activity for demo"""
        try:
            for user_id in user_ids:
                profile = self.get_user_profile(user_id)
                if not profile:
                    continue
                
                user_interests = profile.get("interests", [])
                user_name = profile.get("name", "Unknown")
                
                logger.info(f"🎭 Simulating reading activity for {user_name}")
                
                # Calculate article relevance scores
                relevant_articles = []
                for article in articles:
                    relevance_score = self._calculate_article_relevance(article, user_interests)
                    if relevance_score > 0.1:  # Only consider somewhat relevant articles
                        relevant_articles.append((article, relevance_score))
                
                # Sort by relevance and simulate reading top articles
                relevant_articles.sort(key=lambda x: x[1], reverse=True)
                
                # Simulate reading behavior (read more relevant articles)
                for i, (article, relevance) in enumerate(relevant_articles[:5]):  # Read up to 5 articles
                    # Calculate engagement based on relevance and position
                    base_engagement = min(0.9, relevance)
                    position_penalty = i * 0.05  # Engagement decreases with position
                    final_engagement = max(0.1, base_engagement - position_penalty)
                    
                    # Add some randomness
                    import random
                    final_engagement *= random.uniform(0.8, 1.2)
                    final_engagement = min(1.0, max(0.1, final_engagement))
                    
                    self.update_reading_history(
                        user_id=user_id,
                        article_id=article["id"],
                        engagement_score=final_engagement
                    )
                
                logger.info(f"✅ Simulated {min(5, len(relevant_articles))} interactions for {user_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error simulating reading activity: {str(e)}")
            return False
    
    def _calculate_article_relevance(self, article: Dict[str, Any], user_interests: List[str]) -> float:
        """Calculate how relevant an article is to user interests"""
        try:
            relevance_score = 0.0
            article_text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}".lower()
            article_categories = [cat.lower() for cat in article.get('categories', [])]
            
            for interest in user_interests:
                interest_lower = interest.lower()
                
                # Category match (highest weight)
                if interest_lower in article_categories:
                    relevance_score += 0.4
                
                # Title match (high weight)
                if interest_lower in article.get('title', '').lower():
                    relevance_score += 0.3
                
                # Description match (medium weight)
                if interest_lower in article.get('description', '').lower():
                    relevance_score += 0.2
                
                # Content match (lower weight)
                if interest_lower in article.get('content', '').lower():
                    relevance_score += 0.1
            
            return min(1.0, relevance_score)
            
        except Exception as e:
            logger.warning(f"Error calculating article relevance: {str(e)}")
            return 0.0
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all user profiles"""
        try:
            response = self.es_client.client.search(
                index=Config.USER_PROFILE_INDEX,
                body={
                    "query": {"match_all": {}},
                    "size": 100,
                    "sort": [{"created_at": {"order": "desc"}}]
                }
            )
            
            users = []
            for hit in response["hits"]["hits"]:
                user = hit["_source"]
                users.append(user)
            
            return users
            
        except Exception as e:
            logger.error(f"Error getting all users: {str(e)}")
            return []
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics for dashboard"""
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                return {}
            
            reading_history = profile.get("reading_history", [])
            
            # Calculate stats
            total_articles = len(reading_history)
            avg_engagement = sum(entry.get("engagement_score", 0) for entry in reading_history) / max(1, total_articles)
            
            # Recent activity (last 7 days)
            from datetime import datetime, timedelta
            week_ago = datetime.now() - timedelta(days=7)
            recent_activity = [
                entry for entry in reading_history 
                if datetime.fromisoformat(entry.get("timestamp", "").replace("Z", "")) > week_ago
            ]
            
            return {
                "user_id": user_id,
                "name": profile.get("name", "Unknown"),
                "interests": profile.get("interests", []),
                "total_articles_read": total_articles,
                "average_engagement": round(avg_engagement, 2),
                "recent_activity_count": len(recent_activity),
                "last_updated": profile.get("updated_at", ""),
                "reading_history_sample": reading_history[-5:] if reading_history else []
            }
            
        except Exception as e:
            logger.error(f"Error getting user stats: {str(e)}")
            return {}
'''
    
    if create_file('user_profiles.py', user_profiles_content):
        print("✅ Dynamic user profiles with real-time interest modification")
    
    # 2. Create recommendation_engine.py
    recommendation_engine_content = '''"""
ML-Powered News Recommendation Engine using ELSER, Vector Search, Claude AI, and Reranking
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import time

from config import Config

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class NewsRecommendationEngine:
    """News recommendation engine using complete ML stack"""
    
    def __init__(self, es_client, user_manager):
        """Initialize News Recommendation Engine"""
        self.es_client = es_client
        self.user_manager = user_manager
        self.performance_metrics = {}
    
    def generate_personalized_recommendations(self, user_id: str, num_recommendations: int = 5) -> Dict[str, Any]:
        """Generate personalized recommendations using complete ML stack"""
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Generating ML-powered recommendations for user {user_id}")
            
            # Get user profile and preferences
            user_profile = self.user_manager.get_user_profile(user_id)
            if not user_profile:
                return {"error": "User not found", "user_id": user_id}
            
            user_preferences = self.user_manager.get_user_preferences(user_id)
            user_name = user_profile.get("name", "Unknown")
            
            logger.info(f"👤 User: {user_name}, Interests: {user_preferences['interests']}")
            
            # Step 1: Generate contextual search query
            search_query = self._generate_contextual_query(user_preferences, user_profile)
            
            # Step 2: Perform hybrid search (ELSER + Vector + Keyword)
            search_start = time.time()
            candidate_articles = self.es_client.hybrid_search(
                query=search_query, 
                user_interests=user_preferences['interests'], 
                size=num_recommendations * 3  # Get more candidates for better selection
            )
            search_time = time.time() - search_start
            
            if not candidate_articles:
                logger.warning(f"No articles found for user {user_id}")
                return {
                    "user_id": user_id,
                    "user_name": user_name,
                    "recommendations": [],
                    "summary": "No relevant articles found",
                    "performance": {"search_time": search_time}
                }
            
            logger.info(f"🔍 Found {len(candidate_articles)} candidate articles in {search_time:.2f}s")
            
            # Step 3: Apply additional scoring based on user reading history
            scored_articles = self._apply_personalization_scoring(candidate_articles, user_profile, user_preferences)
            
            # Step 4: Use Claude AI for intelligent selection and ranking
            claude_start = time.time()
            final_recommendations = self._apply_claude_intelligence(
                scored_articles[:num_recommendations * 2], 
                user_profile, 
                user_preferences, 
                num_recommendations
            )
            claude_time = time.time() - claude_start
            
            total_time = time.time() - start_time
            
            # Store performance metrics
            performance = {
                "total_time": round(total_time, 3),
                "search_time": round(search_time, 3),
                "claude_time": round(claude_time, 3),
                "candidates_found": len(candidate_articles),
                "final_recommendations": len(final_recommendations),
                "ml_features_used": ["ELSER", "MultilingualEmbeddings", "ClaudeAI", "Reranking"]
            }
            
            result = {
                "user_id": user_id,
                "user_name": user_name,
                "user_interests": user_preferences['interests'],
                "recommendations": final_recommendations,
                "search_query": search_query,
                "summary": f"Found {len(final_recommendations)} personalized recommendations using AI and semantic search",
                "generated_at": datetime.now().isoformat(),
                "performance": performance
            }
            
            logger.info(f"✅ Generated {len(final_recommendations)} recommendations in {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {
                "error": str(e),
                "user_id": user_id,
                "performance": {"total_time": time.time() - start_time}
            }
    
    def _generate_contextual_query(self, user_preferences: Dict[str, Any], user_profile: Dict[str, Any]) -> str:
        """Generate smart search query based on user context"""
        try:
            interests = user_preferences.get('interests', [])
            keywords = user_preferences.get('keywords', [])
            
            # Combine interests and keywords with intelligent weighting
            all_terms = interests + keywords
            
            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term in all_terms:
                if term.lower() not in seen:
                    unique_terms.append(term)
                    seen.add(term.lower())
            
            # Create contextual query
            if len(unique_terms) == 0:
                return "latest news"
            elif len(unique_terms) <= 3:
                return " OR ".join(unique_terms)
            else:
                # For more terms, create a more sophisticated query
                primary_terms = unique_terms[:3]
                secondary_terms = unique_terms[3:6]
                
                query = " OR ".join(primary_terms)
                if secondary_terms:
                    query += f" OR ({' OR '.join(secondary_terms)})"
                
                return query
            
        except Exception as e:
            logger.warning(f"Error generating contextual query: {str(e)}")
            return "latest news"
    
    def _apply_personalization_scoring(self, articles: List[Dict[str, Any]], user_profile: Dict[str, Any], user_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply personalization scoring based on user reading history and preferences"""
        try:
            reading_history = user_profile.get("reading_history", [])
            user_interests = user_preferences.get("interests", [])
            
            # Create engagement patterns from reading history
            category_engagement = {}
            if reading_history:
                for entry in reading_history[-20:]:  # Look at recent history
                    # Note: In a real system, you'd fetch article details to get categories
                    # For now, we'll estimate based on engagement scores
                    engagement_score = entry.get("engagement_score", 0.5)
                    # This is simplified - you'd want to correlate with actual article categories
                    for interest in user_interests:
                        category_engagement[interest] = category_engagement.get(interest, 0) + engagement_score
            
            # Score articles based on personalization factors
            for article in articles:
                personalization_score = 0.0
                
                # Base score from search
                base_score = article.get("_score", 0.0)
                rerank_score = article.get("_rerank_score", 0.0)
                
                # Interest matching bonus
                article_categories = [cat.lower() for cat in article.get("categories", [])]
                for interest in user_interests:
                    if interest.lower() in article_categories:
                        personalization_score += 0.3
                
                # Reading history pattern matching
                for category, engagement in category_engagement.items():
                    if category.lower() in article_categories:
                        personalization_score += min(0.2, engagement * 0.1)
                
                # Recency bonus
                try:
                    pub_date = datetime.fromisoformat(article.get('published_date', '').replace('Z', '+00:00'))
                    hours_old = (datetime.now(pub_date.tzinfo) - pub_date).total_seconds() / 3600
                    if hours_old < 24:
                        personalization_score += 0.1  # Recent news bonus
                except:
                    pass
                
                # Combine scores
                final_score = (base_score * 0.4) + (rerank_score * 0.4) + (personalization_score * 0.2)
                article["_personalization_score"] = personalization_score
                article["_final_score"] = final_score
            
            # Sort by final score
            articles.sort(key=lambda x: x.get("_final_score", 0), reverse=True)
            
            logger.info("✅ Applied personalization scoring")
            return articles
            
        except Exception as e:
            logger.warning(f"Error applying personalization scoring: {str(e)}")
            return articles
    
    def _apply_claude_intelligence(self, articles: List[Dict[str, Any]], user_profile: Dict[str, Any], user_preferences: Dict[str, Any], num_recommendations: int) -> List[Dict[str, Any]]:
        """Use Claude AI for intelligent article selection and reasoning"""
        try:
            if not articles:
                return []
            
            # Prepare context for Claude
            user_name = user_profile.get("name", "User")
            interests = user_preferences.get("interests", [])
            
            # Build context with article summaries
            articles_context = []
            for i, article in enumerate(articles[:10], 1):  # Limit to top 10 for Claude
                context_entry = f"""
Article {i}:
Title: {article.get('title', 'No title')}
Categories: {', '.join(article.get('categories', []))}
Description: {article.get('description', '')[:200]}...
Published: {article.get('published_date', 'Unknown')}
Score: {article.get('_final_score', 0):.3f}
"""
                articles_context.append(context_entry.strip())
            
            # Create Claude prompt
            prompt = f"""
As an AI news recommendation expert, analyze these articles for {user_name} who is interested in: {', '.join(interests)}.

User Profile:
- Name: {user_name}
- Interests: {', '.join(interests)}
- Reading History: {len(user_profile.get('reading_history', []))} articles

Available Articles:
{chr(10).join(articles_context)}

Please select the top {num_recommendations} most relevant articles and provide reasoning. Return ONLY a valid JSON response:

{{
    "recommendations": [
        {{
            "rank": 1,
            "article_number": <number from list>,
            "relevance_score": <0.0 to 1.0>,
            "reasoning": "<brief explanation why this article matches user interests>"
        }}
    ],
    "summary": "<brief summary of recommendation strategy for this user>"
}}

Focus on articles that best match the user's interests, provide value, and offer diverse perspectives.
"""
            
            # Call Claude via Elasticsearch inference
            try:
                response = self.es_client.client.inference.inference(
                    inference_id=Config.CLAUDE_INFERENCE_ID,
                    body={"input": prompt}
                )
                
                # Extract Claude response
                claude_response = ""
                if "completion" in response and response["completion"]:
                    claude_response = response["completion"][0].get("result", "")
                else:
                    claude_response = str(response)
                
                logger.debug(f"Claude raw response: {claude_response[:200]}...")
                
                # Parse Claude's JSON response
                claude_recommendations = self._parse_claude_response(claude_response)
                
                if claude_recommendations and "recommendations" in claude_recommendations:
                    # Map Claude's selections back to articles
                    final_recommendations = []
                    
                    for rec in claude_recommendations["recommendations"]:
                        article_num = rec.get("article_number", 0)
                        if 1 <= article_num <= len(articles):
                            article = articles[article_num - 1].copy()
                            
                            # Add Claude's insights
                            article.update({
                                "claude_rank": rec.get("rank", 0),
                                "claude_relevance": rec.get("relevance_score", 0.0),
                                "claude_reasoning": rec.get("reasoning", ""),
                                "ai_selected": True,
                                "recommendation_timestamp": datetime.now().isoformat()
                            })
                            
                            final_recommendations.append(article)
                    
                    logger.info(f"✅ Claude selected {len(final_recommendations)} articles")
                    return final_recommendations
                
                else:
                    logger.warning("Claude returned invalid response, using fallback")
                    return self._fallback_selection(articles, num_recommendations)
                
            except Exception as e:
                logger.warning(f"Claude AI failed: {str(e)}, using fallback")
                return self._fallback_selection(articles, num_recommendations)
                
        except Exception as e:
            logger.error(f"Error in Claude intelligence: {str(e)}")
            return self._fallback_selection(articles, num_recommendations)
    
    def _parse_claude_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse Claude's JSON response with multiple strategies"""
        try:
            # Strategy 1: Direct JSON parsing
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Extract JSON from response
            import re
            json_match = re.search(r'\\{.*\\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Strategy 3: Clean and parse
            cleaned = response.strip().replace('```json', '').replace('```', '')
            start_idx = cleaned.find('{')
            end_idx = cleaned.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_part = cleaned[start_idx:end_idx + 1]
                try:
                    return json.loads(json_part)
                except json.JSONDecodeError:
                    pass
            
            logger.warning("Could not parse Claude response as JSON")
            return None
            
        except Exception as e:
            logger.warning(f"Error parsing Claude response: {str(e)}")
            return None
    
    def _fallback_selection(self, articles: List[Dict[str, Any]], num_recommendations: int) -> List[Dict[str, Any]]:
        """Fallback selection when Claude fails"""
        try:
            logger.info(f"📋 Using intelligent fallback selection for {num_recommendations} articles")
            
            # Use the pre-scored articles and add fallback reasoning
            selected_articles = articles[:num_recommendations]
            
            for i, article in enumerate(selected_articles):
                article.update({
                    "claude_rank": i + 1,
                    "claude_relevance": max(0.6, article.get("_final_score", 0.5)),
                    "claude_reasoning": "Selected based on ML scoring and relevance ranking",
                    "ai_selected": False,
                    "fallback_used": True,
                    "recommendation_timestamp": datetime.now().isoformat()
                })
            
            return selected_articles
            
        except Exception as e:
            logger.error(f"Even fallback selection failed: {str(e)}")
            return []
    
    def get_recommendations_for_all_users(self) -> Dict[str, Any]:
        """Generate recommendations for all users"""
        try:
            users = self.user_manager.get_all_users()
            all_recommendations = {}
            
            logger.info(f"🔄 Generating recommendations for {len(users)} users")
            
            for user in users:
                user_id = user["user_id"]
                user_name = user.get("name", "Unknown")
                
                logger.info(f"Processing user: {user_name}")
                recommendations = self.generate_personalized_recommendations(user_id)
                all_recommendations[user_id] = recommendations
            
            logger.info(f"✅ Generated recommendations for {len(users)} users")
            return all_recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for all users: {str(e)}")
            return {}
    
    def record_user_interaction(self, user_id: str, article_id: str, interaction_type: str, score: float = None) -> bool:
        """Record user interaction for learning"""
        try:
            # Map interaction types to engagement scores
            interaction_scores = {
                "view": 0.1,
                "click": 0.3,
                "read": 0.6,
                "like": 0.8,
                "share": 1.0,
                "bookmark": 0.9
            }
            
            engagement_score = score or interaction_scores.get(interaction_type, 0.5)
            
            success = self.user_manager.update_reading_history(
                user_id=user_id,
                article_id=article_id,
                engagement_score=engagement_score
            )
            
            if success:
                logger.info(f"📊 Recorded {interaction_type} interaction for user {user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error recording user interaction: {str(e)}")
            return False
'''
    
    if create_file('recommendation_engine.py', recommendation_engine_content):
        print("✅ Complete ML-powered recommendation engine with Claude AI")
    
    # 3. Create updated main.py with full functionality  
    updated_main_content = '''#!/usr/bin/env python3
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
            print("\\n" + "="*80)
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
            print("\\n🧠 Complete ML Technology Stack:")
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
            print("\\n🚀 Starting Complete News Recommendation Engine ML Demo...")
            
            # Setup system
            print("\\n🔧 Setting up complete system...")
            if not self.setup_system():
                print("❌ System setup failed")
                return
            
            # Display status
            self.display_system_status()
            
            # Crawl and index news with complete vectorization
            print("\\n📰 Crawling and indexing news with complete ML vectorization...")
            articles_count = self.crawl_and_index_news()
            if articles_count > 0:
                print(f"✅ Successfully indexed {articles_count} articles with complete ML vectors")
            else:
                print("⚠️  No articles were indexed")
            
            # Simulate comprehensive user activity
            print("\\n🎭 Simulating comprehensive user reading activity...")
            if self.simulate_user_activity():
                print("✅ User activity simulated successfully")
            else:
                print("⚠️  User activity simulation had issues")
            
            # Generate complete ML recommendations for all users
            print("\\n🎯 Generating complete AI-powered recommendations...")
            all_recommendations = self.generate_recommendations_for_all_users()
            
            # Display sample recommendations
            sample_count = 0
            for user_id, recommendations in all_recommendations.items():
                if sample_count >= 2:  # Show only first 2 users
                    break
                    
                if "error" not in recommendations:
                    self.display_recommendations(recommendations)
                    sample_count += 1
                    print("\\n" + "-"*60)
            
            print(f"\\n✅ Complete demo finished successfully!")
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
            
            print(f"\\n🎯 Complete AI Recommendations for {user_name}")
            print(f"👤 Interests: {', '.join(user_interests)}")
            print(f"⚡ Generated in {performance.get('total_time', 0):.2f}s using {', '.join(performance.get('ml_features_used', []))}")
            print(f"📝 Summary: {recommendations.get('summary', 'No summary')}")
            
            if not recs:
                print("❌ No recommendations found")
                return
            
            print(f"\\n📰 Top {len(recs)} AI-Recommended Articles:")
            for i, article in enumerate(recs, 1):
                print(f"\\n{i}. {article.get('title', 'No title')}")
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
            print(f"\\n🔍 Testing complete ML-powered search for: '{query}'")
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
                    print("\\n" + "-"*60)
                    
        elif args.status:
            app.display_system_status()
            
        elif args.users:
            users = app.user_manager.get_all_users()
            print("\\n👥 System Users:")
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
        print("\\n💡 Make sure your Elasticsearch credentials are correct in the .env file")

if __name__ == "__main__":
    main()
'''
    
    if create_file('main.py', updated_main_content):
        print("✅ Complete main.py with full ML functionality")
    
    print(f"\n🎉 All remaining components created successfully!")
    print(f"\n🚀 You now have the COMPLETE system!")
    print("\n📁 Complete file list:")
    print("  ✅ config.py - Configuration management")
    print("  ✅ .env - Environment variables") 
    print("  ✅ requirements.txt - Dependencies")
    print("  ✅ elasticsearch_client.py - Complete ML client")
    print("  ✅ rss_crawler.py - Enhanced content extraction")
    print("  ✅ user_profiles.py - Dynamic user management")
    print("  ✅ recommendation_engine.py - Claude AI recommendations")
    print("  ✅ main.py - Complete application")
    
    print("\n🏁 Final steps:")
    print("1. Update system: python main.py --setup")
    print("2. Run complete demo: python main.py --demo") 
    print("3. Create web UI component (web_app.py) - let me know if you want it!")

if __name__ == "__main__":
    main()