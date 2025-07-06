"""
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
