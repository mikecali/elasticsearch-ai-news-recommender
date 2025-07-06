"""
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
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
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
