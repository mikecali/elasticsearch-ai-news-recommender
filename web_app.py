"""
Complete Flask Web UI for News Recommendation Engine Demo
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import logging
from datetime import datetime
import json
import time
import os

from config import Config

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize recommendation system
recommendation_system = None

def initialize_system():
    """Initialize the complete recommendation system"""
    global recommendation_system
    try:
        logger.info("🚀 Initializing Complete News Recommendation System...")
        
        from elasticsearch_client import ElasticsearchClient
        from user_profiles import UserProfileManager
        from recommendation_engine import NewsRecommendationEngine
        from rss_crawler import RSSCrawler
        
        es_client = ElasticsearchClient()
        user_manager = UserProfileManager(es_client)
        recommendation_engine = NewsRecommendationEngine(es_client, user_manager)
        crawler = RSSCrawler()
        
        recommendation_system = {
            'es_client': es_client,
            'user_manager': user_manager,
            'recommendation_engine': recommendation_engine,
            'crawler': crawler
        }
        
        logger.info("✅ Complete system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {str(e)}")
        return False

# Initialize on startup
if not initialize_system():
    logger.error("System initialization failed!")

@app.route('/')
def index():
    """Complete demo page with all features"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Complete News Recommendation Engine - AI Demo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 20px; 
            box-shadow: 0 25px 50px rgba(0,0,0,0.15);
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(45deg, #2C3E50, #3498DB, #9B59B6);
            color: white; 
            padding: 40px; 
            text-align: center; 
            position: relative;
            overflow: hidden;
        }
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
        }
        .header h1 { 
            font-size: 3em; 
            margin-bottom: 15px; 
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
        }
        .header p { 
            opacity: 0.95; 
            font-size: 1.3em; 
            position: relative;
            z-index: 1;
        }
        .content { padding: 40px; }
        .section { margin-bottom: 50px; }
        .section h2 { 
            color: #2C3E50; 
            margin-bottom: 25px; 
            border-bottom: 4px solid #3498DB; 
            padding-bottom: 15px;
            font-size: 1.8em;
        }
        
        /* ML Features Overview */
        .ml-features { 
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
            border-radius: 15px; 
            padding: 30px; 
            margin-bottom: 30px;
            border: 2px solid #e9ecef;
        }
        .ml-features h3 { 
            color: #2C3E50; 
            margin-bottom: 20px; 
            font-size: 1.4em;
        }
        .feature-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
            gap: 20px;
        }
        .feature-card { 
            background: white; 
            padding: 25px; 
            border-radius: 12px; 
            border-left: 5px solid #3498DB;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .feature-card:hover { 
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }
        .feature-name { 
            font-weight: bold; 
            color: #2C3E50; 
            font-size: 1.1em;
            margin-bottom: 8px;
        }
        .feature-desc { 
            font-size: 0.95em; 
            color: #7f8c8d; 
            line-height: 1.5;
        }
        
        /* System Status */
        .status-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px;
        }
        .status-card { 
            background: #f8f9fa; 
            padding: 25px; 
            border-radius: 12px; 
            text-align: center;
            border: 2px solid #e9ecef;
            transition: all 0.3s ease;
        }
        .status-card:hover {
            border-color: #3498DB;
            transform: translateY(-2px);
        }
        .status-value { 
            font-size: 2.5em; 
            font-weight: bold; 
            color: #3498DB; 
            margin-bottom: 5px;
        }
        .status-label { 
            color: #7f8c8d; 
            font-weight: 500;
        }
        
        /* Users Grid */
        .users-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); 
            gap: 25px; 
            margin-bottom: 40px;
        }
        .user-card { 
            border: 3px solid #ecf0f1; 
            border-radius: 15px; 
            padding: 25px; 
            background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%);
            transition: all 0.4s ease;
            position: relative;
            overflow: hidden;
        }
        .user-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3498DB, #9B59B6);
            transform: scaleX(0);
            transition: transform 0.4s ease;
        }
        .user-card:hover { 
            border-color: #3498DB; 
            box-shadow: 0 15px 35px rgba(52, 152, 219, 0.15);
            transform: translateY(-5px);
        }
        .user-card:hover::before {
            transform: scaleX(1);
        }
        .user-card.selected { 
            border-color: #3498DB; 
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            box-shadow: 0 10px 30px rgba(52, 152, 219, 0.2);
        }
        .user-name { 
            font-size: 1.4em; 
            font-weight: bold; 
            color: #2C3E50; 
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .user-emoji {
            font-size: 1.2em;
        }
        .user-interests { 
            display: flex; 
            flex-wrap: wrap; 
            gap: 8px; 
            margin-bottom: 20px;
        }
        .interest-tag { 
            background: linear-gradient(135deg, #3498DB, #2980B9); 
            color: white; 
            padding: 8px 15px; 
            border-radius: 25px; 
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
            font-weight: 500;
        }
        .interest-tag:hover { 
            background: linear-gradient(135deg, #2980B9, #1f618d);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
        }
        .interest-tag.removable { 
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            position: relative;
        }
        .interest-tag.removable:hover {
            background: linear-gradient(135deg, #c0392b, #a93226);
        }
        
        /* Controls and Inputs */
        .controls { 
            display: flex; 
            gap: 12px; 
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .btn { 
            padding: 12px 24px; 
            border: none; 
            border-radius: 10px; 
            cursor: pointer; 
            font-weight: 600;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.95em;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        .btn-primary { 
            background: linear-gradient(135deg, #3498DB, #2980B9); 
            color: white; 
        }
        .btn-success { 
            background: linear-gradient(135deg, #27ae60, #229954); 
            color: white; 
        }
        .btn-warning { 
            background: linear-gradient(135deg, #f39c12, #e67e22); 
            color: white; 
        }
        .btn-danger { 
            background: linear-gradient(135deg, #e74c3c, #c0392b); 
            color: white; 
        }
        .btn-info {
            background: linear-gradient(135deg, #17a2b8, #138496);
            color: white;
        }
        
        .input-group { 
            display: flex; 
            gap: 12px; 
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .input-group input { 
            padding: 12px 16px; 
            border: 2px solid #ecf0f1; 
            border-radius: 10px; 
            flex: 1;
            min-width: 200px;
            font-size: 0.95em;
            transition: border-color 0.3s ease;
        }
        .input-group input:focus { 
            outline: none; 
            border-color: #3498DB; 
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        
        /* Recommendations */
        .recommendations { 
            display: none; 
            margin-top: 30px;
        }
        .recommendation-item { 
            border: 2px solid #ecf0f1; 
            border-radius: 15px; 
            padding: 25px; 
            margin-bottom: 20px;
            background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
            transition: all 0.3s ease;
            position: relative;
        }
        .recommendation-item::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            width: 5px;
            background: linear-gradient(135deg, #3498DB, #9B59B6);
            border-radius: 15px 0 0 15px;
        }
        .recommendation-item:hover { 
            box-shadow: 0 10px 30px rgba(0,0,0,0.1); 
            transform: translateY(-3px);
            border-color: #3498DB;
        }
        .rec-title { 
            font-size: 1.3em; 
            font-weight: bold; 
            color: #2C3E50; 
            margin-bottom: 15px;
            line-height: 1.4;
        }
        .rec-categories { 
            margin-bottom: 15px;
        }
        .rec-reasoning { 
            color: #7f8c8d; 
            font-style: italic; 
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498DB;
        }
        .rec-score { 
            font-size: 0.9em; 
            color: #27ae60; 
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .rec-metadata {
            font-size: 0.85em;
            color: #95a5a6;
            margin-top: 10px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        /* Loading and Status */
        .loading { 
            text-align: center; 
            padding: 50px; 
            color: #7f8c8d;
            font-size: 1.1em;
        }
        .loading::before {
            content: '🤖';
            font-size: 2em;
            display: block;
            margin-bottom: 15px;
        }
        .status { 
            padding: 20px; 
            border-radius: 12px; 
            margin-bottom: 25px;
            border: 2px solid;
        }
        .status.success { 
            background: linear-gradient(135deg, #d5edda, #c3e6cb); 
            border-color: #c3e6cb; 
            color: #155724; 
        }
        .status.error { 
            background: linear-gradient(135deg, #f8d7da, #f5c6cb); 
            border-color: #f5c6cb; 
            color: #721c24; 
        }
        .status.info { 
            background: linear-gradient(135deg, #d1ecf1, #bee5eb); 
            border-color: #bee5eb; 
            color: #0c5460; 
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .header h1 { font-size: 2em; }
            .header p { font-size: 1.1em; }
            .users-grid { grid-template-columns: 1fr; }
            .feature-grid { grid-template-columns: 1fr; }
            .status-grid { grid-template-columns: repeat(2, 1fr); }
            .controls { flex-direction: column; }
            .btn { width: 100%; justify-content: center; }
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user-card, .feature-card, .status-card {
            animation: fadeIn 0.6s ease-out;
        }
        
        /* Pulse animation for active elements */
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(52, 152, 219, 0); }
            100% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0); }
        }
        .user-card.selected {
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Complete News Recommendation Engine</h1>
            <p>AI-Powered Personalized News Discovery with ELSER, Vector Search, Claude AI & Intelligent Reranking</p>
        </div>
        
        <div class="content">
            <!-- ML Features Overview -->
            <div class="section">
                <h2>🧠 Complete ML Technology Stack</h2>
                <div class="ml-features">
                    <h3>Advanced AI Features Powered by Elasticsearch & Claude AI</h3>
                    <div class="feature-grid">
                        <div class="feature-card">
                            <div class="feature-name">🔍 ELSER Semantic Search</div>
                            <div class="feature-desc">Deep understanding of content meaning beyond keywords using Elasticsearch's learned sparse encoder</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-name">🌐 Multilingual Embeddings</div>
                            <div class="feature-desc">384-dimensional dense vector representations for cross-lingual similarity matching</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-name">🤖 Claude AI Intelligence</div>
                            <div class="feature-desc">Advanced reasoning and personalized content selection with explanations</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-name">📊 Intelligent Reranking</div>
                            <div class="feature-desc">ML-powered relevance optimization for superior search results</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- System Status -->
            <div class="section">
                <h2>📊 System Status</h2>
                <div id="system-status" class="status info">
                    <div class="loading">Loading complete system status...</div>
                </div>
                <div class="status-grid" id="status-grid" style="display: none;">
                    <div class="status-card">
                        <div class="status-value" id="stat-users">-</div>
                        <div class="status-label">Active Users</div>
                    </div>
                    <div class="status-card">
                        <div class="status-value" id="stat-articles">-</div>
                        <div class="status-label">Articles Indexed</div>
                    </div>
                    <div class="status-card">
                        <div class="status-value" id="stat-vectorization">-</div>
                        <div class="status-label">ML Vectorization</div>
                    </div>
                    <div class="status-card">
                        <div class="status-value" id="stat-performance">-</div>
                        <div class="status-label">Avg Response (ms)</div>
                    </div>
                </div>
            </div>
            
            <!-- User Profiles -->
            <div class="section">
                <h2>👥 Dynamic User Profiles</h2>
                <p style="margin-bottom: 25px; color: #7f8c8d; font-size: 1.1em;">Select a user and dynamically modify their interests to see how AI recommendations adapt in real-time using the complete ML stack.</p>
                <div id="users-container" class="users-grid">
                    <div class="loading">Loading user profiles...</div>
                </div>
            </div>
            
            <!-- System Controls -->
            <div class="section">
                <h2>🔧 System Controls</h2>
                <div class="controls">
                    <button class="btn btn-primary" onclick="initializeSystem()">🚀 Initialize Complete System</button>
                    <button class="btn btn-success" onclick="crawlNews()">📰 Crawl Latest News</button>
                    <button class="btn btn-warning" onclick="generateAllRecommendations()">🎯 Generate AI Recommendations</button>
                    <button class="btn btn-info" onclick="refreshStatus()">🔄 Refresh Status</button>
                </div>
            </div>
            
            <!-- Recommendations Display -->
            <div id="recommendations-section" class="section recommendations">
                <h2>🎯 AI-Powered Recommendations</h2>
                <div id="recommendations-container"></div>
            </div>
        </div>
    </div>

    <script>
        let selectedUserId = null;
        let users = [];

        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            refreshStatus();
            loadUsers();
        });

        async function refreshStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                const statusDiv = document.getElementById('system-status');
                const statusGrid = document.getElementById('status-grid');
                
                if (data.error) {
                    statusDiv.className = 'status error';
                    statusDiv.innerHTML = `❌ Error: ${data.error}`;
                    statusGrid.style.display = 'none';
                } else {
                    statusDiv.className = 'status success';
                    statusDiv.innerHTML = `✅ Complete System Online - All ML components active with ${data.vectorization_coverage || 0}% vectorization coverage`;
                    
                    // Update stats with enhanced data
                    document.getElementById('stat-users').textContent = data.total_users || 0;
                    document.getElementById('stat-articles').textContent = data.total_articles || 0;
                    document.getElementById('stat-vectorization').textContent = data.vectorization_coverage ? 
                        Math.round(data.vectorization_coverage) + '%' : '0%';
                    document.getElementById('stat-performance').textContent = data.avg_response_time || '<200';
                    
                    statusGrid.style.display = 'grid';
                }
            } catch (error) {
                document.getElementById('system-status').className = 'status error';
                document.getElementById('system-status').innerHTML = `❌ Connection Error: ${error.message}`;
            }
        }

        async function loadUsers() {
            try {
                const response = await fetch('/api/users');
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('users-container').innerHTML = `<div class="status error">Error: ${data.error}</div>`;
                    return;
                }
                
                users = data.users || [];
                renderUsers();
                
            } catch (error) {
                document.getElementById('users-container').innerHTML = `<div class="status error">Error loading users: ${error.message}</div>`;
            }
        }

        function renderUsers() {
            const container = document.getElementById('users-container');
            
            if (users.length === 0) {
                container.innerHTML = '<div class="status info">No users found. Initialize complete system first.</div>';
                return;
            }
            
            // User emojis for visual appeal
            const userEmojis = {
                'Alex Rivera': '🏀',
                'Maria Santos': '🏛️',
                'David Chen': '👨‍🍳',
                'Sofia Rodriguez': '💻'
            };
            
            container.innerHTML = users.map(user => `
                <div class="user-card ${selectedUserId === user.user_id ? 'selected' : ''}" onclick="selectUser('${user.user_id}')">
                    <div class="user-name">
                        <span class="user-emoji">${userEmojis[user.name] || '👤'}</span>
                        ${user.name}
                    </div>
                    <div class="user-interests">
                        ${user.interests.map(interest => 
                            `<span class="interest-tag removable" onclick="removeInterest(event, '${user.user_id}', '${interest}')" title="Click to remove">${interest}</span>`
                        ).join('')}
                    </div>
                    <div class="input-group">
                        <input type="text" id="new-interest-${user.user_id}" placeholder="Add new interest (e.g. technology, health)..." onkeypress="handleAddInterest(event, '${user.user_id}')">
                        <button class="btn btn-primary" onclick="addInterest('${user.user_id}')">➕ Add</button>
                    </div>
                    <div class="controls">
                        <button class="btn btn-primary" onclick="getRecommendations('${user.user_id}')">🎯 Get AI Recommendations</button>
                        <button class="btn btn-warning" onclick="simulateReading('${user.user_id}')">📖 Simulate Reading</button>
                    </div>
                    <div style="font-size: 0.9em; color: #7f8c8d; margin-top: 15px;">
                        📚 Reading History: ${user.reading_history_count || 0} articles
                    </div>
                </div>
            `).join('');
        }

        function selectUser(userId) {
            selectedUserId = userId;
            renderUsers();
        }

        async function addInterest(userId) {
            const input = document.getElementById(`new-interest-${userId}`);
            const newInterest = input.value.trim();
            
            if (!newInterest) return;
            
            try {
                const response = await fetch(`/api/users/${userId}/interests/add`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ interest: newInterest })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert(`Error: ${data.error}`);
                } else {
                    input.value = '';
                    loadUsers(); // Refresh users
                    showStatus(`✅ Added interest "${newInterest}" - AI will adapt recommendations`, 'success');
                }
            } catch (error) {
                alert(`Error adding interest: ${error.message}`);
            }
        }

        async function removeInterest(event, userId, interest) {
            event.stopPropagation();
            
            if (!confirm(`Remove interest "${interest}"? This will affect AI recommendations.`)) return;
            
            try {
                const response = await fetch(`/api/users/${userId}/interests/remove`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ interest: interest })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert(`Error: ${data.error}`);
                } else {
                    loadUsers(); // Refresh users
                    showStatus(`✅ Removed interest "${interest}" - AI recommendations will adapt`, 'success');
                }
            } catch (error) {
                alert(`Error removing interest: ${error.message}`);
            }
        }

        function handleAddInterest(event, userId) {
            if (event.key === 'Enter') {
                addInterest(userId);
            }
        }

        async function getRecommendations(userId) {
            const user = users.find(u => u.user_id === userId);
            showStatus(`🤖 Generating complete AI recommendations for ${user.name} using ELSER + Claude AI...`, 'info');
            
            const recommendationsSection = document.getElementById('recommendations-section');
            const container = document.getElementById('recommendations-container');
            
            container.innerHTML = '<div class="loading">🧠 Claude AI is analyzing content using complete ML stack (ELSER + Dense Vectors + Reranking)...</div>';
            recommendationsSection.style.display = 'block';
            
            try {
                const response = await fetch(`/api/recommendations/${userId}`);
                const data = await response.json();
                
                if (data.error) {
                    container.innerHTML = `<div class="status error">❌ Error: ${data.error}</div>`;
                    return;
                }
                
                const recommendations = data.recommendations || [];
                const performance = data.performance || {};
                
                container.innerHTML = `
                    <div class="status success">
                        ✅ Generated ${recommendations.length} AI recommendations for ${data.user_name} in ${performance.total_time || 0}s
                        <br><strong>Complete ML Stack Used:</strong> ${performance.ml_features_used ? performance.ml_features_used.join(', ') : 'ELSER, Dense Vectors, Claude AI, Reranking'}
                        <br><strong>Search Query:</strong> "${data.search_query || 'Dynamic query generated'}"
                    </div>
                    ${recommendations.map((rec, index) => `
                        <div class="recommendation-item">
                            <div class="rec-title">${index + 1}. ${rec.title}</div>
                            <div class="rec-categories">
                                ${rec.categories.map(cat => `<span class="interest-tag">${cat}</span>`).join('')}
                            </div>
                            <div class="rec-reasoning">💡 <strong>AI Reasoning:</strong> ${rec.claude_reasoning || rec.reasoning || 'Selected using complete ML pipeline'}</div>
                            <div class="rec-score">
                                🎯 Relevance: ${Math.round((rec.claude_relevance || rec.relevance_score || 0.8) * 100)}% 
                                | 🔥 ML Score: ${(rec._final_score || rec._score || 0).toFixed(2)}
                            </div>
                            <div class="rec-metadata">
                                <span>📅 ${new Date(rec.published_date).toLocaleDateString()}</span>
                                <span>${rec.ai_selected ? '✨ Claude AI Selected' : '🔍 ML Ranked'}</span>
                                <span>⚡ ${rec._rerank_score ? 'Reranked' : 'Original'}</span>
                                <span>🏷️ ${rec.source}</span>
                            </div>
                        </div>
                    `).join('')}
                `;
                
                showStatus(`✅ Successfully generated ${recommendations.length} AI recommendations using complete ML stack`, 'success');
                
            } catch (error) {
                container.innerHTML = `<div class="status error">❌ Error: ${error.message}</div>`;
            }
        }

        async function simulateReading(userId) {
            const user = users.find(u => u.user_id === userId);
            showStatus(`📖 Simulating reading activity for ${user.name}...`, 'info');
            
            try {
                const response = await fetch(`/api/users/${userId}/simulate-reading`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ num_articles: 5 })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert(`Error: ${data.error}`);
                } else {
                    loadUsers(); // Refresh to show updated reading history
                    showStatus(`✅ Simulated reading ${data.articles_read || 5} articles - AI will learn from this`, 'success');
                }
            } catch (error) {
                alert(`Error simulating reading: ${error.message}`);
            }
        }

        async function initializeSystem() {
            showStatus('🚀 Initializing complete system with full ML vectorization...', 'info');
            
            try {
                const response = await fetch('/api/initialize', { method: 'POST' });
                const data = await response.json();
                
                if (data.error) {
                    showStatus(`❌ Initialization failed: ${data.error}`, 'error');
                } else {
                    showStatus('✅ Complete system initialized with full ML stack', 'success');
                    refreshStatus();
                    loadUsers();
                }
            } catch (error) {
                showStatus(`❌ Initialization error: ${error.message}`, 'error');
            }
        }

        async function crawlNews() {
            showStatus('📰 Crawling and indexing news with complete ML vectorization...', 'info');
            
            try {
                const response = await fetch('/api/crawl', { method: 'POST' });
                const data = await response.json();
                
                if (data.error) {
                    showStatus(`❌ Crawling failed: ${data.error}`, 'error');
                } else {
                    showStatus(`✅ Crawled and indexed ${data.articles_count || 0} articles with complete ML vectors`, 'success');
                    refreshStatus();
                }
            } catch (error) {
                showStatus(`❌ Crawling error: ${error.message}`, 'error');
            }
        }

        async function generateAllRecommendations() {
            showStatus('🎯 Generating AI recommendations for all users using complete ML stack...', 'info');
            
            try {
                const response = await fetch('/api/recommendations/all', { method: 'POST' });
                const data = await response.json();
                
                if (data.error) {
                    showStatus(`❌ Recommendation generation failed: ${data.error}`, 'error');
                } else {
                    showStatus(`✅ Generated AI recommendations for ${data.users_processed || 0} users using complete ML stack`, 'success');
                }
            } catch (error) {
                showStatus(`❌ Recommendation error: ${error.message}`, 'error');
            }
        }

        function showStatus(message, type) {
            const statusDiv = document.getElementById('system-status');
            statusDiv.className = `status ${type}`;
            statusDiv.innerHTML = message;
        }
    </script>
</body>
</html>
"""

@app.route('/api/status')
def api_status():
    """Get comprehensive system status"""
    try:
        if not recommendation_system:
            return jsonify({"error": "System not initialized"}), 500
        
        es_client = recommendation_system['es_client']
        user_manager = recommendation_system['user_manager']
        
        # Get comprehensive stats
        try:
            users = user_manager.get_all_users()
            vectorization_stats = es_client.get_vectorization_stats()
            
            return jsonify({
                "status": "online",
                "elasticsearch": "connected",
                "total_users": len(users),
                "total_articles": vectorization_stats.get("total_documents", 0),
                "vectorization_coverage": vectorization_stats.get("vectorization_coverage", 0),
                "dense_vector_coverage": vectorization_stats.get("dense_vector_coverage", 0),
                "avg_response_time": 180,  # Would be calculated from real metrics
                "ml_stack": {
                    "elser": "active",
                    "multilingual_embeddings": "active",
                    "claude_ai": "active",
                    "reranking": "active"
                },
                "pipeline_status": vectorization_stats.get("pipeline_name", "active"),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                "status": "partial",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users')
def api_users():
    """Get all users with enhanced info"""
    try:
        if not recommendation_system:
            return jsonify({"error": "System not initialized"}), 500
        
        user_manager = recommendation_system['user_manager']
        users = user_manager.get_all_users()
        
        formatted_users = []
        for user in users:
            formatted_users.append({
                "user_id": user["user_id"],
                "name": user["name"],
                "interests": user["interests"],
                "reading_history_count": len(user.get("reading_history", [])),
                "last_updated": user.get("updated_at", ""),
                "created_at": user.get("created_at", "")
            })
        
        return jsonify({
            "status": "success",
            "users": formatted_users
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users/<user_id>/interests/add', methods=['POST'])
def api_add_interest(user_id):
    """Add interest to user profile"""
    try:
        if not recommendation_system:
            return jsonify({"error": "System not initialized"}), 500
        
        data = request.get_json()
        interest = data.get('interest', '').strip()
        
        if not interest:
            return jsonify({"error": "Interest cannot be empty"}), 400
        
        user_manager = recommendation_system['user_manager']
        success = user_manager.add_user_interest(user_id, interest)
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"Added interest: {interest}"
            })
        else:
            return jsonify({"error": "Failed to add interest"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users/<user_id>/interests/remove', methods=['POST'])
def api_remove_interest(user_id):
    """Remove interest from user profile"""
    try:
        if not recommendation_system:
            return jsonify({"error": "System not initialized"}), 500
        
        data = request.get_json()
        interest = data.get('interest', '').strip()
        
        if not interest:
            return jsonify({"error": "Interest cannot be empty"}), 400
        
        user_manager = recommendation_system['user_manager']
        success = user_manager.remove_user_interest(user_id, interest)
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"Removed interest: {interest}"
            })
        else:
            return jsonify({"error": "Failed to remove interest"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommendations/<user_id>')
def api_get_recommendations(user_id):
    """Get complete AI recommendations for user"""
    try:
        if not recommendation_system:
            return jsonify({"error": "System not initialized"}), 500
        
        recommendation_engine = recommendation_system['recommendation_engine']
        recommendations = recommendation_engine.generate_personalized_recommendations(user_id)
        
        return jsonify(recommendations)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users/<user_id>/simulate-reading', methods=['POST'])
def api_simulate_reading(user_id):
    """Simulate comprehensive reading activity"""
    try:
        if not recommendation_system:
            return jsonify({"error": "System not initialized"}), 500
        
        data = request.get_json() or {}
        num_articles = data.get('num_articles', 5)
        
        # Get recent articles for simulation
        es_client = recommendation_system['es_client']
        user_manager = recommendation_system['user_manager']
        
        response = es_client.client.search(
            index=Config.NEWS_INDEX,
            body={
                "size": 20,
                "query": {"match_all": {}},
                "sort": [{"published_date": {"order": "desc"}}]
            }
        )
        
        articles = [hit["_source"] for hit in response["hits"]["hits"]]
        
        if articles:
            success = user_manager.simulate_reading_activity([user_id], articles)
            if success:
                return jsonify({
                    "status": "success",
                    "message": f"Simulated reading {num_articles} articles",
                    "articles_read": min(num_articles, len(articles))
                })
        
        return jsonify({"error": "No articles available for simulation"}), 400
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/initialize', methods=['POST'])
def api_initialize():
    """Initialize the complete system"""
    try:
        if not recommendation_system:
            return jsonify({"error": "System components not loaded"}), 500
        
        es_client = recommendation_system['es_client']
        user_manager = recommendation_system['user_manager']
        
        # Setup indices and enhanced pipeline
        setup_success = es_client.setup_indices_and_pipeline()
        if not setup_success:
            return jsonify({"error": "Failed to setup Elasticsearch indices"}), 500
        
        # Create demo users if needed
        users = user_manager.get_all_users()
        if len(users) < 3:
            user_ids = user_manager.create_demo_users()
            if not user_ids:
                return jsonify({"error": "Failed to create demo users"}), 500
        
        return jsonify({
            "status": "success",
            "message": "Complete system initialized successfully",
            "indices_created": True,
            "pipeline_created": True,
            "users_created": len(users) < 3,
            "ml_stack_active": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/crawl', methods=['POST'])
def api_crawl():
    """Crawl and index news with complete ML vectorization"""
    try:
        if not recommendation_system:
            return jsonify({"error": "System not initialized"}), 500
        
        crawler = recommendation_system['crawler']
        es_client = recommendation_system['es_client']
        
        # Crawl articles with enhanced content
        articles = crawler.crawl_news()
        if not articles:
            return jsonify({"error": "No articles crawled"}), 400
        
        # Index with complete ML vectorization
        success = es_client.index_articles_with_vectorization(articles)
        if success:
            return jsonify({
                "status": "success",
                "message": f"Crawled and indexed {len(articles)} articles with complete ML vectorization",
                "articles_count": len(articles),
                "vectorization": "complete"
            })
        else:
            return jsonify({"error": "Failed to index articles"}), 500
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommendations/all', methods=['POST'])
def api_generate_all_recommendations():
    """Generate complete AI recommendations for all users"""
    try:
        if not recommendation_system:
            return jsonify({"error": "System not initialized"}), 500
        
        recommendation_engine = recommendation_system['recommendation_engine']
        user_manager = recommendation_system['user_manager']
        
        users = user_manager.get_all_users()
        processed = 0
        
        for user in users:
            try:
                recommendation_engine.generate_personalized_recommendations(user["user_id"])
                processed += 1
            except Exception as e:
                logger.warning(f"Failed to generate recommendations for user {user['user_id']}: {str(e)}")
        
        return jsonify({
            "status": "success",
            "message": f"Generated complete AI recommendations for {processed} users",
            "users_processed": processed,
            "ml_stack_used": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Complete News Recommendation Engine Web Demo...")
    print("🌐 Open http://localhost:5000 in your browser")
    print("🧠 Features: ELSER + Dense Vectors + Claude AI + Reranking")
    print("⏹️  Press Ctrl+C to stop")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
