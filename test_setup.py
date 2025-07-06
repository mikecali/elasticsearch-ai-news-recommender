#!/usr/bin/env python3
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
        
        print("\n🎉 All imports and connections working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_import()
