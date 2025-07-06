#!/usr/bin/env python3
"""
Quick startup script for the complete News Recommendation Engine
"""

import subprocess
import sys
import os

def main():
    print("🚀 News Recommendation Engine - Complete System")
    print("="*60)
    
    # Check if main.py exists
    if not os.path.exists('main.py'):
        print("❌ main.py not found. Run create_full_system.py first.")
        return
    
    print("Available options:")
    print("1. 🔧 Setup complete system")
    print("2. 🎯 Run complete demo")
    print("3. 🌐 Start web UI")
    print("4. 📊 Show system status")
    print("5. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                print("\n🔧 Setting up complete system...")
                subprocess.run([sys.executable, 'main.py', '--setup'])
                
            elif choice == '2':
                print("\n🎯 Running complete demo...")
                subprocess.run([sys.executable, 'main.py', '--demo'])
                
            elif choice == '3':
                print("\n🌐 Starting web UI...")
                print("Open http://localhost:5000 in your browser")
                subprocess.run([sys.executable, 'main.py', '--web'])
                
            elif choice == '4':
                print("\n📊 System status:")
                subprocess.run([sys.executable, 'main.py', '--status'])
                
            elif choice == '5':
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
