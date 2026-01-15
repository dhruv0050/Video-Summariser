# Quick test script to verify setup
import sys
import subprocess

def check_python():
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor} (need 3.10+)")
        return False

def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg installed: {version_line}")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("❌ FFmpeg not found")
    print("   Install: choco install ffmpeg")
    print("   Or download from: https://ffmpeg.org/download.html")
    return False

def check_env_file():
    import os
    if os.path.exists('.env'):
        print("✅ .env file exists")
        return True
    else:
        print("❌ .env file not found")
        return False

def check_packages():
    try:
        import fastapi
        import google.generativeai
        import pymongo
        import cv2
        print("✅ Key packages installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def main():
    print("🔍 Checking Video Intelligence Pipeline setup...\n")
    
    checks = [
        ("Python 3.10+", check_python()),
        ("FFmpeg", check_ffmpeg()),
        ("Environment file", check_env_file()),
        ("Python packages", check_packages())
    ]
    
    print("\n" + "="*50)
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    if passed == total:
        print(f"✅ All checks passed ({passed}/{total})")
        print("\n🚀 Ready to start!")
        print("   Run: python main.py")
        print("   Or: uvicorn main:app --reload")
    else:
        print(f"⚠️  {total - passed} check(s) failed ({passed}/{total} passed)")
        print("\n📋 Fix the issues above and try again")

if __name__ == "__main__":
    main()
