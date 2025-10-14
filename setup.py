#!/usr/bin/env python3
"""
Setup script for Student Dropout Prediction Model

This script helps set up the environment and install dependencies.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")
        return True

def create_virtual_environment():
    """Create a virtual environment."""
    if os.path.exists("venv"):
        print("📁 Virtual environment already exists!")
        return True
    
    return run_command("python -m venv venv", "Creating virtual environment")

def activate_and_install():
    """Activate virtual environment and install dependencies."""
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install dependencies
    install_cmd = f"{pip_cmd} install -r requirements.txt"
    return run_command(install_cmd, "Installing dependencies")

def verify_installation():
    """Verify that key packages are installed."""
    print("🔍 Verifying installation...")
    
    try:
        import pandas
        import numpy
        import sklearn
        import xgboost
        import tensorflow
        print("✅ All key packages imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Student Dropout Prediction Model - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("⚠️  Continuing without virtual environment...")
    
    # Install dependencies
    if not activate_and_install():
        print("❌ Setup failed during dependency installation!")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Setup verification failed!")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("=" * 50)
    
    print("\n📋 Next steps:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    
    print("2. Run the example:")
    print("   python example_usage.py")
    
    print("3. Or run the main script:")
    print("   python student_dropout_prediction.py")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()
