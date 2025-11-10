#!/usr/bin/env python3
"""
Test Installation Script for Student Dropout Prediction Model

This script tests that all dependencies are properly installed and the code can run.
"""

import sys
import os
import importlib

def test_imports():
    """Test that all required packages can be imported."""
    print("🧪 Testing package imports...")

    required_packages = [
        'pandas',
        'numpy',
        'matplotlib.pyplot',
        'seaborn',
        'sklearn',
        'xgboost',
    ]

    optional_packages = [
        'tensorflow',
        'keras',
    ]

    failed_imports = []

    # Test required packages
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)

    # Test optional packages
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} (optional)")
        except ImportError as e:
            print(f"⚠️  {package} (optional): {e}")

    return failed_imports

def test_data_files():
    """Test that data files exist and can be loaded."""
    print("\n🧪 Testing data files...")
    
    data_files = [
        'Stage1_data.csv',
        'Stage2_data.csv', 
        'Stage3_data.csv'
    ]
    
    missing_files = []
    
    for filename in data_files:
        if os.path.exists(filename):
            try:
                import pandas as pd
                df = pd.read_csv(filename)
                print(f"✅ {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"❌ {filename}: Error loading - {e}")
                missing_files.append(filename)
        else:
            print(f"❌ {filename}: File not found")
            missing_files.append(filename)
    
    return missing_files

def test_main_module():
    """Test that the main module can be imported."""
    print("\n🧪 Testing main module...")
    
    try:
        from student_dropout_prediction import StudentDropoutPredictor
        print("✅ Main module imports successfully")
        
        # Test instantiation
        predictor = StudentDropoutPredictor()
        print("✅ Class instantiation works")
        
        return True
    except Exception as e:
        print(f"❌ Main module error: {e}")
        return False

def test_data_validation():
    """Test that data validation script works."""
    print("\n🧪 Testing data validation...")
    
    try:
        import data_validation
        print("✅ Data validation module imports successfully")
        return True
    except Exception as e:
        print(f"❌ Data validation error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Student Dropout Prediction - Installation Test")
    print("=" * 60)
    
    # Test Python version
    print(f"🐍 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    else:
        print("✅ Python version is compatible")
    
    # Run tests
    failed_imports = test_imports()
    missing_files = test_data_files()
    main_module_ok = test_main_module()
    validation_ok = test_data_validation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if not failed_imports and not missing_files and main_module_ok and validation_ok:
        print("🎉 All tests passed! Installation is successful.")
        print("\n📋 Next steps:")
        print("1. Run: python data_validation.py")
        print("2. Run: python student_dropout_prediction.py")
        print("3. Or run: python example_usage.py")
        return True
    else:
        print("❌ Some tests failed. Please address the issues above.")
        
        if failed_imports:
            print(f"\n🔧 Install missing packages:")
            print(f"pip install {' '.join(failed_imports)}")
        
        if missing_files:
            print(f"\n📁 Missing data files: {', '.join(missing_files)}")
            print("Make sure all CSV data files are in the project directory.")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
