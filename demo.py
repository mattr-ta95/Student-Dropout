#!/usr/bin/env python3
"""
Demo Script for Student Dropout Prediction Model

This script provides a quick demonstration of the project capabilities
without requiring all ML dependencies to be installed.
"""

import pandas as pd
import numpy as np
import os

def demo_data_loading():
    """Demonstrate data loading and basic analysis."""
    print("🎯 STUDENT DROPOUT PREDICTION - DEMO")
    print("=" * 50)
    
    # Check data files
    data_files = {
        'Stage1_data.csv': 'Applicant & Course Information',
        'Stage2_data.csv': 'Student Engagement Data', 
        'Stage3_data.csv': 'Academic Performance Data'
    }
    
    print("\n📊 Data Overview:")
    for filename, description in data_files.items():
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            print(f"✅ {description}")
            print(f"   📈 {df.shape[0]:,} students, {df.shape[1]} features")
            
            # Show target distribution
            if 'CompletedCourse' in df.columns:
                completion_rate = df['CompletedCourse'].value_counts(normalize=True)
                dropout_rate = completion_rate.get('No', 0) * 100
                print(f"   🎯 Dropout rate: {dropout_rate:.1f}%")
            
            # Show sample features
            feature_cols = [col for col in df.columns if col not in ['CompletedCourse', 'LearnerCode']][:3]
            print(f"   🔧 Sample features: {', '.join(feature_cols)}")
            print()
        else:
            print(f"❌ {filename}: Not found")
    
    return True

def demo_feature_analysis():
    """Demonstrate feature analysis capabilities."""
    print("🔍 Feature Analysis Demo:")
    print("-" * 30)
    
    if not os.path.exists('Stage1_data.csv'):
        print("❌ Stage1_data.csv not found")
        return False
    
    df = pd.read_csv('Stage1_data.csv')
    
    # Data types
    print(f"📋 Data Types:")
    print(f"   Categorical: {df.select_dtypes(include=['object']).shape[1]} features")
    print(f"   Numerical: {df.select_dtypes(include=[np.number]).shape[1]} features")
    
    # Missing values
    missing_data = df.isnull().sum()
    high_missing = missing_data[missing_data > len(df) * 0.5]
    if len(high_missing) > 0:
        print(f"⚠️  High missing data (>50%): {', '.join(high_missing.index)}")
    else:
        print("✅ No features with >50% missing data")
    
    # High cardinality
    categorical_cols = df.select_dtypes(include=['object']).columns
    high_cardinality = []
    for col in categorical_cols:
        if df[col].nunique() > 200:
            high_cardinality.append(col)
    
    if high_cardinality:
        print(f"⚠️  High cardinality features: {', '.join(high_cardinality)}")
    else:
        print("✅ No high cardinality features detected")
    
    return True

def demo_target_analysis():
    """Demonstrate target variable analysis."""
    print("\n🎯 Target Variable Analysis:")
    print("-" * 30)
    
    if not os.path.exists('Stage1_data.csv'):
        print("❌ Stage1_data.csv not found")
        return False
    
    df = pd.read_csv('Stage1_data.csv')
    
    if 'CompletedCourse' not in df.columns:
        print("❌ Target variable 'CompletedCourse' not found")
        return False
    
    # Distribution
    target_counts = df['CompletedCourse'].value_counts()
    target_percent = df['CompletedCourse'].value_counts(normalize=True) * 100
    
    print("📊 Distribution:")
    for value, count in target_counts.items():
        percent = target_percent[value]
        status = "✅ Complete" if value == 'Yes' else "❌ Dropout"
        print(f"   {status}: {count:,} students ({percent:.1f}%)")
    
    # Class imbalance assessment
    min_class_percent = target_percent.min()
    if min_class_percent < 10:
        print(f"⚠️  Severe class imbalance: {min_class_percent:.1f}% minority class")
    elif min_class_percent < 20:
        print(f"⚠️  Moderate class imbalance: {min_class_percent:.1f}% minority class")
    else:
        print(f"✅ Balanced dataset: {min_class_percent:.1f}% minority class")
    
    return True

def demo_stage_comparison():
    """Demonstrate stage comparison."""
    print("\n📈 Stage Comparison:")
    print("-" * 30)
    
    stages = ['Stage1_data.csv', 'Stage2_data.csv', 'Stage3_data.csv']
    stage_info = []
    
    for stage_file in stages:
        if os.path.exists(stage_file):
            df = pd.read_csv(stage_file)
            stage_info.append({
                'file': stage_file,
                'students': df.shape[0],
                'features': df.shape[1]
            })
    
    if len(stage_info) > 1:
        print("📊 Feature Evolution:")
        for i, info in enumerate(stage_info):
            stage_name = info['file'].replace('_data.csv', '')
            if i == 0:
                print(f"   {stage_name}: {info['features']} features (baseline)")
            else:
                prev_features = stage_info[i-1]['features']
                new_features = info['features'] - prev_features
                print(f"   {stage_name}: {info['features']} features (+{new_features} new)")
        
        # Consistency check
        student_counts = [info['students'] for info in stage_info]
        if len(set(student_counts)) == 1:
            print("✅ All stages have consistent student count")
        else:
            print("⚠️  Inconsistent student counts across stages")
    
    return True

def demo_ml_readiness():
    """Demonstrate ML pipeline readiness."""
    print("\n🤖 Machine Learning Readiness:")
    print("-" * 30)
    
    # Check if main module can be imported
    try:
        from student_dropout_prediction import StudentDropoutPredictor
        print("✅ Main ML module imports successfully")
        
        # Test instantiation
        predictor = StudentDropoutPredictor()
        print("✅ ML pipeline can be instantiated")
        
        # Check neural network availability
        try:
            import tensorflow as tf
            import keras
            print("✅ Neural networks available (TensorFlow/Keras)")
        except ImportError:
            print("⚠️  Neural networks not available (TensorFlow/Keras missing)")
            print("   XGBoost models will still work")
        
        return True
        
    except ImportError as e:
        print(f"❌ ML module import failed: {e}")
        print("   Install dependencies: pip install -r requirements.txt")
        return False

def main():
    """Run the complete demo."""
    print("🚀 Student Dropout Prediction Model - Demo")
    print("=" * 60)
    
    demos = [
        ("Data Loading", demo_data_loading),
        ("Feature Analysis", demo_feature_analysis), 
        ("Target Analysis", demo_target_analysis),
        ("Stage Comparison", demo_stage_comparison),
        ("ML Readiness", demo_ml_readiness)
    ]
    
    results = []
    for name, demo_func in demos:
        try:
            result = demo_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} demo failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\n🎯 Demo Results: {passed}/{total} components working")
    
    if passed == total:
        print("🎉 All components are ready!")
        print("\n📋 Next steps:")
        print("1. Run: python data_validation.py")
        print("2. Run: python student_dropout_prediction.py")
        print("3. Or run: python example_usage.py")
    else:
        print("⚠️  Some components need attention")
        print("   Check the errors above and install missing dependencies")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
