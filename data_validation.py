#!/usr/bin/env python3
"""
Data Validation Script for Student Dropout Prediction

This script validates and provides insights about the student data files
before running the machine learning pipeline.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def validate_data_file(filename, stage_name):
    """
    Validate a single data file and provide insights.
    
    Args:
        filename (str): Path to the CSV file
        stage_name (str): Name of the stage for reporting
        
    Returns:
        dict: Validation results and insights
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING {stage_name.upper()}")
    print(f"{'='*60}")
    
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return None
    
    try:
        # Load data
        df = pd.read_csv(filename)
        print(f"✅ File loaded successfully: {filename}")
        print(f"📊 Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Basic info
        print(f"\n📋 Data Types:")
        print(df.dtypes.value_counts())
        
        # Missing values
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        print(f"\n🔍 Missing Values Analysis:")
        if missing_data.sum() == 0:
            print("✅ No missing values found!")
        else:
            missing_summary = pd.DataFrame({
                'Missing Count': missing_data,
                'Missing %': missing_percent
            }).sort_values('Missing %', ascending=False)
            
            print(missing_summary[missing_summary['Missing Count'] > 0])
        
        # Target variable analysis
        if 'CompletedCourse' in df.columns:
            print(f"\n🎯 Target Variable Analysis:")
            target_counts = df['CompletedCourse'].value_counts()
            target_percent = df['CompletedCourse'].value_counts(normalize=True) * 100
            
            print("Distribution:")
            for value, count in target_counts.items():
                percent = target_percent[value]
                print(f"  {value}: {count:,} students ({percent:.2f}%)")
            
            # Class imbalance check
            min_class_percent = target_percent.min()
            if min_class_percent < 10:
                print(f"⚠️  Warning: Severe class imbalance detected ({min_class_percent:.2f}% minority class)")
            elif min_class_percent < 20:
                print(f"⚠️  Note: Moderate class imbalance ({min_class_percent:.2f}% minority class)")
            else:
                print(f"✅ Balanced dataset ({min_class_percent:.2f}% minority class)")
        
        # Feature analysis
        print(f"\n🔧 Feature Analysis:")
        
        # Categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if 'CompletedCourse' in categorical_cols:
            categorical_cols.remove('CompletedCourse')
        
        print(f"  Categorical features: {len(categorical_cols)}")
        if categorical_cols:
            print(f"  Categorical columns: {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
        
        # Numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"  Numerical features: {len(numerical_cols)}")
        if numerical_cols:
            print(f"  Numerical columns: {numerical_cols[:5]}{'...' if len(numerical_cols) > 5 else ''}")
        
        # High cardinality check
        high_cardinality = []
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count > 200:
                high_cardinality.append((col, unique_count))
        
        if high_cardinality:
            print(f"\n⚠️  High Cardinality Features (>200 unique values):")
            for col, count in high_cardinality:
                print(f"  {col}: {count} unique values")
        else:
            print(f"\n✅ No high cardinality features detected")
        
        # Duplicate check
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"\n⚠️  Found {duplicates} duplicate rows")
        else:
            print(f"\n✅ No duplicate rows found")
        
        return {
            'shape': df.shape,
            'missing_data': missing_data,
            'target_distribution': target_counts if 'CompletedCourse' in df.columns else None,
            'categorical_features': categorical_cols,
            'numerical_features': numerical_cols,
            'high_cardinality': high_cardinality,
            'duplicates': duplicates
        }
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def compare_stages(results):
    """
    Compare data across different stages.
    
    Args:
        results (dict): Results from validate_data_file for each stage
    """
    print(f"\n{'='*60}")
    print("STAGE COMPARISON")
    print(f"{'='*60}")
    
    stages = list(results.keys())
    
    print(f"\n📊 Data Size Comparison:")
    for stage, result in results.items():
        if result:
            print(f"  {stage}: {result['shape'][0]:,} students, {result['shape'][1]} features")
    
    print(f"\n🎯 Target Variable Consistency:")
    target_distributions = {}
    for stage, result in results.items():
        if result and result['target_distribution'] is not None:
            target_distributions[stage] = result['target_distribution']
    
    if len(target_distributions) > 1:
        # Check if target distributions are consistent
        first_dist = target_distributions[stages[0]]
        consistent = True
        for stage, dist in target_distributions.items():
            if not dist.equals(first_dist):
                consistent = False
                break
        
        if consistent:
            print("✅ Target variable distribution is consistent across all stages")
        else:
            print("⚠️  Target variable distribution varies across stages")
    
    print(f"\n🔧 Feature Evolution:")
    feature_counts = [result['shape'][1] for result in results.values() if result]
    if len(feature_counts) > 1:
        print("Feature count progression:")
        for i, (stage, result) in enumerate(results.items()):
            if result:
                if i == 0:
                    print(f"  {stage}: {result['shape'][1]} features (baseline)")
                else:
                    prev_features = feature_counts[i-1]
                    new_features = result['shape'][1] - prev_features
                    print(f"  {stage}: {result['shape'][1]} features (+{new_features} new)")

def generate_data_report():
    """
    Generate a comprehensive data validation report.
    """
    print("🔍 STUDENT DROPOUT PREDICTION - DATA VALIDATION REPORT")
    print("=" * 60)
    
    # Define data files
    data_files = {
        'Stage1_data.csv': 'Stage 1 (Applicant & Course Information)',
        'Stage2_data.csv': 'Stage 2 (Student Engagement Data)',
        'Stage3_data.csv': 'Stage 3 (Academic Performance Data)'
    }
    
    # Validate each file
    results = {}
    for filename, stage_name in data_files.items():
        results[stage_name] = validate_data_file(filename, stage_name)
    
    # Compare stages
    compare_stages(results)
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    valid_files = sum(1 for result in results.values() if result is not None)
    total_files = len(data_files)
    
    print(f"✅ Valid files: {valid_files}/{total_files}")
    
    if valid_files == total_files:
        print("🎉 All data files are ready for machine learning pipeline!")
        print("\n📋 Next steps:")
        print("1. Run: python student_dropout_prediction.py")
        print("2. Or run: python example_usage.py")
    else:
        print("⚠️  Some data files have issues. Please review the validation results above.")
    
    return results

def main():
    """Main function to run data validation."""
    try:
        results = generate_data_report()
        return results
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return None

if __name__ == "__main__":
    main()
