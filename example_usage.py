#!/usr/bin/env python3
"""
Example usage of the Student Dropout Prediction Model

This script demonstrates how to use the StudentDropoutPredictor class
for predicting student dropout rates.
"""

from student_dropout_prediction import StudentDropoutPredictor
import matplotlib.pyplot as plt

def main():
    """Example usage of the StudentDropoutPredictor."""
    
    print("Student Dropout Prediction - Example Usage")
    print("=" * 50)
    
    # Initialize the predictor
    predictor = StudentDropoutPredictor()
    
    # Option 1: Use local data files (if available)
    try:
        print("\nAttempting to load local data files...")
        predictor.load_data(
            stage1_path="Stage1_data.csv",
            stage2_path="Stage2_data.csv",
            stage3_path="Stage3_data.csv"
        )
        print("✓ Local data files loaded successfully!")
        
    except FileNotFoundError:
        print("⚠ Local data files not found. Using default data sources...")
        predictor.load_data()
        print("✓ Default data sources loaded successfully!")
    
    # Explore the data
    print("\n" + "="*30)
    print("DATA EXPLORATION")
    print("="*30)
    
    for stage in ['s1', 's2', 's3']:
        print(f"\n--- Stage {stage.upper()} Data ---")
        data = predictor.explore_data(stage)
        print(f"Shape: {data.shape}")
        if 'CompletedCourse' in data.columns:
            completion_rate = data['CompletedCourse'].value_counts(normalize=True)
            print(f"Completion Rate: {completion_rate.get('Yes', 0):.2%}")
    
    # Run the full pipeline
    print("\n" + "="*30)
    print("RUNNING ML PIPELINE")
    print("="*30)
    
    # Preprocess all stages
    predictor.preprocess_stage1()
    predictor.preprocess_stage2()
    predictor.preprocess_stage3()
    
    # Train models for each stage
    stages = ['s1', 's2', 's3']
    
    for stage in stages:
        print(f"\n--- Training Models for Stage {stage.upper()} ---")
        
        # Train XGBoost model
        print("Training XGBoost model...")
        xgb_model, xgb_pipeline = predictor.train_xgboost_model(stage, hyperparameter_tuning=True)
        
        # Train Neural Network model
        print("Training Neural Network model...")
        nn_model, nn_pipeline, nn_history = predictor.train_neural_network(stage, hyperparameter_tuning=True)
        
        # Plot results
        print("Generating visualizations...")
        predictor.plot_results(stage, 'xgboost')
        predictor.plot_results(stage, 'neural_network')
    
    # Generate comprehensive report
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    
    predictor.generate_report()
    
    # Additional analysis
    print("\n" + "="*30)
    print("ADDITIONAL ANALYSIS")
    print("="*30)
    
    # Compare model performance across stages
    print("\nModel Performance Comparison:")
    print("-" * 40)
    
    for stage in stages:
        xgb_key = f'xgboost_{stage}'
        nn_key = f'neural_network_{stage}'
        
        if xgb_key in predictor.results and nn_key in predictor.results:
            xgb_auc = predictor.results[xgb_key]['auc']
            nn_auc = predictor.results[nn_key]['auc']
            
            print(f"Stage {stage.upper()}:")
            print(f"  XGBoost AUC: {xgb_auc:.4f}")
            print(f"  Neural Network AUC: {nn_auc:.4f}")
            print(f"  Best Model: {'XGBoost' if xgb_auc > nn_auc else 'Neural Network'}")
            print()
    
    # Feature importance analysis (if available)
    print("Feature Importance Analysis:")
    print("-" * 30)
    
    for stage in stages:
        model_key = f'xgboost_{stage}'
        if model_key in predictor.models:
            model = predictor.models[model_key]
            try:
                # Get feature importance
                importance = model.feature_importances_
                print(f"Stage {stage.upper()} - Top 5 Most Important Features:")
                # Note: Feature names would need to be extracted from the pipeline
                # This is a simplified example
                print("  (Feature names would be displayed here)")
            except AttributeError:
                print(f"Stage {stage.upper()}: Feature importance not available")
    
    print("\n" + "="*50)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\nNext steps:")
    print("1. Review the generated plots and metrics")
    print("2. Analyze the model performance across different stages")
    print("3. Consider feature engineering improvements")
    print("4. Implement the best model for production use")
    
    # Keep plots open
    plt.show()

if __name__ == "__main__":
    main()
