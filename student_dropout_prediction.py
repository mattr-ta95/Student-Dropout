# -*- coding: utf-8 -*-
"""
Student Dropout Prediction Model

A machine learning project that predicts student dropout rates using supervised learning techniques.
This project analyzes student data across three stages of their academic journey and employs
XGBoost and Neural Network models to predict dropout likelihood.

Author: Matthew Russell
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import joblib
import json
from typing import Optional, Dict, Tuple, Any
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, recall_score, f1_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from dateutil.relativedelta import relativedelta
from datetime import datetime
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings only
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Optional imports for neural networks
try:
    import tensorflow as tf
    import keras
    NEURAL_NETWORK_AVAILABLE = True
except ImportError:
    NEURAL_NETWORK_AVAILABLE = False
    logger.warning("TensorFlow/Keras not available. Neural network models will be skipped.")

class StudentDropoutPredictor:
    """
    A comprehensive machine learning pipeline for predicting student dropout rates.
    
    This class handles data preprocessing, feature engineering, model training,
    and evaluation across three stages of student data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the StudentDropoutPredictor.

        Args:
            config (dict, optional): Configuration dictionary with model parameters
        """
        self.models = {}
        self.preprocessors = {}
        self.results = {}
        self.config = config or self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'random_state': 42,
            'test_size': 0.2,
            'target_column': 'CompletedCourse',
            'target_mapping': {'No': 0, 'Yes': 1},
            'xgboost_params': {
                'learning_rate': [0.1, 0.2, 0.3],
                'max_depth': [3, 4, 5],
                'n_estimators': [100, 200, 300]
            },
            'neural_network_params': {
                'learning_rates': [0.01, 0.1],
                'neurons': [8, 12],
                'activations': ['relu', 'tanh'],
                'optimizers': ['adam']
            }
        }
        
    def load_data(self, stage1_path: Optional[str] = None, stage2_path: Optional[str] = None,
                  stage3_path: Optional[str] = None) -> None:
        """
        Load student data from CSV files or use default Google Drive URLs.

        Args:
            stage1_path (str, optional): Path to Stage 1 data CSV file
            stage2_path (str, optional): Path to Stage 2 data CSV file
            stage3_path (str, optional): Path to Stage 3 data CSV file

        Raises:
            FileNotFoundError: If data files cannot be loaded
            ValueError: If data is invalid or missing required columns
        """
        logger.info("Loading student data...")

        # Default URLs if local files not provided
        if stage1_path is None:
            stage1_path = "https://drive.google.com/uc?id=1pA8DDYmQuaLyxADCOZe1QaSQwF16q1J6"
        if stage2_path is None:
            stage2_path = "https://drive.google.com/uc?id=1vy1JFQZva3lhMJQV69C43AB1NTM4W-DZ"
        if stage3_path is None:
            stage3_path = "https://drive.google.com/uc?id=18oyu-RQotQN6jaibsLBoPdqQJbj_cV2-"

        try:
            # Load data files
            self.s1_data = pd.read_csv(stage1_path)
            self.s2_data = pd.read_csv(stage2_path)
            self.s3_data = pd.read_csv(stage3_path)

            # Validate data
            self._validate_data()

            logger.info(f"Stage 1 data shape: {self.s1_data.shape}")
            logger.info(f"Stage 2 data shape: {self.s2_data.shape}")
            logger.info(f"Stage 3 data shape: {self.s3_data.shape}")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _validate_data(self) -> None:
        """Validate that loaded data has required columns and structure."""
        target_col = self.config['target_column']

        for stage_name, data in [('Stage 1', self.s1_data), ('Stage 2', self.s2_data), ('Stage 3', self.s3_data)]:
            if data is None or data.empty:
                raise ValueError(f"{stage_name} data is empty")

            if target_col not in data.columns:
                raise ValueError(f"{stage_name} data missing required column: {target_col}")

            # Check target values
            target_values = data[target_col].unique()
            expected_values = set(self.config['target_mapping'].keys())
            actual_values = set(target_values)

            if not actual_values.issubset(expected_values):
                logger.warning(f"{stage_name} has unexpected target values: {actual_values - expected_values}")

    def explore_data(self, stage='s1'):
        """
        Perform exploratory data analysis on the specified stage.
        
        Args:
            stage (str): Stage to explore ('s1', 's2', or 's3')
        """
        if stage == 's1':
            data = self.s1_data
        elif stage == 's2':
            data = self.s2_data
        elif stage == 's3':
            data = self.s3_data
        else:
            raise ValueError("Stage must be 's1', 's2', or 's3'")
            
        print(f"\n=== {stage.upper()} Data Exploration ===")
        print(f"Data shape: {data.shape}")
        print(f"Data types:\n{data.dtypes}")
        print(f"Missing values:\n{data.isnull().sum()}")
        print(f"Unique values per column:\n{data.nunique()}")
        
        # Check target variable distribution
        if 'CompletedCourse' in data.columns:
            print(f"Target variable distribution:\n{data['CompletedCourse'].value_counts()}")
            
        return data

    def preprocess_stage1(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Preprocess Stage 1 data (applicant and course information)."""
        logger.info("=== Preprocessing Stage 1 Data ===")

        # Create target variable
        target_col = self.config['target_column']
        target = self.s1_data[target_col].copy()
        target = target.map(self.config['target_mapping'])

        # Remove target from features
        X = self.s1_data.drop([target_col], axis=1)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, target,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )

        # Store processed data
        self.X_train_s1 = X_train
        self.X_test_s1 = X_test
        self.y_train_s1 = y_train
        self.y_test_s1 = y_test

        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def preprocess_stage2(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Preprocess Stage 2 data (student and engagement data)."""
        logger.info("=== Preprocessing Stage 2 Data ===")

        # Create target variable
        target_col = self.config['target_column']
        target = self.s2_data[target_col].copy()
        target = target.map(self.config['target_mapping'])

        # Remove target from features
        X = self.s2_data.drop([target_col], axis=1)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, target,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )

        # Store processed data
        self.X_train_s2 = X_train
        self.X_test_s2 = X_test
        self.y_train_s2 = y_train
        self.y_test_s2 = y_test

        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def preprocess_stage3(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Preprocess Stage 3 data (academic performance data)."""
        logger.info("=== Preprocessing Stage 3 Data ===")

        # Create target variable
        target_col = self.config['target_column']
        target = self.s3_data[target_col].copy()
        target = target.map(self.config['target_mapping'])

        # Remove target from features
        X = self.s3_data.drop([target_col], axis=1)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, target,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )

        # Store processed data
        self.X_train_s3 = X_train
        self.X_test_s3 = X_test
        self.y_train_s3 = y_train
        self.y_test_s3 = y_test

        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def create_preprocessing_pipeline(self, stage):
        """
        Create preprocessing pipeline for the specified stage.
        
        Args:
            stage (str): Stage identifier ('s1', 's2', or 's3')
        """
        if stage == 's1':
            X_train = self.X_train_s1
        elif stage == 's2':
            X_train = self.X_train_s2
        elif stage == 's3':
            X_train = self.X_train_s3
        else:
            raise ValueError("Stage must be 's1', 's2', or 's3'")
        
        # Custom transformer for age calculation
        class AgeCalculator(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                X = X.copy()
                if 'DateofBirth' in X.columns:
                    X['DateofBirth'] = pd.to_datetime(X['DateofBirth'], format='%d/%m/%Y')
                    today = datetime.now()
                    X['Age'] = X['DateofBirth'].apply(
                        lambda dob: relativedelta(today, dob).years if pd.notnull(dob) and isinstance(dob, datetime) else np.nan
                    )
                    X = X.drop(['DateofBirth'], axis=1, errors='ignore')
                return X

        # Identify features
        numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()
        
        # Add Age to numerical features
        numerical_features.append('Age')
        
        # Stage-specific columns to drop
        if stage == 's1':
            columns_to_drop = ['LearnerCode', 'DiscountType', 'HomeState', 'CentreName', 'ProgressionDegree', 'HomeCity']
        elif stage == 's2':
            columns_to_drop = ['LearnerCode', 'DiscountType', 'HomeState', 'ProgressionDegree', 'CentreName', 'HomeCity', 'UnauthorisedAbsenceCount']
        elif stage == 's3':
            columns_to_drop = ['LearnerCode', 'DiscountType', 'HomeState', 'ProgressionDegree', 'CentreName', 'HomeCity', 'UnauthorisedAbsenceCount', 'PassedModules', 'AssessedModules']
        
        # Remove columns_to_drop from feature lists
        categorical_features = [col for col in categorical_features if col not in columns_to_drop]
        numerical_features = [col for col in numerical_features if col not in columns_to_drop]
        
        # Create pipelines
        categorical_pipeline = Pipeline([
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ])
        
        numerical_pipeline = Pipeline([
            ('scaler', StandardScaler()),
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features),
            ],
            remainder='drop'
        )
        
        drop_columns_transformer = FunctionTransformer(lambda X: X.drop(columns_to_drop, axis=1, errors='ignore'))
        
        pipeline = Pipeline([
            ('drop_columns', drop_columns_transformer),
            ('age_calculator', AgeCalculator()),
            ('preprocessor', preprocessor)
        ])
        
        return pipeline

    def train_xgboost_model(self, stage, hyperparameter_tuning=True):
        """
        Train XGBoost model for the specified stage.
        
        Args:
            stage (str): Stage identifier ('s1', 's2', or 's3')
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        """
        print(f"\n=== Training XGBoost Model for Stage {stage.upper()} ===")
        
        # Get data
        if stage == 's1':
            X_train, X_test, y_train, y_test = self.X_train_s1, self.X_test_s1, self.y_train_s1, self.y_test_s1
        elif stage == 's2':
            X_train, X_test, y_train, y_test = self.X_train_s2, self.X_test_s2, self.y_train_s2, self.y_test_s2
        elif stage == 's3':
            X_train, X_test, y_train, y_test = self.X_train_s3, self.X_test_s3, self.y_train_s3, self.y_test_s3
        
        # Create and fit preprocessing pipeline
        pipeline = self.create_preprocessing_pipeline(stage)
        X_train_processed = pipeline.fit_transform(X_train)
        X_test_processed = pipeline.transform(X_test)
        
        # Train XGBoost model
        xgb_model = XGBClassifier(random_state=42)
        
        if hyperparameter_tuning:
            # Hyperparameter tuning
            param_grid = {
                'learning_rate': [0.1, 0.2, 0.3],
                'max_depth': [3, 4, 5],
                'n_estimators': [100, 200, 300]
            }
            
            grid = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5)
            grid.fit(X_train_processed, y_train)
            
            best_model = grid.best_estimator_
            best_params = grid.best_params_
            print(f"Best parameters: {best_params}")
        else:
            best_model = xgb_model
            best_model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = best_model.predict(X_test_processed)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        
        # Store results
        self.models[f'xgboost_{stage}'] = best_model
        self.preprocessors[f'xgboost_{stage}'] = pipeline
        self.results[f'xgboost_{stage}'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'predictions': y_pred,
            'y_test': y_test
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc:.4f}")
        
        return best_model, pipeline

    def create_neural_network(self, input_shape, learning_rate=0.01, neurons=128, activation='relu', optimizer='adam'):
        """
        Create a neural network model.
        
        Args:
            input_shape (int): Number of input features
            learning_rate (float): Learning rate
            neurons (int): Number of neurons in hidden layers
            activation (str): Activation function
            optimizer (str): Optimizer to use
            
        Returns:
            keras.Model: Compiled neural network model
        """
        if not NEURAL_NETWORK_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required for neural network models. Please install: pip install tensorflow")
        
        model = keras.Sequential([
            keras.layers.Dense(units=neurons, activation=activation, input_shape=(input_shape,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=neurons, activation=activation),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=neurons, activation=activation),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=neurons, activation=activation),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_neural_network(self, stage, hyperparameter_tuning=True):
        """
        Train neural network model for the specified stage.
        
        Args:
            stage (str): Stage identifier ('s1', 's2', or 's3')
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        """
        if not NEURAL_NETWORK_AVAILABLE:
            logger.warning(f"Skipping Neural Network training for Stage {stage.upper()} - TensorFlow/Keras not available")
            return None, None, None

        logger.info(f"=== Training Neural Network Model for Stage {stage.upper()} ===")
        
        # Get data
        if stage == 's1':
            X_train, X_test, y_train, y_test = self.X_train_s1, self.X_test_s1, self.y_train_s1, self.y_test_s1
        elif stage == 's2':
            X_train, X_test, y_train, y_test = self.X_train_s2, self.X_test_s2, self.y_train_s2, self.y_test_s2
        elif stage == 's3':
            X_train, X_test, y_train, y_test = self.X_train_s3, self.X_test_s3, self.y_train_s3, self.y_test_s3
        
        # Create and fit preprocessing pipeline
        pipeline = self.create_preprocessing_pipeline(stage)
        X_train_processed = pipeline.fit_transform(X_train)
        X_test_processed = pipeline.transform(X_test)
        
        input_shape = X_train_processed.shape[1]
        
        if hyperparameter_tuning:
            # Hyperparameter tuning (optimized with fewer combinations)
            nn_params = self.config['neural_network_params']
            learning_rates = nn_params['learning_rates']
            neurons_list = nn_params['neurons']
            activations = nn_params['activations']
            optimizers = nn_params['optimizers']

            results = []
            total_combinations = len(learning_rates) * len(neurons_list) * len(activations) * len(optimizers)
            logger.info(f"Testing {total_combinations} hyperparameter combinations...")

            combination_num = 0
            for lr in learning_rates:
                for neuron in neurons_list:
                    for opt in optimizers:
                        for activation in activations:
                            combination_num += 1
                            logger.debug(f"Testing combination {combination_num}/{total_combinations}: "
                                       f"LR={lr}, Neurons={neuron}, Activation={activation}, Optimizer={opt}")

                            model = self.create_neural_network(input_shape, learning_rate=lr, neurons=neuron,
                                                              activation=activation, optimizer=opt)

                            # Train with early stopping
                            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                            history = model.fit(X_train_processed, y_train, epochs=10, validation_split=0.2,
                                              callbacks=[early_stopping], verbose=0)

                            # Evaluate
                            y_pred = (model.predict(X_test_processed, verbose=0) > 0.5).astype(int)
                            accuracy = accuracy_score(y_test, y_pred)

                            results.append({
                                'learning_rate': lr,
                                'neurons': neuron,
                                'activation': activation,
                                'optimizer': opt,
                                'accuracy': accuracy,
                                'model': model,
                                'history': history
                            })

            # Find best model
            best_result = max(results, key=lambda x: x['accuracy'])
            best_model = best_result['model']
            best_history = best_result['history']

            logger.info(f"Best parameters: LR={best_result['learning_rate']}, Neurons={best_result['neurons']}, "
                       f"Activation={best_result['activation']}, Optimizer={best_result['optimizer']}")
        else:
            # Use default parameters
            best_model = self.create_neural_network(input_shape)
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            best_history = best_model.fit(X_train_processed, y_train, epochs=10, validation_split=0.2, callbacks=[early_stopping], verbose=0)
        
        # Make predictions
        y_pred = (best_model.predict(X_test_processed) > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        
        # Store results
        self.models[f'neural_network_{stage}'] = best_model
        self.preprocessors[f'neural_network_{stage}'] = pipeline
        self.results[f'neural_network_{stage}'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'predictions': y_pred,
            'y_test': y_test,
            'history': best_history
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc:.4f}")
        
        return best_model, pipeline, best_history

    def plot_results(self, stage, model_type='xgboost'):
        """
        Plot model results including confusion matrix and training history.
        
        Args:
            stage (str): Stage identifier ('s1', 's2', or 's3')
            model_type (str): Type of model ('xgboost' or 'neural_network')
        """
        model_key = f'{model_type}_{stage}'
        
        if model_key not in self.results:
            print(f"No results found for {model_key}")
            return
        
        results = self.results[model_key]
        
        # Confusion Matrix
        cm = confusion_matrix(results['y_test'], results['predictions'])
        plt.figure(figsize=(8, 6))
        cfmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Dropout", "Dropout"])
        cfmd.plot()
        plt.title(f'{model_type.upper()} Model - Stage {stage.upper()} - Confusion Matrix')
        plt.show()
        
        # Training history for neural networks
        if model_type == 'neural_network' and 'history' in results:
            history = results['history']
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.tight_layout()
            plt.show()

    def save_model(self, model_key: str, output_dir: str = "models") -> None:
        """
        Save a trained model and its preprocessor to disk.

        Args:
            model_key (str): Key of the model to save (e.g., 'xgboost_s1')
            output_dir (str): Directory to save the model files

        Raises:
            ValueError: If model_key not found
        """
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not found. Available models: {list(self.models.keys())}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = output_path / f"{model_key}_model.pkl"
        joblib.dump(self.models[model_key], model_file)
        logger.info(f"Model saved to {model_file}")

        # Save preprocessor
        if model_key in self.preprocessors:
            preprocessor_file = output_path / f"{model_key}_preprocessor.pkl"
            joblib.dump(self.preprocessors[model_key], preprocessor_file)
            logger.info(f"Preprocessor saved to {preprocessor_file}")

        # Save results
        if model_key in self.results:
            results_file = output_path / f"{model_key}_results.json"
            # Convert numpy types to Python types for JSON serialization
            results_to_save = {}
            for key, value in self.results[model_key].items():
                if key not in ['predictions', 'y_test', 'history']:  # Skip non-serializable items
                    if isinstance(value, (np.integer, np.floating)):
                        results_to_save[key] = float(value)
                    else:
                        results_to_save[key] = value

            with open(results_file, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            logger.info(f"Results saved to {results_file}")

    def load_model(self, model_key: str, input_dir: str = "models") -> None:
        """
        Load a trained model and its preprocessor from disk.

        Args:
            model_key (str): Key of the model to load (e.g., 'xgboost_s1')
            input_dir (str): Directory containing the model files

        Raises:
            FileNotFoundError: If model files not found
        """
        input_path = Path(input_dir)

        # Load model
        model_file = input_path / f"{model_key}_model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        self.models[model_key] = joblib.load(model_file)
        logger.info(f"Model loaded from {model_file}")

        # Load preprocessor
        preprocessor_file = input_path / f"{model_key}_preprocessor.pkl"
        if preprocessor_file.exists():
            self.preprocessors[model_key] = joblib.load(preprocessor_file)
            logger.info(f"Preprocessor loaded from {preprocessor_file}")

        # Load results
        results_file = input_path / f"{model_key}_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.results[model_key] = json.load(f)
            logger.info(f"Results loaded from {results_file}")

    def save_all_models(self, output_dir: str = "models") -> None:
        """Save all trained models to disk."""
        for model_key in self.models.keys():
            self.save_model(model_key, output_dir)
        logger.info(f"All models saved to {output_dir}")

    def generate_report(self):
        """Generate a comprehensive report of all model results."""
        logger.info("\n" + "="*60)
        logger.info("STUDENT DROPOUT PREDICTION - MODEL PERFORMANCE REPORT")
        logger.info("="*60)

        for model_key, results in self.results.items():
            stage = model_key.split('_')[-1]
            model_type = '_'.join(model_key.split('_')[:-1])

            logger.info(f"\n{model_type.upper()} - Stage {stage.upper()}:")
            logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
            logger.info(f"  Precision: {results['precision']:.4f}")
            logger.info(f"  Recall:    {results['recall']:.4f}")
            logger.info(f"  AUC:       {results['auc']:.4f}")

        # Find best performing model
        if self.results:
            best_model = max(self.results.items(), key=lambda x: x[1]['auc'])
            logger.info(f"\nBest performing model: {best_model[0]} (AUC: {best_model[1]['auc']:.4f})")

    def run_full_pipeline(self, stage1_path=None, stage2_path=None, stage3_path=None):
        """
        Run the complete machine learning pipeline.
        
        Args:
            stage1_path (str): Path to Stage 1 data CSV file
            stage2_path (str): Path to Stage 2 data CSV file
            stage3_path (str): Path to Stage 3 data CSV file
        """
        print("Starting Student Dropout Prediction Pipeline...")
        
        # Load data
        self.load_data(stage1_path, stage2_path, stage3_path)
        
        # Preprocess all stages
        self.preprocess_stage1()
        self.preprocess_stage2()
        self.preprocess_stage3()
        
        # Train models for all stages
        stages = ['s1', 's2', 's3']
        
        for stage in stages:
            print(f"\n{'='*50}")
            print(f"PROCESSING STAGE {stage.upper()}")
            print(f"{'='*50}")
            
            # Train XGBoost
            self.train_xgboost_model(stage)
            
            # Train Neural Network (if available)
            nn_model, nn_pipeline, nn_history = self.train_neural_network(stage)
            
            # Plot results
            self.plot_results(stage, 'xgboost')
            if nn_model is not None:
                self.plot_results(stage, 'neural_network')
        
        # Generate final report
        self.generate_report()
        
        print("\nPipeline completed successfully!")


def main():
    """Main function to run the student dropout prediction pipeline."""
    # Initialize predictor
    predictor = StudentDropoutPredictor()
    
    # Run full pipeline with local data files
    try:
        predictor.run_full_pipeline(
            stage1_path="Stage1_data.csv",
            stage2_path="Stage2_data.csv",
            stage3_path="Stage3_data.csv"
        )
    except FileNotFoundError:
        print("Local data files not found. Using default URLs...")
        predictor.run_full_pipeline()


if __name__ == "__main__":
    main()
