#!/usr/bin/env python3
"""
Unit tests for Student Dropout Prediction Model

Run with: pytest test_student_dropout.py -v
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from student_dropout_prediction import StudentDropoutPredictor


class TestStudentDropoutPredictor:
    """Test suite for StudentDropoutPredictor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100

        data = {
            'LearnerCode': [f'L{i:04d}' for i in range(n_samples)],
            'DateofBirth': ['01/01/1990'] * n_samples,
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Nationality': np.random.choice(['UK', 'International'], n_samples),
            'Course': np.random.choice(['Course A', 'Course B'], n_samples),
            'Score': np.random.randint(50, 100, n_samples),
            'CompletedCourse': np.random.choice(['Yes', 'No'], n_samples)
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def temp_data_files(self, sample_data):
        """Create temporary CSV files for testing."""
        temp_dir = tempfile.mkdtemp()

        # Save sample data to temporary files
        stage1_path = Path(temp_dir) / 'stage1.csv'
        stage2_path = Path(temp_dir) / 'stage2.csv'
        stage3_path = Path(temp_dir) / 'stage3.csv'

        sample_data.to_csv(stage1_path, index=False)
        sample_data.to_csv(stage2_path, index=False)
        sample_data.to_csv(stage3_path, index=False)

        yield str(stage1_path), str(stage2_path), str(stage3_path)

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def predictor(self):
        """Create a StudentDropoutPredictor instance."""
        return StudentDropoutPredictor()

    def test_initialization(self, predictor):
        """Test that the predictor initializes correctly."""
        assert predictor.models == {}
        assert predictor.preprocessors == {}
        assert predictor.results == {}
        assert predictor.config is not None
        assert 'random_state' in predictor.config

    def test_default_config(self, predictor):
        """Test that default configuration is correct."""
        config = predictor._default_config()

        assert config['random_state'] == 42
        assert config['test_size'] == 0.2
        assert config['target_column'] == 'CompletedCourse'
        assert config['target_mapping'] == {'No': 0, 'Yes': 1}
        assert 'xgboost_params' in config
        assert 'neural_network_params' in config

    def test_custom_config(self):
        """Test that custom configuration is accepted."""
        custom_config = {
            'random_state': 123,
            'test_size': 0.3,
            'target_column': 'CompletedCourse',
            'target_mapping': {'No': 0, 'Yes': 1}
        }

        predictor = StudentDropoutPredictor(config=custom_config)
        assert predictor.config['random_state'] == 123
        assert predictor.config['test_size'] == 0.3

    def test_load_data(self, predictor, temp_data_files):
        """Test data loading functionality."""
        stage1, stage2, stage3 = temp_data_files

        predictor.load_data(stage1, stage2, stage3)

        assert predictor.s1_data is not None
        assert predictor.s2_data is not None
        assert predictor.s3_data is not None
        assert len(predictor.s1_data) == 100
        assert 'CompletedCourse' in predictor.s1_data.columns

    def test_data_validation(self, predictor, temp_data_files):
        """Test that data validation works correctly."""
        stage1, stage2, stage3 = temp_data_files

        # This should work without errors
        predictor.load_data(stage1, stage2, stage3)

    def test_data_validation_missing_column(self, predictor, sample_data):
        """Test that validation catches missing required columns."""
        temp_dir = tempfile.mkdtemp()

        # Create data without target column
        bad_data = sample_data.drop('CompletedCourse', axis=1)
        bad_path = Path(temp_dir) / 'bad.csv'
        bad_data.to_csv(bad_path, index=False)

        with pytest.raises(ValueError, match="missing required column"):
            predictor.s1_data = pd.read_csv(bad_path)
            predictor.s2_data = pd.read_csv(bad_path)
            predictor.s3_data = pd.read_csv(bad_path)
            predictor._validate_data()

        shutil.rmtree(temp_dir)

    def test_preprocess_stage1(self, predictor, temp_data_files):
        """Test Stage 1 preprocessing."""
        stage1, stage2, stage3 = temp_data_files
        predictor.load_data(stage1, stage2, stage3)

        X_train, X_test, y_train, y_test = predictor.preprocess_stage1()

        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        assert len(X_train) > len(X_test)  # Train should be larger
        assert 'CompletedCourse' not in X_train.columns

    def test_preprocess_stage2(self, predictor, temp_data_files):
        """Test Stage 2 preprocessing."""
        stage1, stage2, stage3 = temp_data_files
        predictor.load_data(stage1, stage2, stage3)

        X_train, X_test, y_train, y_test = predictor.preprocess_stage2()

        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None

    def test_preprocess_stage3(self, predictor, temp_data_files):
        """Test Stage 3 preprocessing."""
        stage1, stage2, stage3 = temp_data_files
        predictor.load_data(stage1, stage2, stage3)

        X_train, X_test, y_train, y_test = predictor.preprocess_stage3()

        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None

    def test_create_preprocessing_pipeline(self, predictor, temp_data_files):
        """Test preprocessing pipeline creation."""
        stage1, stage2, stage3 = temp_data_files
        predictor.load_data(stage1, stage2, stage3)
        predictor.preprocess_stage1()

        pipeline = predictor.create_preprocessing_pipeline('s1')

        assert pipeline is not None
        # Pipeline should have the expected steps
        assert len(pipeline.steps) == 3

    def test_model_save_load(self, predictor, temp_data_files):
        """Test model saving and loading functionality."""
        stage1, stage2, stage3 = temp_data_files
        predictor.load_data(stage1, stage2, stage3)
        predictor.preprocess_stage1()

        # Train a simple model
        predictor.train_xgboost_model('s1', hyperparameter_tuning=False)

        # Save model
        temp_dir = tempfile.mkdtemp()
        predictor.save_model('xgboost_s1', output_dir=temp_dir)

        # Load model into new predictor
        new_predictor = StudentDropoutPredictor()
        new_predictor.load_model('xgboost_s1', input_dir=temp_dir)

        assert 'xgboost_s1' in new_predictor.models
        assert 'xgboost_s1' in new_predictor.preprocessors

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_save_model_not_found(self, predictor):
        """Test that saving non-existent model raises error."""
        with pytest.raises(ValueError, match="not found"):
            predictor.save_model('non_existent_model')

    def test_load_model_not_found(self, predictor):
        """Test that loading non-existent model raises error."""
        with pytest.raises(FileNotFoundError):
            predictor.load_model('non_existent_model', input_dir='/tmp/non_existent')

    def test_xgboost_training(self, predictor, temp_data_files):
        """Test XGBoost model training."""
        stage1, stage2, stage3 = temp_data_files
        predictor.load_data(stage1, stage2, stage3)
        predictor.preprocess_stage1()

        # Train without hyperparameter tuning for speed
        model, pipeline = predictor.train_xgboost_model('s1', hyperparameter_tuning=False)

        assert model is not None
        assert pipeline is not None
        assert 'xgboost_s1' in predictor.models
        assert 'xgboost_s1' in predictor.preprocessors
        assert 'xgboost_s1' in predictor.results

        # Check results structure
        results = predictor.results['xgboost_s1']
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'auc' in results

    def test_generate_report(self, predictor, temp_data_files):
        """Test report generation."""
        stage1, stage2, stage3 = temp_data_files
        predictor.load_data(stage1, stage2, stage3)
        predictor.preprocess_stage1()
        predictor.train_xgboost_model('s1', hyperparameter_tuning=False)

        # Should not raise any errors
        predictor.generate_report()

    def test_save_all_models(self, predictor, temp_data_files):
        """Test saving all models at once."""
        stage1, stage2, stage3 = temp_data_files
        predictor.load_data(stage1, stage2, stage3)
        predictor.preprocess_stage1()
        predictor.train_xgboost_model('s1', hyperparameter_tuning=False)

        temp_dir = tempfile.mkdtemp()
        predictor.save_all_models(output_dir=temp_dir)

        # Check that files were created
        model_files = list(Path(temp_dir).glob('*.pkl'))
        assert len(model_files) > 0

        # Cleanup
        shutil.rmtree(temp_dir)


class TestConfigurationManagement:
    """Test configuration management features."""

    def test_config_from_dict(self):
        """Test creating predictor with custom config."""
        config = {
            'random_state': 999,
            'test_size': 0.15,
            'target_column': 'CompletedCourse',
            'target_mapping': {'No': 0, 'Yes': 1}
        }

        predictor = StudentDropoutPredictor(config=config)
        assert predictor.config['random_state'] == 999
        assert predictor.config['test_size'] == 0.15


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
