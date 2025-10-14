# Student Dropout Prediction Model

A comprehensive machine learning project that predicts student dropout rates using supervised learning techniques. This project analyzes student data across three stages of their academic journey and employs XGBoost and Neural Network models to predict dropout likelihood.

## 🎯 Project Overview

Student retention is a critical challenge in educational institutions. High dropout rates can lead to significant revenue loss, diminished institutional reputation, and lower overall student satisfaction. This project addresses this challenge by developing predictive models that can identify students at risk of dropping out early in their academic journey.

### Key Features

- **Multi-stage Analysis**: Analyzes student data across three distinct stages:
  - Stage 1: Applicant and course information
  - Stage 2: Student engagement data
  - Stage 3: Academic performance data
- **Multiple ML Models**: Implements both XGBoost and Neural Network approaches
- **Comprehensive Evaluation**: Provides detailed performance metrics and visualizations
- **Automated Pipeline**: End-to-end machine learning pipeline with preprocessing and feature engineering

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd student-dropout-prediction
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Option 1: Run with Local Data Files

1. Place your CSV data files in the project directory:
   - `Stage1_data.csv` (applicant and course information)
   - `Stage2_data.csv` (student engagement data)
   - `Stage3_data.csv` (academic performance data)

2. (Optional) Run a quick demo to see what the project does:
   ```bash
   python demo.py
   ```

3. (Optional) Validate your data first:
   ```bash
   python data_validation.py
   ```

4. Run the main script:
   ```bash
   python student_dropout_prediction.py
   ```

#### Option 2: Run with Default Data Sources

If local files are not available, the script will automatically download data from the default sources:

```bash
python student_dropout_prediction.py
```

#### Option 3: Use as a Library

```python
from student_dropout_prediction import StudentDropoutPredictor

# Initialize predictor
predictor = StudentDropoutPredictor()

# Load data
predictor.load_data(
    stage1_path="path/to/stage1.csv",
    stage2_path="path/to/stage2.csv", 
    stage3_path="path/to/stage3.csv"
)

# Run full pipeline
predictor.run_full_pipeline()

# Access results
results = predictor.results
```

## 📊 Data Structure

### Stage 1: Applicant and Course Information
- Student demographics
- Course selection
- Application details
- Target variable: Course completion status

### Stage 2: Student Engagement Data
- Attendance records
- Engagement metrics
- Additional demographic information

### Stage 3: Academic Performance Data
- Module assessment results
- Academic performance indicators
- Detailed engagement metrics

## 🔧 Model Architecture

### XGBoost Model
- Gradient boosting classifier
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Robust performance across all stages

### Neural Network Model
- Multi-layer perceptron with dropout regularization
- Early stopping to prevent overfitting
- Hyperparameter optimization
- Comprehensive training history tracking

## 📈 Performance Metrics

The models are evaluated using multiple metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity to positive cases
- **AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification breakdown

## 🛠️ Technical Details

### Preprocessing Pipeline
- **Data Cleaning**: Removal of high-cardinality and missing data columns
- **Feature Engineering**: Age calculation from date of birth
- **Encoding**: One-hot encoding for categorical variables
- **Scaling**: StandardScaler for numerical features
- **Imputation**: Strategic handling of missing values

### Model Training
- **Cross-validation**: 5-fold cross-validation for hyperparameter tuning
- **Early Stopping**: Prevents overfitting in neural networks
- **Grid Search**: Systematic hyperparameter optimization
- **Reproducibility**: Fixed random seeds for consistent results

## 📁 Project Structure

```
student-dropout-prediction/
├── student_dropout_prediction.py  # Main implementation
├── example_usage.py              # Example usage script
├── data_validation.py            # Data validation and insights
├── demo.py                       # Quick demo script
├── test_installation.py          # Installation test script
├── setup.py                      # Automated setup script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # MIT License
├── .gitignore                    # Git ignore rules
├── Stage1_data.csv              # Stage 1 data (applicant info)
├── Stage2_data.csv              # Stage 2 data (engagement)
└── Stage3_data.csv              # Stage 3 data (performance)
```

## 🔍 Data Validation

Before running the machine learning pipeline, you can validate your data using the included validation script:

```bash
python data_validation.py
```

This script provides:
- **Data Quality Assessment**: Missing values, duplicates, data types
- **Target Variable Analysis**: Class distribution and imbalance detection
- **Feature Analysis**: Categorical vs numerical features, high cardinality detection
- **Stage Comparison**: Consistency checks across all three data stages
- **Data Insights**: Summary statistics and recommendations

### Sample Validation Output
- ✅ **25,059 students** across all three stages
- ✅ **Consistent target distribution**: 85.02% completion rate, 14.98% dropout rate
- ✅ **Progressive feature addition**: 16 → 18 → 21 features across stages
- ⚠️ **Moderate class imbalance**: 14.98% minority class (dropouts)

## 🔍 Key Insights

### Model Performance
- **Stage 1**: Baseline performance with limited features
- **Stage 2**: Improved performance with engagement data
- **Stage 3**: Best performance with academic performance indicators

### Feature Importance
- Nationality and course selection are strong predictors
- Academic performance metrics significantly improve predictions
- Engagement data provides valuable intermediate insights

### Business Impact
- Early identification of at-risk students
- Targeted intervention strategies
- Improved student retention rates
- Data-driven decision making

## 🚨 Important Notes

### Data Privacy
- All student data should be anonymized
- Follow institutional data protection policies
- Ensure compliance with privacy regulations

### Model Limitations
- Performance may vary with different datasets
- Requires regular retraining with new data
- Consider model interpretability for educational settings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

If you encounter any issues or have questions:

1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Include error messages and system information

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔮 Future Enhancements

- [ ] Real-time prediction API
- [ ] Interactive dashboard
- [ ] Additional ML algorithms (Random Forest, SVM)
- [ ] Feature selection optimization
- [ ] Model interpretability tools
- [ ] Automated retraining pipeline

## 📚 References

- XGBoost Documentation: https://xgboost.readthedocs.io/
- TensorFlow Documentation: https://www.tensorflow.org/
- Scikit-learn Documentation: https://scikit-learn.org/

---

**Author**: Matthew Russell  
**Last Updated**: 2024  
**Version**: 1.0.0
# Student-Dropout
