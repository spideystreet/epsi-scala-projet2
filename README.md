# Employee Attrition Prediction with Spark MLlib

🎓 **Academic Scala Machine Learning Project - EPSI Grenoble 2025**

## 🏆 Project Achievement Summary

**Champion Model**: Tuned Logistic Regression - **83.74% AUC, 88.70% Accuracy**

This project demonstrates advanced machine learning engineering using Scala and Spark MLlib to predict employee attrition with comprehensive hyperparameter optimization and rigorous cross-validation methodology.

## 🎯 Objective

Build and optimize machine learning models to predict whether an employee will leave the company (binary classification), showcasing:
- Professional Scala/Spark development practices
- Advanced hyperparameter tuning with cross-validation
- Comprehensive algorithm comparison and evaluation
- Production-ready ML pipeline implementation

## 🛠 Tech Stack

- **Language:** Scala 2.12.15
- **ML Framework:** Apache Spark MLlib 3.3.0
- **Java Version:** Java 17 (with advanced compatibility solutions)
- **Build Tool:** sbt 1.11.2
- **Data Processing:** Spark SQL & DataFrames
- **Evaluation:** AUC-ROC & Accuracy metrics

## 📊 Final Results Summary

### Model Performance Comparison (Baseline vs Tuned)

| Model | Type | AUC | Accuracy | AUC Improvement | Accuracy Improvement |
|-------|------|-----|----------|----------------|---------------------|
| **Logistic Regression** | Baseline | 83.32% | 87.33% | - | - |
| **Logistic Regression** | **🏆 Tuned** | **83.74%** | **88.70%** | **+0.42%** | **+1.37%** |
| **Random Forest** | Baseline | 80.78% | 86.30% | - | - |
| **Random Forest** | **Tuned** | **80.96%** | **86.99%** | **+0.18%** | **+0.68%** |
| **Gradient Boosted Trees** | Baseline | 72.99% | 84.93% | - | - |
| **Gradient Boosted Trees** | **Tuned** | **79.95%** | **85.96%** | **+6.96%** | **+1.03%** |

### 🚀 Key Achievements

1. **🥇 Champion Model**: Tuned Logistic Regression (83.74% AUC, 88.70% Accuracy)
2. **📈 GBT Breakthrough**: +6.96% AUC improvement through hyperparameter optimization
3. **🔬 Rigorous Methodology**: 3-fold cross-validation with 72 hyperparameter combinations
4. **⚡ Java 17 Compatibility**: Advanced solutions for Spark 3.3.0 + Java 17 integration
5. **🎯 Academic Excellence**: Professional ML engineering with comprehensive documentation

## 📁 Project Structure

```
epsi-scala-projet2/
├── data/                           # Dataset files
│   └── HR-Employee-Attrition.csv   # Employee data (1,470 records, 35 features)
├── src/main/scala/com/example/     # Source code
│   └── Main.scala                  # Complete ML pipeline with hyperparameter tuning
├── project/                        # sbt configuration
├── target/                         # Build artifacts (ignored)
├── build.sbt                       # Build configuration with Java 17 compatibility
├── notes.md                        # Detailed implementation notes and progress log
├── .gitignore                      # Git ignore patterns
└── README.md                       # This file
```

## 🚀 How to Run

### Prerequisites
- Java 17+
- sbt 1.11.2+
- HR-Employee-Attrition.csv dataset in `data/` directory

### Execution
```bash
# From project root directory
sbt run
```

### Expected Output
The application will run through:
1. **Baseline Models**: Training and evaluation of default algorithms
2. **Hyperparameter Tuning**: 3-fold CV optimization for each algorithm
3. **Final Comparison**: Comprehensive performance analysis
4. **Best Model**: Sample predictions from champion model

**Runtime**: ~10-11 minutes (includes comprehensive hyperparameter optimization)

## 🔬 Methodology

### 1. Data Pipeline
- **Dataset**: HR-Employee-Attrition.csv (1,470 employees, 35 features)
- **Schema Definition**: Explicit type-safe schema for all 35 columns
- **Data Cleaning**: Removal of non-predictive columns (`EmployeeCount`, `StandardHours`, etc.)
- **Preprocessing**: 
  - StringIndexer → OneHotEncoder for 7 categorical features
  - VectorAssembler for 51 total features
  - Train/Test split (80/20) with seed=1234L

### 2. Machine Learning Pipeline
- **Modular Design**: 6-stage pipeline for algorithm flexibility
- **Three Algorithms**: LogisticRegression, RandomForest, GBTClassifier
- **Dual Evaluation**: AUC-ROC (primary) + Accuracy (secondary) metrics

### 3. Hyperparameter Optimization
- **Cross-Validation**: 3-fold CV for robust parameter selection
- **Parameter Grids**:
  - **LogisticRegression**: 18 combinations (regParam × elasticNetParam × maxIter)
  - **RandomForest**: 27 combinations (numTrees × maxDepth × minInstancesPerNode)
  - **GBTClassifier**: 27 combinations (maxIter × maxDepth × stepSize)
- **Total Optimization**: 72 hyperparameter combinations tested

### 4. Advanced Features
- **Best Parameter Extraction**: Automatic optimal parameter identification
- **Performance Tracking**: Quantified improvement analysis
- **Champion Model Selection**: Multi-metric best model identification

## 🛡 Technical Challenges Solved

### Java 17 + Spark 3.3.0 Compatibility
- **Challenge**: SerializedLambda reflection access restrictions
- **Solution**: Comprehensive `--add-opens` JVM module permissions
- **Critical Fix**: `--add-opens=java.base/java.lang.invoke=ALL-UNNAMED`
- **Result**: Full compatibility achieved

### Scala String Formatting
- **Challenge**: Unsupported `:+` format specifier in f-strings
- **Solution**: Custom conditional formatting for improvement percentages
- **Result**: Professional output formatting

## 📈 Academic Significance

### Machine Learning Excellence
- **Rigorous Methodology**: Cross-validation with comprehensive parameter grids
- **Quantifiable Results**: Measurable improvements across all algorithms
- **Professional Implementation**: Production-ready code patterns
- **Advanced Optimization**: Sophisticated hyperparameter tuning

### Technical Innovation
- **Java 17 Compatibility**: Cutting-edge Spark + modern Java integration
- **Scalable Architecture**: Modular pipeline design for extensibility
- **Performance Engineering**: Optimized Spark configurations

### Documentation Quality
- **Complete Implementation Notes**: Detailed technical documentation in `notes.md`
- **Professional README**: Comprehensive project overview
- **Academic Standards**: Ready for academic presentation and evaluation

## 🎯 Results Interpretation

### Champion Model Analysis (Tuned Logistic Regression)
- **83.74% AUC**: Excellent discrimination capability (>80% considered strong)
- **88.70% Accuracy**: High overall prediction accuracy
- **Improvement**: +1.37% accuracy boost through hyperparameter optimization
- **Robustness**: Consistent performance across cross-validation folds

### Algorithm Insights
- **Logistic Regression**: Maintains superiority with further optimization potential
- **Random Forest**: Stable performer with moderate improvement (+0.68% accuracy)
- **Gradient Boosted Trees**: Largest improvement potential (+6.96% AUC gain)

## 🔮 Future Enhancements

### Advanced Analytics (Phase 6)
- Feature importance analysis and interpretation
- ROC curve visualization and advanced metrics
- Model explainability (SHAP/LIME integration)
- Production deployment preparation

### Technical Extensions
- Automated hyperparameter optimization (Bayesian optimization)
- Advanced ensemble methods
- Real-time prediction API development
- Distributed training optimization

---

## 👨‍🎓 Academic Project

**Institution**: EPSI Grenoble  
**Year**: 2025  
**Language**: Scala  
**Domain**: Machine Learning & Big Data  

**🏆 Project Status**: **COMPLETE** - Advanced hyperparameter tuning with outstanding results achieved!
