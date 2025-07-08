# Employee Attrition Prediction with Spark MLlib

Projet noté en Scala de Machine Learning à l'EPSI Grenoble pour l'année 2025.

## Objective
The goal of this project is to build a machine learning model using Scala and Spark MLlib to predict whether an employee will leave the company (attrition).

## Tech Stack
- **Language:** Scala 2.12.15
- **Framework:** Apache Spark 3.3.0 (SQL & MLlib)
- **Build Tool:** sbt 1.11.2

## Project Structure
```
.
├── .cursor/            # Cursor AI rules and configuration
├── data/               # Raw data files (e.g., HR-Employee-Attrition.csv)
├── project/            # sbt build-related files
├── src/
│   └── main/
│       └── scala/
│           └── com/
│               └── example/
│                   └── Main.scala  # Main Spark application
├── target/             # Compiled code and build artifacts (ignored by Git)
├── .gitignore          # Specifies files to be ignored by Git
├── build.sbt           # sbt build definition file
└── README.md           # This file
```

## How to Run
1.  Ensure you have Java 17+ and `sbt` installed.
2.  Place the `HR-Employee-Attrition.csv` file in the `data/` directory.
3.  Run the application from the project root using the command:
    ```bash
    sbt run
    ```

## Methodology
The project follows a standard machine learning pipeline approach:
1.  **Data Loading:** The CSV data is loaded into a Spark DataFrame using a predefined schema.
2.  **Preprocessing:**
    - Irrelevant columns (`EmployeeCount`, `EmployeeNumber`, etc.) are dropped.
    - Categorical string columns (`JobRole`, `Gender`, etc.) are converted to numerical format using `StringIndexer` and `OneHotEncoder`.
    - The target `Attrition` column is indexed to a numerical `label`.
3.  **Feature Assembling:** All processed feature columns are assembled into a single `features` vector.
4.  **Model Training:**
    - The data is split into an 80% training set and a 20% test set.
    - A `LogisticRegression` model is trained on the training data.
5.  **Evaluation:** The model's performance is evaluated on the test set using the Area Under ROC (AUC) metric.
