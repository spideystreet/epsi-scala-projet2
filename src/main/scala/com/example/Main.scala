package com.example

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Employee Attrition Prediction")
      .master("local[*]") // Use all available cores on the local machine
      .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer") // Use Java serialization instead of Kryo for Java 17 compatibility
      .config("spark.sql.adaptive.enabled", "false") // Disable adaptive query execution to avoid Kryo issues
      .config("spark.sql.adaptive.coalescePartitions.enabled", "false") // Disable partition coalescing
      .config("spark.sql.adaptive.skewJoin.enabled", "false") // Disable skew join optimization
      .getOrCreate()

    println("Spark session created successfully.")

    // According to our best practices, we define a schema to ensure type safety and performance
    val schema = StructType(Array(
      StructField("Age", IntegerType, nullable = false),
      StructField("Attrition", StringType, nullable = false),
      StructField("BusinessTravel", StringType, nullable = false),
      StructField("DailyRate", IntegerType, nullable = false),
      StructField("Department", StringType, nullable = false),
      StructField("DistanceFromHome", IntegerType, nullable = false),
      StructField("Education", IntegerType, nullable = false),
      StructField("EducationField", StringType, nullable = false),
      StructField("EmployeeCount", IntegerType, nullable = false),
      StructField("EmployeeNumber", IntegerType, nullable = false),
      StructField("EnvironmentSatisfaction", IntegerType, nullable = false),
      StructField("Gender", StringType, nullable = false),
      StructField("HourlyRate", IntegerType, nullable = false),
      StructField("JobInvolvement", IntegerType, nullable = false),
      StructField("JobLevel", IntegerType, nullable = false),
      StructField("JobRole", StringType, nullable = false),
      StructField("JobSatisfaction", IntegerType, nullable = false),
      StructField("MaritalStatus", StringType, nullable = false),
      StructField("MonthlyIncome", IntegerType, nullable = false),
      StructField("MonthlyRate", IntegerType, nullable = false),
      StructField("NumCompaniesWorked", IntegerType, nullable = false),
      StructField("Over18", StringType, nullable = false),
      StructField("OverTime", StringType, nullable = false),
      StructField("PercentSalaryHike", IntegerType, nullable = false),
      StructField("PerformanceRating", IntegerType, nullable = false),
      StructField("RelationshipSatisfaction", IntegerType, nullable = false),
      StructField("StandardHours", IntegerType, nullable = false),
      StructField("StockOptionLevel", IntegerType, nullable = false),
      StructField("TotalWorkingYears", IntegerType, nullable = false),
      StructField("TrainingTimesLastYear", IntegerType, nullable = false),
      StructField("WorkLifeBalance", IntegerType, nullable = false),
      StructField("YearsAtCompany", IntegerType, nullable = false),
      StructField("YearsInCurrentRole", IntegerType, nullable = false),
      StructField("YearsSinceLastPromotion", IntegerType, nullable = false),
      StructField("YearsWithCurrManager", IntegerType, nullable = false)
    ))

    // Load the data, applying the schema and specifying that there's a header
    val dataPath = "data/HR-Employee-Attrition.csv"
    val df = spark.read
      .option("header", "true")
      .schema(schema)
      .csv(dataPath)

    // Drop columns that are not useful for prediction
    val cleanedDf = df.drop("EmployeeCount", "StandardHours", "Over18", "EmployeeNumber")

    // Identify categorical and numerical columns for feature processing
    val categoricalCols = Array("BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime")
    // Note: We keep original integer columns as is, as they are already numerical.
    val numericalCols = cleanedDf.columns.filterNot(c => categoricalCols.contains(c) || c == "Attrition")

    // --- Define the ML Pipeline Stages ---

    // Stage 1: Indexers for all categorical columns
    val indexers = categoricalCols.map { colName =>
      new StringIndexer().setInputCol(colName).setOutputCol(colName + "_index").setHandleInvalid("keep")
    }

    // Stage 2: Encoders for all indexed categorical columns
    val encoders = categoricalCols.map { colName =>
      new OneHotEncoder().setInputCol(colName + "_index").setOutputCol(colName + "_vec")
    }

    // Stage 3: Indexer for the label column
    val labelIndexer = new StringIndexer().setInputCol("Attrition").setOutputCol("label")

    // Stage 4: Assembler to combine all feature columns into a single vector
    val assemblerInputs = numericalCols ++ categoricalCols.map(_ + "_vec")
    val assembler = new VectorAssembler()
      .setInputCols(assemblerInputs)
      .setOutputCol("features")

    // Stage 5: The classification model
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")

    // --- Train and Evaluate the Model ---

    // Split the data into training and test sets
    val Array(trainingData, testData) = cleanedDf.randomSplit(Array(0.8, 0.2), seed = 1234L)

    // Create the full pipeline
    val pipeline = new Pipeline().setStages(indexers ++ encoders ++ Array(labelIndexer, assembler, lr))

    // Train the model
    println("Training the Logistic Regression model...")
    val model = pipeline.fit(trainingData)

    // Make predictions on the test data
    println("Making predictions on the test data...")
    val predictions = model.transform(testData)

    // Show a sample of the predictions
    println("Sample predictions:")
    predictions.select("label", "prediction", "probability").show(10, truncate = false)

    // --- Evaluate the Model ---
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction") // default is rawPrediction
      .setMetricName("areaUnderROC")

    val auc = evaluator.evaluate(predictions)
    println(s"Area Under ROC Curve (AUC) on test data = $auc")

    spark.stop()
  }
} 