package com.example

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier, GBTClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

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

    // --- Train and Evaluate Multiple Models ---

    // Split the data into training and test sets
    val Array(trainingData, testData) = cleanedDf.randomSplit(Array(0.8, 0.2), seed = 1234L)

    // Create evaluators
    val binaryEvaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")
    
    val multiclassEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    println("\n" + "="*60)
    println("MODEL COMPARISON RESULTS")
    println("="*60)

    // 1. LOGISTIC REGRESSION (Baseline)
    println("\n1. LOGISTIC REGRESSION (Baseline)")
    println("-" * 40)
    
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
    val lrPipeline = new Pipeline().setStages(indexers ++ encoders ++ Array(labelIndexer, assembler, lr))
    
    println("Training Logistic Regression model...")
    val lrModel = lrPipeline.fit(trainingData)
    val lrPredictions = lrModel.transform(testData)
    
    val lrAuc = binaryEvaluator.evaluate(lrPredictions)
    val lrAccuracy = multiclassEvaluator.evaluate(lrPredictions)
    
    println(f"AUC: $lrAuc%.4f")
    println(f"Accuracy: $lrAccuracy%.4f")

    // 2. RANDOM FOREST
    println("\n2. RANDOM FOREST")
    println("-" * 40)
    
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(100)
      .setMaxDepth(5)
      .setSeed(1234L)
    
    val rfPipeline = new Pipeline().setStages(indexers ++ encoders ++ Array(labelIndexer, assembler, rf))
    
    println("Training Random Forest model...")
    val rfModel = rfPipeline.fit(trainingData)
    val rfPredictions = rfModel.transform(testData)
    
    val rfAuc = binaryEvaluator.evaluate(rfPredictions)
    val rfAccuracy = multiclassEvaluator.evaluate(rfPredictions)
    
    println(f"AUC: $rfAuc%.4f")
    println(f"Accuracy: $rfAccuracy%.4f")

    // 3. GRADIENT BOOSTED TREES
    println("\n3. GRADIENT BOOSTED TREES")
    println("-" * 40)
    
    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(20)
      .setMaxDepth(5)
      .setSeed(1234L)
    
    val gbtPipeline = new Pipeline().setStages(indexers ++ encoders ++ Array(labelIndexer, assembler, gbt))
    
    println("Training Gradient Boosted Trees model...")
    val gbtModel = gbtPipeline.fit(trainingData)
    val gbtPredictions = gbtModel.transform(testData)
    
    val gbtAuc = binaryEvaluator.evaluate(gbtPredictions)
    val gbtAccuracy = multiclassEvaluator.evaluate(gbtPredictions)
    
    println(f"AUC: $gbtAuc%.4f")
    println(f"Accuracy: $gbtAccuracy%.4f")

    // --- RESULTS SUMMARY ---
    println("\n" + "="*60)
    println("FINAL COMPARISON SUMMARY")
    println("="*60)
    
    val results = Array(
      ("Logistic Regression", lrAuc, lrAccuracy),
      ("Random Forest", rfAuc, rfAccuracy),
      ("Gradient Boosted Trees", gbtAuc, gbtAccuracy)
    )
    
    println(f"${"Model"}%-20s | ${"AUC"}%-8s | ${"Accuracy"}%-8s")
    println("-" * 42)
    results.foreach { case (name, auc, acc) =>
      println(f"$name%-20s | ${auc}%.4f   | ${acc}%.4f")
    }
    
    // Find best model
    val bestByAuc = results.maxBy(_._2)
    val bestByAccuracy = results.maxBy(_._3)
    
    println(f"\nBest AUC: ${bestByAuc._1} (${bestByAuc._2}%.4f)")
    println(f"Best Accuracy: ${bestByAccuracy._1} (${bestByAccuracy._3}%.4f)")

    // Show sample predictions from best AUC model
    println(f"\nSample predictions from best model (${bestByAuc._1}):")
    val bestPredictions = if (bestByAuc._1 == "Logistic Regression") lrPredictions
                         else if (bestByAuc._1 == "Random Forest") rfPredictions
                         else gbtPredictions
    
    bestPredictions.select("label", "prediction", "probability").show(10, truncate = false)

    spark.stop()
  }
} 