package com.example

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier, GBTClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Employee Attrition Prediction - Hyperparameter Tuning")
      .master("local[*]") // Use all available cores on the local machine
      .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer") // Use Java serialization instead of Kryo for Java 17 compatibility
      .config("spark.sql.adaptive.enabled", "false") // Disable adaptive query execution to avoid Kryo issues
      .config("spark.sql.adaptive.coalescePartitions.enabled", "false") // Disable partition coalescing
      .config("spark.sql.adaptive.skewJoin.enabled", "false") // Disable skew join optimization
      .getOrCreate()

    println("=== Employee Attrition Prediction with Hyperparameter Tuning ===")
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

    println("\n" + "="*80)
    println("PHASE 5: HYPERPARAMETER TUNING COMPARISON")
    println("="*80)
    println("Training baseline models and tuned models for comparison...")

    // ==================== BASELINE MODELS (from Phase 4) ====================
    
    println("\n" + "="*60)
    println("BASELINE MODELS (Default Parameters)")
    println("="*60)

    // 1. BASELINE LOGISTIC REGRESSION
    println("\n1. BASELINE LOGISTIC REGRESSION")
    println("-" * 40)
    
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
    val lrPipeline = new Pipeline().setStages(indexers ++ encoders ++ Array(labelIndexer, assembler, lr))
    
    println("Training baseline Logistic Regression...")
    val lrModel = lrPipeline.fit(trainingData)
    val lrPredictions = lrModel.transform(testData)
    
    val baselineLrAuc = binaryEvaluator.evaluate(lrPredictions)
    val baselineLrAccuracy = multiclassEvaluator.evaluate(lrPredictions)
    
    println(f"Baseline AUC: $baselineLrAuc%.4f")
    println(f"Baseline Accuracy: $baselineLrAccuracy%.4f")

    // 2. BASELINE RANDOM FOREST
    println("\n2. BASELINE RANDOM FOREST")
    println("-" * 40)
    
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(100)
      .setMaxDepth(5)
      .setSeed(1234L)
    
    val rfPipeline = new Pipeline().setStages(indexers ++ encoders ++ Array(labelIndexer, assembler, rf))
    
    println("Training baseline Random Forest...")
    val rfModel = rfPipeline.fit(trainingData)
    val rfPredictions = rfModel.transform(testData)
    
    val baselineRfAuc = binaryEvaluator.evaluate(rfPredictions)
    val baselineRfAccuracy = multiclassEvaluator.evaluate(rfPredictions)
    
    println(f"Baseline AUC: $baselineRfAuc%.4f")
    println(f"Baseline Accuracy: $baselineRfAccuracy%.4f")

    // 3. BASELINE GRADIENT BOOSTED TREES
    println("\n3. BASELINE GRADIENT BOOSTED TREES")
    println("-" * 40)
    
    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(20)
      .setMaxDepth(5)
      .setSeed(1234L)
    
    val gbtPipeline = new Pipeline().setStages(indexers ++ encoders ++ Array(labelIndexer, assembler, gbt))
    
    println("Training baseline Gradient Boosted Trees...")
    val gbtModel = gbtPipeline.fit(trainingData)
    val gbtPredictions = gbtModel.transform(testData)
    
    val baselineGbtAuc = binaryEvaluator.evaluate(gbtPredictions)
    val baselineGbtAccuracy = multiclassEvaluator.evaluate(gbtPredictions)
    
    println(f"Baseline AUC: $baselineGbtAuc%.4f")
    println(f"Baseline Accuracy: $baselineGbtAccuracy%.4f")

    // ==================== HYPERPARAMETER TUNING ====================

    println("\n" + "="*60)
    println("HYPERPARAMETER TUNING WITH CROSS-VALIDATION")
    println("="*60)

    // 1. TUNED LOGISTIC REGRESSION
    println("\n1. TUNING LOGISTIC REGRESSION")
    println("-" * 40)
    
    val lrTuned = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
    val lrTunedPipeline = new Pipeline().setStages(indexers ++ encoders ++ Array(labelIndexer, assembler, lrTuned))
    
    val lrParamGrid = new ParamGridBuilder()
      .addGrid(lrTuned.regParam, Array(0.01, 0.1, 0.3))
      .addGrid(lrTuned.elasticNetParam, Array(0.0, 0.5, 1.0))
      .addGrid(lrTuned.maxIter, Array(100, 200))
      .build()
    
    val lrCrossValidator = new CrossValidator()
      .setEstimator(lrTunedPipeline)
      .setEvaluator(binaryEvaluator)
      .setEstimatorParamMaps(lrParamGrid)
      .setNumFolds(3) // 3-fold CV for efficiency
      .setSeed(1234L)
    
    println("Running 3-fold cross-validation for Logistic Regression...")
    println(s"Testing ${lrParamGrid.length} parameter combinations...")
    
    val lrTunedModel = lrCrossValidator.fit(trainingData)
    val lrTunedPredictions = lrTunedModel.transform(testData)
    
    val tunedLrAuc = binaryEvaluator.evaluate(lrTunedPredictions)
    val tunedLrAccuracy = multiclassEvaluator.evaluate(lrTunedPredictions)
    
    // Extract best parameters
    val bestLrModel = lrTunedModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
    val bestLr = bestLrModel.stages.last.asInstanceOf[org.apache.spark.ml.classification.LogisticRegressionModel]
    
    println(f"Tuned AUC: $tunedLrAuc%.4f")
    println(f"Tuned Accuracy: $tunedLrAccuracy%.4f")
    println(f"Best regParam: ${bestLr.getRegParam}")
    println(f"Best elasticNetParam: ${bestLr.getElasticNetParam}")
    println(f"Best maxIter: ${bestLr.getMaxIter}")
    val lrAucImprovement = (tunedLrAuc - baselineLrAuc) * 100
    val lrAccuracyImprovement = (tunedLrAccuracy - baselineLrAccuracy) * 100
    println(f"Improvement: AUC ${if (lrAucImprovement >= 0) "+" else ""}${lrAucImprovement}%.2f%%, Accuracy ${if (lrAccuracyImprovement >= 0) "+" else ""}${lrAccuracyImprovement}%.2f%%")

    // 2. TUNED RANDOM FOREST
    println("\n2. TUNING RANDOM FOREST")
    println("-" * 40)
    
    val rfTuned = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setSeed(1234L)
    
    val rfTunedPipeline = new Pipeline().setStages(indexers ++ encoders ++ Array(labelIndexer, assembler, rfTuned))
    
    val rfParamGrid = new ParamGridBuilder()
      .addGrid(rfTuned.numTrees, Array(50, 100, 200))
      .addGrid(rfTuned.maxDepth, Array(5, 7, 10))
      .addGrid(rfTuned.minInstancesPerNode, Array(1, 5, 10))
      .build()
    
    val rfCrossValidator = new CrossValidator()
      .setEstimator(rfTunedPipeline)
      .setEvaluator(binaryEvaluator)
      .setEstimatorParamMaps(rfParamGrid)
      .setNumFolds(3)
      .setSeed(1234L)
    
    println("Running 3-fold cross-validation for Random Forest...")
    println(s"Testing ${rfParamGrid.length} parameter combinations...")
    
    val rfTunedModel = rfCrossValidator.fit(trainingData)
    val rfTunedPredictions = rfTunedModel.transform(testData)
    
    val tunedRfAuc = binaryEvaluator.evaluate(rfTunedPredictions)
    val tunedRfAccuracy = multiclassEvaluator.evaluate(rfTunedPredictions)
    
    // Extract best parameters
    val bestRfModel = rfTunedModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
    val bestRf = bestRfModel.stages.last.asInstanceOf[org.apache.spark.ml.classification.RandomForestClassificationModel]
    
    println(f"Tuned AUC: $tunedRfAuc%.4f")
    println(f"Tuned Accuracy: $tunedRfAccuracy%.4f")
    println(f"Best numTrees: ${bestRf.getNumTrees}")
    println(f"Best maxDepth: ${bestRf.getMaxDepth}")
    println(f"Best minInstancesPerNode: ${bestRf.getMinInstancesPerNode}")
    val rfAucImprovement = (tunedRfAuc - baselineRfAuc) * 100
    val rfAccuracyImprovement = (tunedRfAccuracy - baselineRfAccuracy) * 100
    println(f"Improvement: AUC ${if (rfAucImprovement >= 0) "+" else ""}${rfAucImprovement}%.2f%%, Accuracy ${if (rfAccuracyImprovement >= 0) "+" else ""}${rfAccuracyImprovement}%.2f%%")

    // 3. TUNED GRADIENT BOOSTED TREES
    println("\n3. TUNING GRADIENT BOOSTED TREES")
    println("-" * 40)
    
    val gbtTuned = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setSeed(1234L)
    
    val gbtTunedPipeline = new Pipeline().setStages(indexers ++ encoders ++ Array(labelIndexer, assembler, gbtTuned))
    
    val gbtParamGrid = new ParamGridBuilder()
      .addGrid(gbtTuned.maxIter, Array(10, 20, 30))
      .addGrid(gbtTuned.maxDepth, Array(4, 5, 6))
      .addGrid(gbtTuned.stepSize, Array(0.1, 0.2, 0.3))
      .build()
    
    val gbtCrossValidator = new CrossValidator()
      .setEstimator(gbtTunedPipeline)
      .setEvaluator(binaryEvaluator)
      .setEstimatorParamMaps(gbtParamGrid)
      .setNumFolds(3)
      .setSeed(1234L)
    
    println("Running 3-fold cross-validation for Gradient Boosted Trees...")
    println(s"Testing ${gbtParamGrid.length} parameter combinations...")
    
    val gbtTunedModel = gbtCrossValidator.fit(trainingData)
    val gbtTunedPredictions = gbtTunedModel.transform(testData)
    
    val tunedGbtAuc = binaryEvaluator.evaluate(gbtTunedPredictions)
    val tunedGbtAccuracy = multiclassEvaluator.evaluate(gbtTunedPredictions)
    
    // Extract best parameters
    val bestGbtModel = gbtTunedModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
    val bestGbt = bestGbtModel.stages.last.asInstanceOf[org.apache.spark.ml.classification.GBTClassificationModel]
    
    println(f"Tuned AUC: $tunedGbtAuc%.4f")
    println(f"Tuned Accuracy: $tunedGbtAccuracy%.4f")
    println(f"Best maxIter: ${bestGbt.getMaxIter}")
    println(f"Best maxDepth: ${bestGbt.getMaxDepth}")
    println(f"Best stepSize: ${bestGbt.getStepSize}")
    val gbtAucImprovement = (tunedGbtAuc - baselineGbtAuc) * 100
    val gbtAccuracyImprovement = (tunedGbtAccuracy - baselineGbtAccuracy) * 100
    println(f"Improvement: AUC ${if (gbtAucImprovement >= 0) "+" else ""}${gbtAucImprovement}%.2f%%, Accuracy ${if (gbtAccuracyImprovement >= 0) "+" else ""}${gbtAccuracyImprovement}%.2f%%")

    // ==================== FINAL COMPARISON ====================

    println("\n" + "="*80)
    println("FINAL COMPARISON: BASELINE vs TUNED MODELS")
    println("="*80)
    
    val results = Array(
      ("Logistic Regression", "Baseline", baselineLrAuc, baselineLrAccuracy),
      ("Logistic Regression", "Tuned", tunedLrAuc, tunedLrAccuracy),
      ("Random Forest", "Baseline", baselineRfAuc, baselineRfAccuracy),
      ("Random Forest", "Tuned", tunedRfAuc, tunedRfAccuracy),
      ("Gradient Boosted Trees", "Baseline", baselineGbtAuc, baselineGbtAccuracy),
      ("Gradient Boosted Trees", "Tuned", tunedGbtAuc, tunedGbtAccuracy)
    )
    
    println(f"${"Model"}%-22s | ${"Type"}%-8s | ${"AUC"}%-8s | ${"Accuracy"}%-8s")
    println("-" * 55)
    results.foreach { case (name, typ, auc, acc) =>
      println(f"$name%-22s | $typ%-8s | ${auc}%.4f   | ${acc}%.4f")
    }
    
    // Find best overall model
    val bestOverallByAuc = results.maxBy(_._3)
    val bestOverallByAccuracy = results.maxBy(_._4)
    
    println(f"\n🏆 BEST AUC: ${bestOverallByAuc._1} (${bestOverallByAuc._2}) - ${bestOverallByAuc._3}%.4f")
    println(f"🏆 BEST ACCURACY: ${bestOverallByAccuracy._1} (${bestOverallByAccuracy._2}) - ${bestOverallByAccuracy._4}%.4f")

    // Performance improvements summary
    println(f"\n📈 PERFORMANCE IMPROVEMENTS:")
    println(f"• Logistic Regression: AUC ${if (lrAucImprovement >= 0) "+" else ""}${lrAucImprovement}%.2f%%, Accuracy ${if (lrAccuracyImprovement >= 0) "+" else ""}${lrAccuracyImprovement}%.2f%%")
    println(f"• Random Forest: AUC ${if (rfAucImprovement >= 0) "+" else ""}${rfAucImprovement}%.2f%%, Accuracy ${if (rfAccuracyImprovement >= 0) "+" else ""}${rfAccuracyImprovement}%.2f%%")
    println(f"• Gradient Boosted Trees: AUC ${if (gbtAucImprovement >= 0) "+" else ""}${gbtAucImprovement}%.2f%%, Accuracy ${if (gbtAccuracyImprovement >= 0) "+" else ""}${gbtAccuracyImprovement}%.2f%%")

    // Show sample predictions from best model
    val bestPredictions = if (bestOverallByAuc._1.contains("Logistic") && bestOverallByAuc._2 == "Tuned") lrTunedPredictions
                         else if (bestOverallByAuc._1.contains("Random") && bestOverallByAuc._2 == "Tuned") rfTunedPredictions
                         else if (bestOverallByAuc._1.contains("Gradient") && bestOverallByAuc._2 == "Tuned") gbtTunedPredictions
                         else if (bestOverallByAuc._1.contains("Logistic")) lrPredictions
                         else if (bestOverallByAuc._1.contains("Random")) rfPredictions
                         else gbtPredictions
    
    println(f"\nSample predictions from best model (${bestOverallByAuc._1} - ${bestOverallByAuc._2}):")
    bestPredictions.select("label", "prediction", "probability").show(10, truncate = false)

    spark.stop()
  }
} 