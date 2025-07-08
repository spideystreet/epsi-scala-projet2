package com.example

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Employee Attrition Prediction")
      .master("local[*]") // Use all available cores on the local machine
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

    // Show a sample of the data to verify it's loaded correctly
    println("Successfully loaded data. Schema after dropping irrelevant columns:")
    cleanedDf.printSchema()
    println("Data sample:")
    cleanedDf.show(5, truncate = false)

    spark.stop()
  }
} 