package com.example

import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Employee Attrition Prediction")
      .master("local[*]") // Use all available cores on the local machine
      .getOrCreate()

    println("Spark session created successfully.")

    // We will add our data loading and processing logic here.

    spark.stop()
  }
} 