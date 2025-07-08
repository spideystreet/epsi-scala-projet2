- We have loaded our CSV into DF on the Main.scala with schema
    - nullable = false : it's for Spark, it would have an error if there were any nulls (but we don't have them normally, Kaggle datasets are treated)
- We dopped irrevelant columns : 
    - EmployeeCount: It's the same value for every employee.
    - StandardHours: Also a constant value.
    - Over18: All employees are over 18.
    - EmployeeNumber: This is just a unique identifier and has no predictive power.

# Launch code
- sbt run : fetch dependancies & compile Scala code