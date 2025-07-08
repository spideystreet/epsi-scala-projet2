name := "spark-employee-attrition"
version := "0.1"
scalaVersion := "2.12.15"

val sparkVersion = "3.3.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion
)

// Add JVM options to resolve compatibility issues with Java 17+
javaOptions ++= Seq(
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
)

// Force sbt to fork a new JVM for running the application, which is necessary for javaOptions to be applied
fork in run := true 