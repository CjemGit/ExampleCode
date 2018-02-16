//Script using Apache Spark APIs to perform regression analysis
//Data and task taken from https://www.kaggle.com/c/house-prices-advanced-regression-techniques

// DATA IMPORT

import org.apache.spark.sql.SQLContext

val test = sqlContext.table("test")
val trainwlabel = sqlContext.table("train")

val labels = trainwlabel.select("Id", "SalePrice")

val label2 = labels.withColumn("SaleTARGET", labels("SalePrice").cast("Int")).select("Id","SaleTARGET")

val train = trainwlabel.drop(trainwlabel.col("SalePrice"))

val totalDF = train.union(test)

println("There are " + test.count() + " observations in the test set")
println("There are " + train.count() + " observations in the train set")
println("There are " + totalDF.count() + " observations in the total set")

// DATA EXPLORATION AND CLEANING

val totalDF2 = totalDF

  .withColumn("MSSub", totalDF("MSSUBClass").cast("Int"))
  .withColumn("LotFront", totalDF("LotFrontage").cast("Int"))
  .withColumn("LotAre", totalDF("LotArea").cast("Int"))
  .withColumn("OverallQt", totalDF("OverallQual").cast("Int"))
  .withColumn("OverallCon", totalDF("OverallCond").cast("Int"))
  .withColumn("YearB", totalDF("YearBuilt").cast("Int"))
  .withColumn("YearR", totalDF("YearRemodAdd").cast("Int"))
  .withColumn("MasVnrA", totalDF("MasVnrArea").cast("Int"))
  .withColumn("BsmtFinSF", totalDF("BsmtFinSF1").cast("Int"))
  .withColumn("1stFlr", totalDF("1StFlrSF").cast("Int"))
  .withColumn("2stFlr", totalDF("2ndFlrSF").cast("Int"))
  .withColumn("GrLivA", totalDF("GrLivArea").cast("Int"))
  .withColumn("GarageYrBl", totalDF("GarageYrBlt").cast("Int"))
  .withColumn("GarageA", totalDF("GarageArea").cast("Int"))
  .withColumn("WoodDeckS", totalDF("WoodDeckSF").cast("Int"))
  .withColumn("OpenPorchS", totalDF("OpenPorchSF").cast("Int"))
  .withColumn("EndclosedPorc", totalDF("EnclosedPorch").cast("Int"))
  .withColumn("3SnPorc", totalDF("3SsnPorch").cast("Int"))
  .withColumn("ScreenPorc", totalDF("ScreenPorch").cast("Int"))
  .withColumn("PoolAre", totalDF("PoolArea").cast("Int"))
  .withColumn("MiscVa", totalDF("MiscVal").cast("Int"))
  .withColumn("MoSol", totalDF("MoSold").cast("Int"))
  .withColumn("YrSol", totalDF("YrSold").cast("Int"))

.drop("MSSUBClass", "LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "1StFlrSF","2ndFlrSF","GrLivArea", "GarageYrBlt","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal","MoSold","YrSold")



println("Missing Values")

println("MSSub " + totalDF2.where($"MSSub".isNull).count())
println("LotFront " + totalDF2.where($"LotFront".isNull).count())
println("LotAre " + totalDF2.where($"LotAre".isNull).count())
println("OverallQt " + totalDF2.where($"OverallQt".isNull).count())
println("OverallCon " + totalDF2.where($"OverallCon".isNull).count())
println("YearB " + totalDF2.where($"YearB".isNull).count())
println("YearR " + totalDF2.where($"YearR".isNull).count())
println("MasVnrA " + totalDF2.where($"MasVnrA".isNull).count())
println("BsmtFinSF " + totalDF2.where($"BsmtFinSF".isNull).count())
println("1stFlr " + totalDF2.where($"1stFlr".isNull).count())
println("2stFlr " + totalDF2.where($"2stFlr".isNull).count())
println("GrLivA " + totalDF2.where($"GrLivA".isNull).count())
println("GarageYrBl " + totalDF2.where($"GarageYrBl".isNull).count())
println("GarageA " + totalDF2.where($"GarageA".isNull).count())
println("WoodDeckS " + totalDF2.where($"WoodDeckS".isNull).count())
println("OpenPorchS " + totalDF2.where($"OpenPorchS".isNull).count())
println("EndclosedPorc " + totalDF2.where($"EndclosedPorc".isNull).count())
println("3SnPorc " + totalDF2.where($"3SnPorc".isNull).count())
println("ScreenPorc " + totalDF2.where($"ScreenPorc".isNull).count())
println("PoolAre " + totalDF2.where($"PoolAre".isNull).count())
println("MiscVa " + totalDF2.where($"MiscVa".isNull).count())
println("MoSol " + totalDF2.where($"MoSol".isNull).count())
println("YrSol " + totalDF2.where($"YrSol".isNull).count())

println("0 Values")

println("MSSub " + totalDF2.where($"MSSub" === 0).count())
println("LotFront " + totalDF2.where($"LotFront" === 0).count())
println("LotAre " + totalDF2.where($"LotAre"=== 0).count())
println("OverallQt " + totalDF2.where($"OverallQt"=== 0).count())
println("OverallCon " + totalDF2.where($"OverallCon"=== 0).count())
println("YearB " + totalDF2.where($"YearB"=== 0).count())
println("YearR " + totalDF2.where($"YearR"=== 0).count())
println("MasVnrA " + totalDF2.where($"MasVnrA"=== 0).count())
println("BsmtFinSF " + totalDF2.where($"BsmtFinSF"=== 0).count())
println("1stFlr " + totalDF2.where($"1stFlr"=== 0).count())
println("2stFlr " + totalDF2.where($"2stFlr"=== 0).count())
println("GrLivA " + totalDF2.where($"GrLivA"=== 0).count())
println("GarageYrBl " + totalDF2.where($"GarageYrBl"=== 0).count())
println("GarageA " + totalDF2.where($"GarageA"=== 0).count())
println("WoodDeckS " + totalDF2.where($"WoodDeckS"=== 0).count())
println("OpenPorchS " + totalDF2.where($"OpenPorchS"=== 0).count())
println("EndclosedPorc " + totalDF2.where($"EndclosedPorc"=== 0).count())
println("3SnPorc " + totalDF2.where($"3SnPorc"=== 0).count())
println("ScreenPorc " + totalDF2.where($"ScreenPorc"=== 0).count())
println("PoolAre " + totalDF2.where($"PoolAre"=== 0).count())
println("MiscVa " + totalDF2.where($"MiscVa"=== 0).count())
println("MoSol " + totalDF2.where($"MoSol"=== 0).count())
println("YrSol " + totalDF2.where($"YrSol"=== 0).count())


val Mosol = traincontinuous.groupBy("MoSol").count()

display(Mosol.sort((Mosol("MoSol").asc)))

val traincategory = trainwlabels2.drop("MSSub", "LotAre","OverallQt", "OverallCon", "YearB", "YearR",  "1stFlr", "GrLivA", "GarageA", "MoSol", "YrSol", "saleTARGET")

val traincontinuous = totalDF3.select("MSSub", "LotAre","OverallQt", "OverallCon", "YearB", "YearR",  "1stFlr", "GrLivA", "GarageA", "MoSol", "YrSol")

traincontinuous.show()

val traincontinuous2 = traincontinuous.filter($"LotAre" < 100000 && $"YearB" > 1900 && $"YearB" > 1951)


// FEATURE ENGINEERING

import org.apache.spark.sql.functions.udf

def summer (x: Int): Int = if (x>=4 && x<=8) 1 else 0

val summerUDF = udf(summer _)

def y2k (x: Int): Int = if (x>=1987) 1 else 0

val y2kUDF = udf(y2k _)

def y2k2 (x: Int): Int = if (x>=2002) 1 else 0

val y2k2UDF = udf(y2k2 _)

val totalcontinuous = traincontinuous2

       .withColumn("built1987", y2kUDF(traincontinuous2("YearB")))
       .withColumn("mod2002", y2k2UDF(traincontinuous2("YearR")))
       .withColumn("soldsummer", summerUDF(traincontinuous2("MoSol")))

  .drop("MoSol", "YrSol", "YearB", "YearR")

totalcontinuous.show()

val traincontinuous3 = totalcontinuous.limit(train.count().toInt)
val trainwlabels2 = train2.join(label2, "Id")

// VECTORISATION FOR MLIB


import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val assembler = new VectorAssembler()
  .setInputCols(Array("MSSub", "LotAre","OverallQt", "OverallCon", "1stFlr", "GrLivA", "GarageA", "built1987", "mod2002", "soldsummer"))
  .setOutputCol("features")

val output = assembler.transform(traincontinuous3)

println("Assembled columns  MSSub, LotAre,OverallQt, OverallCon, YearB, YearR,  1stFlr, GrLivA, GarageA, MoSol, YrSol to vector column 'features' ")

val trainwlabels3 = output.withColumnRenamed("SaleTARGET", "label").select("features", "label")

// REGRESSION MODELLING

import org.apache.spark.sql.functions._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.ml.regression.LinearRegression

val lr =  new LinearRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

// Fit the model
val lrModel = lr.fit(trainwlabels3)

// Print the coefficients and intercept for linear regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// Summarize the model over the training set and print out some metrics
val trainingSummary = lrModel.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
//trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")
trainwlabels3.select(avg($"label")).show()
