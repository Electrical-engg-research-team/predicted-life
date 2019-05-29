import java.text.SimpleDateFormat

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.AFTSurvivalRegression
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

object AFTSurvivalRegression {

  case class aftRow(label:Double,censor:Double,features:org.apache.spark.ml.linalg.Vector)
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("AFT")
      .setMaster("local")
    val sc = new SparkContext(conf)
    val formatter = new SimpleDateFormat("SSSS")
    val startTime = sc.startTime
    val startTime1 = System.currentTimeMillis()
    println("startTime:"+startTime1)
    println(startTime)

    //屏蔽不必要的日志信息
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val spark = SparkSession.builder()
      .appName("AFTSurvivalRegression")
      .getOrCreate()

    val df = spark.createDataFrame(Seq(
      (1.218, 1.0, Vectors.dense(1.560, -0.605, 1)),
      (2.949, 0.0, Vectors.dense(0.346, 2.158, 10)),
      (3.627, 0.0, Vectors.dense(1.380, 0.231, 100 )),
      (0.273, 1.0, Vectors.dense(0.520, 1.151, 1000)),
      (4.199, 0.0, Vectors.dense(0.795, -0.226, 10000))
    )).toDF("label", "censor", "features")

    df.show(false)
    val quantileProbabilities = Array(0.3, 0.6,0.1)
    val aftModel = new AFTSurvivalRegression()
      .setQuantileProbabilities(quantileProbabilities)
      .setQuantilesCol("quantiles")
      .fit(df)

    aftModel.transform(df).show(3)

    // Print the coefficients, intercept and scale parameter for AFT survival regression
    println(s"Coefficients: ${aftModel.coefficients}")
    println(s"Intercept: ${aftModel.intercept}")
    println(s"Scale: ${aftModel.scale}")
    aftModel.transform(df).show(false)
    // $example off$
    // Summarize the model over the training set and print out some metrics

/*
   val Array(trainingDF, testDF) = df.randomSplit(Array(0.7, 0.3))

   val AFT = new AFTSurvivalRegression()
     .setFeaturesCol("features")
     .setLabelCol("label")
     .setCensorCol("censor")
     .fit(trainingDF)

   AFT.transform(testDF).show(10)
   val pipeline = new Pipeline().setStages(Array(AFT))

   vaawdata = sc.textFile("hdfs://localhost:9000/user/hadoop/input/AFTData.txt")
   val sqLContext = new SQLContext(sc)//解决toDF问题
   import sqLContext.implicits._  //通过RDD的隐式转换.toDF()方法完成RDD到DataFrame的转换


   val df = rawdata.map(_.split(","))
     .map(p => aftRow(p(0).toDouble,p(1).toDouble,Vectors.dense(p(4).toDouble,p(5).toDouble,p(6).toDouble,p(11).toDouble)))
     .toDF()
  df.show()

   val quantileProbabilitues = Array(0.3,0.6)


   val atf = new AFTSurvivalRegression()
       .setQuantileProbabilities(quantileProbabilitues)
       .setQuantilesCol("quantiles")
   val model = atf.fit(df)

   println(s"Cofficients: ${model.coefficients}\n Intercept:${model.intercept}\n Scale: ${model.scale}")
   model.transform(df).show(false)


*/
//计算统计分类耗时

    val endTime = System.currentTimeMillis()
    println("endtime:"+endTime)
    val timeConsuming = endTime - startTime1
    println("timeConsuming:"+formatter.format(timeConsuming)+"ms")
   spark.stop()
 }

}
