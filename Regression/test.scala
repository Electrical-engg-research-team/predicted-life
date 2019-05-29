

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.AFTSurvivalRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

object test {

  case class aftRow(label:Double,censor:Double,features:org.apache.spark.ml.linalg.Vector)
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("AFT")
      .setMaster("local")
    val sc = new SparkContext(conf)

    //屏蔽不必要的日志信息
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)


    val spark = SparkSession.builder()
      .appName("Spark Survival regression")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    val rawdata = sc.textFile("hdfs://localhost:9000/user/hadoop/input/AFTData.txt")
    val sqLContext = new SQLContext(sc)//解决toDF问题
    import sqLContext.implicits._  //通过RDD的隐式转换.toDF()方法完成RDD到DataFrame的转换

    val df = rawdata.map(_.split(","))
      .map(p => aftRow(p(0).toDouble,p(1).toDouble,Vectors.dense(p(2).toDouble,p(3).toDouble,p(4).toDouble,p(5).toDouble
        ,p(6).toDouble,p(7).toDouble,p(8).toDouble,p(9).toDouble)))
      .toDF()
    df.show(false)

    //val quantileProbabilities = Array(0.3, 0.6,0.1,0.7)
    val Array(trainingDF, testDF) = df.randomSplit(Array(0.7, 0.3))

    val aftModel = new AFTSurvivalRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setCensorCol("censor")
      .fit(trainingDF)

    val pipeline = new Pipeline().setStages(Array(aftModel))
    val paramGrid = new ParamGridBuilder()
      .addGrid(aftModel.maxIter, Array(100, 500, 1000))
      .addGrid(aftModel.tol, Array(1E-2, 1E-6)).build()

    val RegEvaluator = new RegressionEvaluator()
      .setLabelCol(aftModel.getLabelCol)
      .setPredictionCol(aftModel.getPredictionCol)
      .setMetricName("rmse")

    //jiaochayanzheng
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(RegEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
    val cvModel = cv.fit(trainingDF)
    cvModel.extractParamMap()
    cvModel.avgMetrics.length
    cvModel.avgMetrics // 参数对应的平均度量
    cvModel.getEvaluator.extractParamMap()
    cvModel.getEstimatorParamMaps.length
    cvModel.getEstimatorParamMaps.foreach(println) // 参数组合的集合
    cvModel.getEvaluator.isLargerBetter // 评估的度量值是大的好，还是小的好
    cvModel.getNumFolds // 交叉验证的折数
    cvModel.extractParamMap()
    cvModel.save("hdfs://localhost:9000/user/hadoop/output/cv")
    // 测试模型
    val predictDF: DataFrame = cvModel.transform(testDF).selectExpr(
      //"race","poverty","smoke","alcohol","agemth","ybirth","yschool","pc3mth", "features",
      "label", "censor",
      "round(prediction,2) as prediction").orderBy("label")
    predictDF.show(12)
    //val aftModel = new AFTSurvivalRegression()
    //  .setQuantileProbabilities()
    //  .setQuantilesCol("quantiles")
    //  .fit(df)

    println(s"Coefficients: ${aftModel.coefficients}")
    println(s"Intercept: ${aftModel.intercept}")
    println(s"Scale: ${aftModel.scale}")
    aftModel.transform(df).show(false)

    /*

    AFT.transform(testDF).show(10)

    // cvModel.avgMetrics.length=cvModel.getEstimatorParamMaps.length
    // cvModel.avgMetrics与cvModel.getEstimatorParamMaps中的元素一一对应





      // 评估的参数



    //################################

*/
//计算统计分类耗时
    val startTime = System.currentTimeMillis()
    println("startTime:"+startTime)
    val endTime = System.currentTimeMillis()
    println("endtime:"+endTime)
    val timeConsuming = endTime - startTime
    println("timeConsuming:"+timeConsuming)

    spark.stop()

  }


//



}