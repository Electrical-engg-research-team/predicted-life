import java.text.SimpleDateFormat

import DecisionTreeClassificationTest.Iris
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object GeneralizedLinear {
  /*
   * Licensed to the Apache Software Foundation (ASF) under one or more
   * contributor license agreements.  See the NOTICE file distributed with
   * this work for additional information regarding copyright ownership.
   * The ASF licenses this file to You under the Apache License, Version 2.0
   * (the "License"); you may not use this file except in compliance with
   * the License.  You may obtain a copy of the License at
   *
   *    http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   */

  // scalastyle:off println


  // $example on$
  import org.apache.spark.ml.regression.GeneralizedLinearRegression
  // $example off$
  import org.apache.spark.sql.SparkSession

  /**
    * An example demonstrating generalized linear regression.
    * Run with
    * {{{
    * bin/run-example ml.GeneralizedLinearRegressionExample
    * }}}
    */

  case class GLM(features: org.apache.spark.ml.linalg.Vector, label: Float)
    def main(args: Array[String]): Unit = {
      //设置主机运行模式为local，创建RDD
      val conf =new SparkConf().setAppName("LinearRegression").setMaster("local")
      val sc = new SparkContext(conf)
      //屏蔽不必要的日志信息
      Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
      Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

      val spark = SparkSession
        .builder
        .appName("GeneralizedLinearRegressionExample")
        .getOrCreate()

      // $example on$
      // Load training data
      val data = spark.read.format("libsvm")
       .load("file:///usr/local/spark/data/mllib/sample_linear_regression_data.txt")

      val sqLContext = new SQLContext(sc) //解决toDF问题
      import sqLContext.implicits._

      //val data = spark.sparkContext.textFile("hdfs://localhost:9000/user/hadoop/input/openingdata_final.txt").
       // map(_.split(",")).map(p => GLM(Vectors.dense(p(4).toDouble,p(5).toDouble,p(6).toDouble,p(7).toDouble,p(8).toDouble,p(9).toDouble),
       // p(10).toFloat)).toDF()

       //val data_df = spark.createDataFrame(data)
      data.createOrReplaceTempView("GLM")
      val df = spark.sql("select * from GLM")

      df.map(t => t(1)+":"+t(0)).collect().foreach(println)
      // Split the data into training and test sets (30% held out for testing).
      val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

      //time consume
      val startTime = System.currentTimeMillis()
      //glr_model
      val glr = new GeneralizedLinearRegression()
        .setFamily("gaussian")
        .setLink("identity")
        .setMaxIter(10)
        .setRegParam(0.3)

      // Fit the model
      val model = glr.fit(trainingData)
      // Make predictions.让测试数据按顺序通过拟合的工作流，生成我们所需要的预测结果
      val predictions = model.transform(testData)
      // Select example rows to display.显示所预测的值，以及目标，以及特征
      predictions.select("prediction", "label", "features").show(5)


      // Print the coefficients and intercept for generalized linear regression model
      println(s"Coefficients: ${model.coefficients}")
      println(s"Intercept: ${model.intercept}")
      println(s"Para:${model.extractParamMap()}")

      // Summarize the model over the training set and print out some metrics
      val summary = model.summary
      println(s"Coefficient Standard Errors: ${summary.coefficientStandardErrors.mkString(",")}")
      println(s"T Values: ${summary.tValues.mkString(",")}")
      println(s"P Values: ${summary.pValues.mkString(",")}")
      println(s"Dispersion: ${summary.dispersion}")
      println(s"Null Deviance: ${summary.nullDeviance}")
      println(s"Residual Degree Of Freedom Null: ${summary.residualDegreeOfFreedomNull}")
      println(s"Deviance: ${summary.deviance}")
      println(s"Residual Degree Of Freedom: ${summary.residualDegreeOfFreedom}")
      println(s"AIC: ${summary.aic}")
      println("Deviance Residuals: ")
      summary.residuals().show(5)

      //evaluator
      // Select (prediction, true label) and compute test error.显示错误，如果有的话
      val evaluator = new RegressionEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("rmse")
      val rmse = evaluator.evaluate(predictions)
      println("Root Mean Squared Error (RMSE) on test data = " + rmse)
      val r2 = new RegressionEvaluator().setMetricName("r2").evaluate(predictions)
      println("R^2 = "+r2)
      // $example off$
      //计算统计分类耗时

      val endTime = System.currentTimeMillis()
      println("endtime:"+endTime)
      val timeConsuming = endTime - startTime
      val formatter = new SimpleDateFormat("SSSS")
      println("timeConsuming:"+formatter.format(timeConsuming)+"ms")
      spark.stop()
    }

  // scalastyle:on println
}
