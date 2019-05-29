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
//package org.apache.spark.examples.ml

// $example on$
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql._
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor}
// $example off$
import org.apache.spark.sql.SparkSession

object DecisionTreeRegression {

  case class Life(features: org.apache.spark.ml.linalg.Vector, label: String)
  def main(args: Array[String]): Unit = {
    //设置主机运行模式为local，创建RDD
    val conf =new SparkConf().setAppName("DecisionTreeRegression").setMaster("local")
    val sc = new SparkContext(conf)

    //屏蔽不必要的日志信息
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val spark = SparkSession
      .builder
      .appName("DecisionTreeRegressionExample")
      .getOrCreate()
    // $example on$


    val sqLContext = new SQLContext(sc)//解决toDF问题
    import sqLContext.implicits._

    val data = spark.sparkContext.textFile("file:///usr/local/spark/data/iris.txt").
      map(_.split(",")).map(p => Life(Vectors.dense(p(0).toDouble,p(1).toDouble,p(2).toDouble ),p(4).toString()))
      .toDF()  //转化为DataFrame
    data.createOrReplaceTempView("iris")
    val df = spark.sql("select * from iris")

    //df.map(t => t(1)+":"+t(0)).collect().foreach(println)

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.//分别获取标签列和特征列，进行索引，并进行了重命名。
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Automatically identify categorical features, and index them.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(3) // features with > 3 distinct values are treated as continuous.
      .fit(data)

    // Convert indexed labels back to original labels.//这里我们设置一个labelConverter，目的是把预测的类别重新转化成字符型的。
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Split the data into training and test sets (30% held out for testing).//接下来，我们把数据集随机分成训练集和测试集，其中训练集占70%。
    val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))

    // Train a DecisionTree model.通过setter的方法来设置决策树的参数
    val dtRegressor = new DecisionTreeRegressor().
      setLabelCol("indexedLabel").
      setFeaturesCol("indexedFeatures")
      /*    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
       */

    // Chain indexers and tree in a Pipeline.//在pipeline中进行设置
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dtRegressor, labelConverter))

    // Train model. This also runs the indexers.//训练决策树模型
    val model = pipeline.fit(trainingData)

    // Make predictions.//进行预测
    val predictions = model.transform(testData)

    // Select example rows to display.//查看部分预测的结果
    predictions.select("predictedLabel", "label", "features").show(10)

    //计算统计分类耗时
    val startTime = System.currentTimeMillis()
    println("startTime:"+startTime)
    val endTime = System.currentTimeMillis()
    println("endtime:"+endTime)
    val timeConsuming = endTime - startTime
    println("timeConsuming:"+timeConsuming)

    // Select (prediction, true label) and compute test error.评估决策树分类模型
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)//精确度
    println("Test Error = " + (1.0 - accuracy))//误差

    // $example off$

    val evaluatorRegression = new RegressionEvaluator().
      setLabelCol("indexedLabel").
      setPredictionCol("prediction").
      setMetricName("rmse")
    val rmse = evaluatorRegression.evaluate(predictions)
    println("Root Mean Squard Error on test data ="  + rmse)
    val treeModel  = model.stages(2).asInstanceOf[DecisionTreeRegressionModel]
    println("Learned regression tree model:\n" + treeModel.toDebugString)
    spark.stop()

  }

}
// scalastyle:on println