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
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds
import org.apache.spark.sql._
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
// $example off$
import org.apache.spark.sql.SparkSession

object DecisionTreeClassificationTest {

  case class Iris(features: org.apache.spark.ml.linalg.Vector, label: String)
  def main(args: Array[String]): Unit = {
    //设置主机运行模式为local
    val conf =new SparkConf().setAppName("DecisionTree").setMaster("local")
    val sc = new SparkContext(conf)
    val ssc = new StreamingContext(sc, Seconds(3))
    val spark = SparkSession
      .builder
      .appName("DecisionTreeClassificationTest")
      .getOrCreate()
    // $example on$
    // Load the data stored in LIBSVM format as a DataFrame.
    //val data = spark.read.format("libsvm").load("file:///usr/local/spark/data/mllib/sample_libsvm_data.txt")

    val sqLContext = new SQLContext(sc)//解决toDF问题
    import sqLContext.implicits._

    val data = spark.sparkContext.textFile("file:///usr/local/spark/data/decisiontree.txt").
      map(_.split(",")).map(p => Iris(Vectors.dense(p(0).toDouble,p(1).toDouble),
      p(2).toString())).toDF()
    data.createOrReplaceTempView("decisiontree")
    val df = spark.sql("select * from decisiontree")

    df.map(t => t(1)+":"+t(0)).collect().foreach(println)

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
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a DecisionTree model.通过setter的方法来设置决策树的参数
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")


    // Chain indexers and tree in a Pipeline.//在pipeline中进行设置
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    // Train model. This also runs the indexers.//训练决策树模型
    val model = pipeline.fit(trainingData)

    // Make predictions.//进行预测
    val predictions = model.transform(testData)

    // Select example rows to display.//查看部分预测的结果
    predictions.select("predictedLabel", "label", "features").show(10)

    // Select (prediction, true label) and compute test error.评估决策树分类模型
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)//精确度
    println("Test Error = " + (1.0 - accuracy))//误差

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)
    // $example off$

    spark.stop()

  }

}
// scalastyle:on println