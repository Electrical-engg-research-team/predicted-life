/*import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.impurity.Impurity
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.Duration
import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time.DateTime

object DecisionTreeRegressionExample {

  def PrepareData(sc: SparkContext):(RDD[LabeledPoint],RDD[LabeledPoint], RDD[LabeledPoint]) = {
    //----------导入转换数据----------
    print("开始导入数据...")
    val rawDataWithHeader = sc.textFile("/home/hadoop/文档/hour.csv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex{
      (idx, iter) => if (idx == 0 ) iter.drop(1) else iter
    }
    println("共计："+rawData.count.toString()+"条")
    //---------创建训练评估所需数据---------
    println("准备训练数据...")
    val records = rawData.map(line => line.split(","))
    val data = records.map { fields =>
      val label = fields(fields.size - 1).toInt
      val featureSeason = fields.slice(2,3).map(d => d.toDouble)
      val features = fields.slice(4,fields.size - 3).map(d => d.toDouble)
      LabeledPoint(label, Vectors.dense(featureSeason ++ features))
    }
    //--------以随机方式将数据分为3组并返回-------------
    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8,0.1,0.1))
    println("将数据分为 trainData:"+ trainData.count() + "cvData:" + cvData.count() +"testData:"+ testData.count())
    return (trainData,cvData,testData)
  }

  def trainModel(trainData: RDD[LabeledPoint], impurity: String, maxDepth: Int, maxBins: Int): (DecisionTreeModel, Double) = {
    val startTime = new DateTime()
    val model = DecisionTree.trainRegressor(trainData, Map[Int, Int](), impurity, maxDepth, maxBins)
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    (model, duration.getMillis())
  }


  def evaluateModel(model:DecisionTreeModel , validationData: RDD[LabeledPoint]): (Double) = {
    val scoreAndLabels = validationData.map{data =>
      var predict = model.predict(data.features)
      (predict,data.label)
      }
    val Metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val AUC = Metrics.areaUnderROC()
    (AUC)
  }

  def trainEvaluate(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): DecisionTreeModel = {
    print("开始训练...")
    val (model, time) = trainModel(trainData, "entronpy",10,10)
    println("训练完成，所需时间："+time +"毫秒")
    val AUC = evaluateModel(model, validationData)
    println("评估结果AUC=" +AUC)
    return (model)
  }

  def predictData(sc: SparkContext, model: DecisionTreeModel): Unit = {
    //----------导入转换数据----------
    print("开始导入数据...")
    val rawDataWithHeader = sc.textFile("/home/hadoop/文档/DecisionTree.csv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex{
      (idx, iter) => if (idx == 0 ) iter.drop(1) else iter
    }
    println("共计："+rawData.count.toString()+"条")
    //---------创建训练评估所需数据---------
    println("准备训练数据...")
    val records = rawData.map(line => line.split(","))
    val data = records.map { fields =>
      val label = fields(fields.size - 1).toInt
      val featureSeason = fields.slice(2,3).map(d => d.toDouble)
      val features = fields.slice(4,fields.size - 3).map(d => d.toDouble)
      val featuresVectors = Vectors.dense(featureSeason ++ features)
      var dataDesc = ""
      dataDesc = dataDesc + { featuresVectors(0) match {case 1 => "春"; case 2 => "夏"; case 3 => "秋"; case 4 => "冬"; }} + "天，"
      dataDesc = dataDesc + featuresVectors(1).toInt + "月，"
      dataDesc = dataDesc + featuresVectors(2).toInt + "时，"
      dataDesc = dataDesc + {featuresVectors(3) match { case 0 => "工作周"; case 1 => "假日";}} + "，"
      dataDesc = dataDesc + "星期" + {featuresVectors(4) match {case 0 => "日"; case 1 => "一"; case 2 => "二";
      case 3 => "三"; case 4 => "四"; case 5 => "五"; case 6 => "六"; } } + "，"
      dataDesc = dataDesc + {featuresVectors(5) match { case 0 => "非工作日"; case 1 => "工作日";}} + "，"
      dataDesc = dataDesc + {featuresVectors(6) match {case 1 => "晴";case 2 => "阴";case 3 => "小雨";case 4 => "大雨";}} + "，"
      dataDesc = dataDesc + (featuresVectors(7) * 41).toInt + "度，"
      dataDesc = dataDesc + "体感" + (featuresVectors(8) * 50).toInt + "度，"
      dataDesc = dataDesc + "湿度" + (featuresVectors(9) * 100).toInt + "，"
      dataDesc = dataDesc + "风速" + (featuresVectors(10) * 67).toInt + "，"

      val predict = model.predict(featuresVectors)
      val result = (if ( label == predict) "正确" else  "错误")
      val error = (math.abs(label - predict).toString())
      println("特征：" + dataDesc + "==> 预测结果：" + predict.toInt + "   实际：" + label.toInt + "   误差：" + error)

    }
    }
  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }

  def main(args: Array[String]): Unit = {
    SetLogger()
    val conf = new SparkConf().setAppName("DecisionTreeExample").setMaster("local")
    val sc = SparkContext(conf)
    println("RunDessionTreeExample")
    println("===========数据准备阶段============")
    val(trainData, validationData, testData, categoriesMap) = PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    println("===========数据评估阶段============")
    val model = trainEvaluate(trainData, validationData)
    println("===========测试阶段===========")
    val auc = evaluateModel(model, testData)
    println("使用testata测试最佳模型，结果AUC:"+auc)
    println("===========预测数据===========")
    predictData(sc, model, categoriesMap)
    trainData.unpersist();validationData.unpersist();testData.unpersist()
  }

}

*//*
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
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
// $example off$
import org.apache.spark.sql.SparkSession

object DecisionTreeRegressionExample {
  def main(args: Array[String]): Unit = {
    val conf =new SparkConf().setAppName("DecisionTree").setMaster("local")
    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder
      .appName("DecisionTreeRegressionExample")
      .getOrCreate()

    // $example on$
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm").load("/home/hadoop/文档/file2.txt")

    // Automatically identify categorical features, and index them.
    // Here, we treat features with > 4 distinct values as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, dt))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println("Learned regression tree model:\n" + treeModel.toDebugString)
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
