import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors

object LRCode {

  def main(args:Array[String]): Unit = {
    //create RDD
    val conf = new SparkConf()
      .setAppName("Logisitic Test")
      .setMaster("local")
    val sc = new SparkContext(conf)

    //屏蔽不必要的日志信息
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    //使用MLUtils对象将hdfs中的数据读取到RDD中
    val path = "hdfs://localhost:9000/user/hadoop/input/decisiontree.txt"
    val rawData = sc.textFile(path)
    val startTime = System.currentTimeMillis()  //time
    println("startTime:"+startTime)

    //通过“\t”（制表符）即按行对数据内容进行分割
    val records = rawData.map(_.split("\t"))

    /**
      * 取最后一列列为类标，其余列作为特征值
      */
    val data = records.map{ point =>
      //去除集合中多余的空格,前面是旧的，后面是新的
      val firstdata = point.map(_.replaceAll(" ",""))
      //用空格代替集合中的逗号
      val replaceData=firstdata.map(_.replaceAll(","," "))
      //临时数据作为
      val temp = replaceData(0).split(" ")
      //定义类标
      val label=temp(0).toInt
      //定义特征值
      val features = temp.slice(1,temp.size-1)
        .map(_.hashCode)
        .map(x => x.toDouble)
      //设置为（类标，特征值（向量组））
      LabeledPoint(label,Vectors.dense(features))

    }

    //按照8:2的比例将数据随机分为训练集和测试集
    val splits = data.randomSplit(Array(0.8,0.2),seed = 11L)
    val traning = splits(0).cache()
    val test = splits(1)

    //训练二元分类的logistic回归模型
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(traning)

    //预测测试样本的类别
    val predictionAndLabels = test.map{
      case LabeledPoint(label,features) =>
        val prediction = model.predict(features)
        (prediction,label)
    }

    //输出模型在样本上的准确率
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val auRoc = metrics.areaUnderROC()
    //打印准确率
    println("Area under Roc =" + auRoc)
    //计算统计分类耗时
    val endTime = System.currentTimeMillis()
    println("endtime:"+endTime)
    val timeConsuming = endTime - startTime
    println("timeConsuming:"+timeConsuming)

    //输出逻辑回归权重最大的前5个特征
    val weights = (1 to model.numFeatures) zip model.weights.toArray
    println("Top 5 features:")
    weights.sortBy(-_._2).take(5).foreach{case(k,w) =>
      println("Feature " + k + " = " + w)
    }

    //保存训练好模型
    val modelPath = "hdfs://localhost:9000/user/hadoop/output"
    model.save(sc, modelPath)
    val sameModel = LogisticRegressionModel.load(sc,modelPath)

    //关闭程序
    sc.stop()
  }
}
