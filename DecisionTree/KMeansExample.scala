


import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}


object KMeansExample {

  case class Iris(features: org.apache.spark.ml.linalg.Vector,label:String)
  case class model_instance (features:  org.apache.spark.ml.linalg.Vector) //定义数据类型

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("KMeans").setMaster("local")
    val sc = new SparkContext(conf)

    val data = sc.textFile("file:///usr/local/spark/data/decisiontree.txt")//导入数据
    val sqLContext = new SQLContext(sc)//解决toDF问题
    import sqLContext.implicits._

    //对数据进行预处理，将数据读入RDD[model_instance]的结构中，并通过RDD的隐式转换.toDF()方法完成RDD到DataFrame的转换
    //使用了filter算子，过滤掉类标签
    //正则表达式\\d*(\\.?)\\d*可以用于匹配实数类型的数字，\\d*使用了*限定符，
    // 表示匹配0次或多次的数字字符，\\.?使用了?限定符，表示匹配0次或1次的小数点。
    val df = data.map(line =>
    {model_instance(Vectors.dense(line.split(",").
      filter(p => p.matches("\\d*(\\.?)\\d*")).
      map(_.toDouble)))}).
      toDF()

    //在得到数据后，我们即可通过ML包的固有流程：创建Estimator并调用其fit()方法来生成相应的Transformer对象
    val kmeansmodel = new KMeans().
      setK(3).
      setFeaturesCol("features").
      setPredictionCol("prediction").
      fit(df)

    //提供了一致性的transform()方法，用于将存储在DataFrame中的给定数据集进行整体处理，生成带有预测簇标签的数据集
    val results = kmeansmodel.transform(df)
    //使用collect()方法，该方法将DataFrame中所有的数据组织成一个Array对象进行返回
    results.collect().foreach(row => println(row(0) + "is predicted as cluster " + row(1)))
    //也可以通过KMeansModel类自带的clusterCenters属性获取到模型的所有聚类中心情况
    kmeansmodel.clusterCenters.foreach(center => println("Clustering Center: " + center))

    //集合内误差平方和，方法来度量聚类的有效性，在真实K值未知的情况下，该值的变化可以作为选取合适K值的一个重要参考
    kmeansmodel.computeCost(df)

  }

}
