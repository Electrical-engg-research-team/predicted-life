# from process import main
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.ml.feature import Normalizer, StandardScaler
from pyspark.ml.feature import StringIndexer, VectorIndexer, IndexToString
import numpy as np
# loads data
hdfs_path_final = "/user/hadoop/input/openingdata_final_1.txt"
hdfs_path_simple = "/user/hadoop/input/openingdata_simple.txt"
hdfs_path_aft = "/user/hadoop/input/AFTData.txt"

# if __name__ == '__main__':
def dispose_complext(rdd):
    # data format is (vector("features"), string("label"), float("censor"))
    training = rdd.map(lambda l: l.split(","))\
        .map(lambda p: Row(features=Vectors.dense(float(p[1]), float(p[2]),
                                                  float(p[3]), float(p[4]),
                                                  float(p[5]), float(p[6]),
                                                  float(p[7]), float(p[8])),
                           label=float(p[0]),
                           censor=float(p[-1])))
    # to DF: DataFrame
    # training_df = main.spark.createDataFrame(training, ["features", "label"])
    # test data
    # training_df.first()
    return training

def dispose_simple(rdd):
    # data format is (vector("features"), string("label"))
    training = rdd.map(lambda l: l.split(","))\
        .map(lambda p: Row(features=Vectors.dense(float(p[0]), float(p[1]),
                                                  float(p[2]), float(p[3])),
                           label=float(p[4])))
    return training
def dispose_final(rdd):
    training = rdd.map(lambda l: l.split(","))\
        .map(lambda p: Row(features=Vectors.dense(float(p[1]), float(p[2]),
                                                  float(p[3]), float(p[4]),
                                                  float(p[5]), float(p[6]),
                                                  float(p[7]), float(p[8])),
                           label=float(p[-2])))
    return training
def dispose_aft(rdd):
    # data format is (vector("features"), string("label"), float("censor"))
    training = rdd.map(lambda l: l.split(","))\
        .map(lambda p: Row(features=Vectors.dense(float(p[0]), float(p[1]),
                                                  float(p[2]), float(p[3])),
                           label=float(p[4]),
                           censor=float(p[-1])))
# Log label vector
def log_data(rdd_data):
    # log label
    data_log = rdd_data.map(lambda lp: Row(label=float(np.log(lp.label)), features=lp.features))
    df_log = main.spark.createDataFrame(data_log, ["features", "label"])
    return df_log

# Split the data into training and test sets (30% held out for testing)
def trainiingData(data):
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    return trainingData

def testData(data):
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    return testData

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
def label_indexer(data_df):
    return StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data_df)

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
def feature_indexer(data_df):
    return VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data_df)

# Normalize each Vector using $L^1$ norm.
def l1_norm(data_df):
    return Normalizer(inputCol="features", outputCol="normFeatures", p=1.0).transform(data_df)

# save model on HDFS
# save predictions in local file
result_path = "/usr/local/spark/data/predicted"
hdfs_output = "hdfs://localhost:9000/user/hadoop/output"
def save_aft(model, predictions):
    result_data = predictions.toPandas()
    write = model.write().overwrite()
    result_data.to_csv(result_path + "/aft_model_result.txt")
    write.save(hdfs_output + "/aft")

def save_dt(model, predictions):
    result_data = predictions.toPandas()
    write = model.write().overwrite()
    result_data.to_csv(result_path + "/dt_model_result.txt")
    write.save(hdfs_output + "/dt")

def save_gl(model, predictions):
    result_data = predictions.toPandas()
    write = model.write().overwrite()
    result_data.to_csv(result_path + "/gl_model_result.txt")
    write.save(hdfs_output + "/gl")
    print("The results have been save in " + result_path)
    print("The model upload to HDFS: " + hdfs_output)