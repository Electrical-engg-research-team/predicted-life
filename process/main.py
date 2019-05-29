# coding=utf-8
# main
from pyspark.sql import SparkSession
from process import data, arthmetic, summary
from pyspark import SparkContext, SparkConf

conf = SparkConf() \
    .setMaster('local') \
    .setAppName("analyze_process")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

if __name__ == '__main__':
    print("=========1.Start Spark========")
    spark = SparkSession \
        .builder \
        .appName("DTCModel") \
        .getOrCreate()
    print("Spark Job successful start!\n")
    print("========2.Prepare Data========")
    print("Loads data to RDDs.\n...")
    rdd = sc.textFile(data.hdfs_path_final)
    rdd_aft = sc.textFile(data.hdfs_path_aft)
    temp_data = data.dispose_final(rdd)
    temp_data_aft = data.dispose_complext(rdd_aft)
    print("RDDs to DataFrame.\n...")
    df_data = spark.createDataFrame(temp_data, ["features", "label"])
    df_data_aft = spark.createDataFrame(temp_data_aft, ["censor", "features", "label"])
    print("========3.Training Model=========")
    # print("Based on  %s to create new model." % df_data)
    arth = ['AFTSurvivalRegression', 'DecisionTreeRegression', 'GeneralizedLinearRegression']
    # print("Arthmetic include: ", arth[1], arth[2])
    aft_model = arthmetic.AFT(df_data_aft)
    dt_model = arthmetic.DTR(df_data)
    gl_model = arthmetic.GL(df_data)

    print("========4.Test Data=========")
    # Make predictions.
    # Select example rows to display.
    print("\nThe predictions of %s:" % arth[0])
    aft_pred = summary.predicted_aft(aft_model, df_data_aft)
    print("\nThe predictions of %s:" % arth[1])
    dt_pred = summary.predicted_dt(dt_model, df_data)
    print("\nthe predictions of %s:" % arth[2])
    gl_pred = summary.predicted_gl(gl_model, df_data)

    print("========5.Evaluate Model=========")
    print("\nThe summary of %s:" % str(arth[0]))
    aft_rmse = summary.evaluator(aft_pred)
    print("\nThe summary of %s:" % arth[2])
    dt_rmse = summary.evaluator(dt_pred)
    print("T\nhe summary of %s:" % arth[1])
    gl_rmse = summary.evaluator(gl_pred)
    print("\n")
    min_prams = min(dt_rmse, gl_rmse, aft_rmse)
    if min_prams == aft_rmse:
        print("The best model is:"+arth[0])
    if min_prams == dt_rmse:
        print("The best model is:"+arth[2])
    elif min_prams == gl_rmse:
        print("The best model is:"+arth[1])

    print("========6.Save Model=========")
    # SaveModel(sc)
    data.save_aft(aft_model, aft_pred)
    data.save_dt(dt_model, dt_pred)
    data.save_gl(gl_model, gl_pred)

    print("========7.Stop Spark========")
    spark.stop()
    print("========8.Draw Result========")
