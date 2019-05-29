from pyspark.ml import Pipeline
from pyspark.ml.regression import GeneralizedLinearRegression, AFTSurvivalRegression, DecisionTreeRegressor
from process import data
import time
# AFTSurvivalRegression
def AFT(df_data):
    print("Train a AFTSurvivalRegression model...")
    quantileProbabilities = [0.4, 0.5]
    t1 = time.time()
    # Chain indexer and tree in a Pipeline

    aft_model = AFTSurvivalRegression(quantileProbabilities=quantileProbabilities,
                                quantilesCol="quantiles")\
        .fit(df_data)
    t2 = time.time() - t1
    print("aft_model using time: %.2fs\n" % t2)
    return aft_model
# GeneralizedLinearRegression
def GL_for(df_data):
    print("Train a GeneralizedLinearRegression model...")
    t1 = time.time()
    family = ['gaussian', 'binomial', 'poisson']
    for family_name in family:
        gl_model = GeneralizedLinearRegression(family=family_name, link="identity", maxIter=10, regParam=0.3) \
            .setFeaturesCol("features") \
            .setLabelCol("label") \
            .fit(df_data)
    t2 = time.time() - t1
    print("gl_model using time: %.2fs\n" % t2)
    return gl_model

# GeneralizedLinearRegression with guassian
def GL(df_data):
    print("Train a GeneralizedLinearRegression model...")
    t1 = time.time()
    gl_model = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3) \
        .setFeaturesCol("features") \
        .setLabelCol("label") \
        .fit(df_data)
    t2 = time.time() - t1
    print("gl_model using time: %.2fs\n" % t2)
    return gl_model

# DecisionTreeRegressor
def DTR(df_data):
    # Train a DecisionTree model.
    print("Train a DecisionTree model...")
    t1 = time.time()
    dt = DecisionTreeRegressor(featuresCol="indexedFeatures")
    # Chain indexer and tree in a Pipeline
    pipeline = Pipeline(stages=[data.feature_indexer(df_data), dt])
    # Train model.  This also runs the indexer.
    dtr_model = pipeline.fit(df_data)
    t2 = time.time() - t1
    print("dt_model using time: %.2fs\n" % t2)
    return dtr_model