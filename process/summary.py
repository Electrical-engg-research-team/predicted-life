from pyspark.ml.evaluation import RegressionEvaluator
from process import data

def predicted_dt(model, df_data):
    dt_pred = model.transform(df_data)
    dt_result = dt_pred.select("features", "label", "prediction")
    dt_result.show(10)
    return dt_pred
def predicted_gl(model, df_data):
    gl_pred = model.transform(data.testData(df_data))
    print("Print the coefficients and intercept for generalized linear regression model :")
    # print("Coefficients: " + str(model.coefficients))
    # print("Intercept: " + str(model.intercept))
    gl_result = gl_pred.select("features", "label", "prediction")
    gl_result.show(10)
    return gl_pred
def predicted_aft(model, df_data):
    aft_pred = model.transform(df_data)
    aft_result = aft_pred.select("features", "label", "prediction")
    aft_result.show(5)
    return aft_pred

def summ(gl_model):
    summary = gl_model.summary
    print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
    print("T Values: " + str(summary.tValues))
    print("P Values: " + str(summary.pValues))
    print("Dispersion: " + str(summary.dispersion))
    print("Null Deviance: " + str(summary.nullDeviance))
    print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
    print("Deviance: " + str(summary.deviance))
    print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
    print("AIC: " + str(summary.aic))
    print("Deviance Residuals: ")
    summary.residuals().show(5)

# Select (prediction, true label) and compute test error
# evaluator
def evalute(predictions):
    mse = RegressionEvaluator(metricName="mse").evaluate(predictions)
    print("Mean Squared Error (MSE) on test data = %g" % mse)
    rmse = RegressionEvaluator(metricName="rmse").evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    mae = RegressionEvaluator(metricName="mae").evaluate(predictions)
    print("Mean Absolute Error(EMA) on test data = %g" % mae)
    r2 = RegressionEvaluator(metricName="r2").evaluate(predictions)
    print("R^2 on test data = %.3f" % r2)

def evaluator(predictions):
    metricNames = ['mse', 'rmse', 'mae', 'r2']
    for metric in metricNames:
        evaluator = RegressionEvaluator(metricName=metric)
        prams = evaluator.evaluate(predictions)
        print(metric.title(), prams)
    rmse = RegressionEvaluator(metricName="rmse").evaluate(predictions)
    return rmse
