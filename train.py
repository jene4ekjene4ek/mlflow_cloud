"""
PySpark Decision Tree Regression Example.
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark
# from common import *



print("MLflow Version:", mlflow.version.VERSION)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

metrics = ["rmse","r2", "mae"]

# def train(data, maxDepth, maxBins):
#     data_path = "s3a://orlow-cos/simple-pyspark-model/sample_libsvm_data.txt"
#     data = spark.read.format("libsvm").load(data_path)
#     (trainingData, testData) = data.randomSplit([0.7, 0.3], 2019)
def train(data, max_depth, max_bins):
    print("Parameters: max_depth: {}  max_bins: {}".format(max_depth,max_bins))
#     spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()

    # Load the data stored in LIBSVM format as a DataFrame.
    data = spark.read.format("libsvm").load(data_path)

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    feature_indexer = VectorIndexer(
        inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_bins", max_bins)
    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="indexedLabel",
                                featuresCol="indexedFeatures",
                                maxDepth=max_depth,
                                maxBins=max_bins)

    # Chain indexers and tree in a Pipeline.
    pipeline = Pipeline(stages=[label_indexer, feature_indexer, dt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)

    # Select (prediction, true label) and compute test error.
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel",
                                                  predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    test_error = 1.0 - accuracy
    print("Test Error = {} ".format(test_error))
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("test_error", test_error)

    tree_model = model.stages[2]
    print(tree_model)
    
    mlflow.spark.log_model(model, '')
    
    spark.stop()


if __name__ == "__main__":
    spark = SparkSession.builder.appName("App").getOrCreate()
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", default="pyspark", required=False)
    parser.add_argument("--data_path", dest="data_path", help="data_path", required=True)
    parser.add_argument("--max_depth", dest="max_depth", help="max_depth", default=2, type=int)
    parser.add_argument("--max_bins", dest="max_bins", help="max_bins", default=32, type=int)
    parser.add_argument("--describe", dest="describe", help="Describe data", default=False, action='store_true')
    args = parser.parse_args()

    client = mlflow.tracking.MlflowClient()
    print("experiment_name:",args.experiment_name)
    mlflow.set_experiment(args.experiment_name)
    print("experiment_id:",client.get_experiment_by_name(args.experiment_name).experiment_id)

    
    with mlflow.start_run() as run:
        print("MLflow:")
        print("  run_id:",run.info.run_uuid)
        print("  experiment_id:",run.info.experiment_id)
        train(data, args.max_depth,args.max_bins)
