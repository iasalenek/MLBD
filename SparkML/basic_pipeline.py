import os
import sys

from pyspark.ml import Pipeline
from pyspark.ml.classification import (DecisionTreeClassifier, LinearSVC,
                                       LogisticRegression, NaiveBayes,
                                       RandomForestClassifier)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

from pyspark.sql import SparkSession
from pyspark.sql.functions import count_distinct

MUSHROOMS_DATA = 'SparkML/mushrooms.csv'


def set_up_env():
    os.environ["SPARK_HOME"] = "spark-3.3.1-bin-hadoop3"
    os.environ["PYSPARK_PYTHON"] = sys.executable


def getSparkSession():
    set_up_env()

    spark = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("Titanic") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def readMushrooms(spark):
    mushrooms = spark.read \
        .option("delimiter", ",") \
        .option("header", "true") \
        .csv(MUSHROOMS_DATA)

    return mushrooms


if __name__ == '__main__':

    spark = getSparkSession()

    mushrooms = readMushrooms(spark)

    models = {
        'Linear Regression': LogisticRegression(),
        'Naive Bayes': NaiveBayes(),
        'SVM': LinearSVC(),
        'Decision Trees': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }

    training, test = mushrooms.randomSplit([0.7, 0.3], seed=12345)

    # Считаем число уникальных значений для каждой из колонок
    columns_count_distinct = training.agg(
        *(count_distinct(training[column]).alias(column)
            for column in training.columns)).collect()[0].asDict()

    # Оставляем только информативные колонки
    # (где число различных значений больше 1)
    info_columns = [k for (k, v) in columns_count_distinct.items() if v > 1]

    indexers = [StringIndexer(
                    inputCol=column, outputCol=column+"_index"
                    ) for column in info_columns]

    one_hot_encoder = OneHotEncoder() \
        .setInputCols([column+"_index" for column in info_columns[1:]]) \
        .setOutputCols([column+"_one_hot" for column in info_columns[1:]])

    assembler = VectorAssembler(
        inputCols=[column+"_one_hot" for column in info_columns[1:]],
        outputCol="features")

    evaluator = MulticlassClassificationEvaluator(
        labelCol="class_index",
        predictionCol="prediction",
        metricName="accuracy",
        )

    preprocessing_pipeline = Pipeline(stages=[
        *indexers,
        one_hot_encoder,
        assembler,
        ])

    preprocessor = preprocessing_pipeline.fit(training)
    training_pre = preprocessor.transform(training).cache()
    test_pre = preprocessor.transform(training).cache()

    print("---------- model`s Test Errors ----------")

    for name, model in models.items():

        model.setLabelCol("class_index").setFeaturesCol("features")
        model_fitted = model.fit(training_pre)
        predictions = model_fitted.transform(test_pre)

        accuracy = evaluator.evaluate(predictions)
        print(f"- {name} = %g " % (1.0 - accuracy))

    spark.stop()
