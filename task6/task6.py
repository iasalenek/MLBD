from __future__ import print_function

import os
import sys

from pyspark.sql import SparkSession


def set_up_env():
    os.environ["SPARK_HOME"] = "spark-3.3.1-bin-hadoop3"
    os.environ["PYSPARK_PYTHON"] = sys.executable


if __name__ == "__main__":
    set_up_env()

    spark = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("DataFrame Intro") \
        .getOrCreate()

    salaries = [
        "John 1900 January",
        "Mary 2000 January",
        "John 1800 February",
        "John 1000 March",
        "Mary 1500 February",
        "Mary 2900 March",
        "Mary 1600 April",
        "John 2800 April",
    ]

    # Приводим к необходимому виду
    data = [(n, int(s), m) for n, s, m in [x.split(" ") for x in salaries]]

    columns = ["name", "salary", "month"]
    df = spark.createDataFrame(data, schema=columns)

    answer = df.groupby('name').avg()

    answer.write.mode("overwrite").json('task6/answer')

    spark.stop()
