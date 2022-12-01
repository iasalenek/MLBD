from __future__ import print_function

import os
import sys

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType

# Штаты, граничащие с Канадой
# Аляска, Мичиган, Мэн, Миннесота, Монтана, Нью-Йорк, Вашингтон, Северная Дакота, Огайо, Вермонт, Нью-Гэмпшир, Айдахо, Пенсильвания. 
BORDERS_WITH_CANADA = ['AK', 'MI', 'ME', 'MN', 'MT', 'NY', 'WA', 'ND', 'OH', 'VT', 'NH', 'ID', 'PA']


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

    stateNames = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv('task7/StateNames.csv')

    # Функции для фильтрации
    def isCanadaNeighbourFunction(stateName):
        return stateName.isin(BORDERS_WITH_CANADA)

    def isWorldWarTwoYearFunction(year):
        return True if (1939 <= year <= 1945) else False

    isCanadaNeighbour = udf(
        lambda stateName: isCanadaNeighbourFunction(stateName), BooleanType())

    isWorldWarTwoYear = udf(
        lambda year: isWorldWarTwoYearFunction(year), BooleanType())

    answer = stateNames \
        .filter(isWorldWarTwoYear(stateNames["Year"]) == True) \
        .filter(isCanadaNeighbourFunction(stateNames["State"]) == True) \
        .select("Name").distinct() \

    answer.write.mode("overwrite").parquet('task7/answer')
