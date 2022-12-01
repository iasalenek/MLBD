from __future__ import print_function

import os
import sys
import math

import findspark
from pyspark import SparkContext, SparkConf

from pyspark.ml.stat import KolmogorovSmirnovTest


def set_up_env():
    os.environ["SPARK_HOME"] = "spark-3.3.1-bin-hadoop3"
    os.environ["PYSPARK_PYTHON"] = sys.executable

if __name__ == "__main__":
    set_up_env()

    conf = SparkConf().setAppName('task2').setMaster('local[2]')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    Seq = [-1, 1, 3, 2, 2, 150, 1, 2, 3, 2, 2, 1, 1, 1, -100, 2, 2, 3, 4, 1, 1, 3, 4]

    distSeq = sc.parallelize(Seq).cache()
    sigma = distSeq.sampleStdev()
    outliers = distSeq.filter(lambda x: x > 3 * sigma)
    outliers_q = outliers.count() / distSeq.count()


    print(f'Sigma: {sigma}')
    print(f'Percantage of outliers: {outliers_q}')
    print(f'Normality: {outliers_q < 0.0027}')
    

