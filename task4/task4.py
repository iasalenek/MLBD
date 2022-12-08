from __future__ import print_function

import os
import sys

from pyspark import SparkContext, SparkConf


def set_up_env():
    os.environ["SPARK_HOME"] = "spark-3.3.1-bin-hadoop3"
    os.environ["PYSPARK_PYTHON"] = sys.executable


if __name__ == "__main__":
    set_up_env()

    conf = SparkConf().setAppName('task2').setMaster('local[2]')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

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

    ####
    def split(line):
        name, salary, mounth = line.split(" ")
        return (name, int(salary))

    def print_answer(answer):
        for person in answer:
            print(f'{person[0]}: {person[1]}')

    answer = sc.parallelize(salaries) \
        .map(lambda x: split(x)) \
        .mapValues(lambda x: (x, 1)) \
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
        .mapValues(lambda x: x[0] / x[1]).collect()

    print_answer(answer)

    sc.stop()