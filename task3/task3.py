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

    jvmLanguages = sc.parallelize(
        ["Scala", "Java", "Groovy", "Kotlin", "Ceylon"])
    functionalLanguages = sc.parallelize(
        ["Scala", "Kotlin", "JavaScript", "Haskell", "Python"])
    webLanguages = sc.parallelize(
        ["PHP", "Ruby", "Perl", "JavaScript", "Python"])
    mlLanguages = sc.parallelize(
        ["JavaScript", "Python", "Scala"])

    # 1. Найдите все ЯП, запускаемые на JVM и имеющие поддержку ML
    JVM_ML = jvmLanguages \
        .intersection(mlLanguages) \
        .collect()
    print(', '.join(JVM_ML))

    # 2. Найдите все ЯП, подходящие для web development, но не являющиеся функциональными
    WD_not_Func = webLanguages \
        .subtract(functionalLanguages) \
        .collect()
    print(', '.join(WD_not_Func))

    # 3. Выведите объединенный список ЯП, запускаемых на JVM и являющихся функциональными, убрав оттуда все дубликаты
    JVM_Func_Unique = jvmLanguages \
        .union(functionalLanguages) \
        .distinct() \
        .collect()
    print(', '.join(JVM_Func_Unique))

    sc.stop()
