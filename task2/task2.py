from __future__ import print_function

import os
import sys
import math

from pyspark import SparkContext, SparkConf


def set_up_env():
    os.environ["SPARK_HOME"] = "spark-3.3.1-bin-hadoop3"
    os.environ["PYSPARK_PYTHON"] = sys.executable


if __name__ == "__main__":
    set_up_env()

    conf = SparkConf().setAppName('task2').setMaster('local[2]')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # 0. Удаляем папку data, если она существует, так как у saveAsTextFile нет агрумента overwrite
    os.system('rm -rf task2/data')

    # 1. Создайте RDD с числами от 1 до 100 000 из 10 партиций
    distData = sc.parallelize(c=range(1, 100_001), numSlices=10)

    # 2. Сохраните коллекцию в текстовый файл
    distData.saveAsTextFile("task2/data")

    # 3. Загрузите RDD из файла
    distData = sc.textFile("task2/data").map(lambda x: int(x))

    # 4. Превратите эту RDD в RDD пар, где ключ - это остаток от деления исходного числа на 100, а значение - значение логарифма по основанию e (натурального логарифма) от исходного числа
    distDataKeyVal = distData.map(lambda x: (x % 100, math.log(x)))

    # 5. Сгруппируйте по ключу полученную RDD и посчитайте для каждого ключа количество элементов - значений, у которых 1 цифра после запятой - это четное число.
    count = distDataKeyVal \
        .filter(lambda x: int((abs(x[1]) * 10) % 10) % 2 == 0) \
        .countByKey()

    # 6. Распечатайте результат группировки в стандартный поток вывода (на экран)
    print(count)

    sc.stop()
