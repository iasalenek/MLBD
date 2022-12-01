val x = List.range(1, 100)
val RDD = sc.parallelize(x, 3)
val count = RDD.reduce(_ + _)
println(count)
