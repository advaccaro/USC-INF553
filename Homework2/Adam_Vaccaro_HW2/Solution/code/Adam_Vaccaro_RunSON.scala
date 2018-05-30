

/** INF 553 - Homework 2: SON Algorithm
  * Adam Vaccaro
  */

object RunSON {

  import org.apache.spark.sql.SparkSession
  import scala.collection.mutable.HashMap
  import scala.collection.mutable.ListBuffer
  import scala.collection.mutable.{Set => mutSet}
  import java.io.PrintWriter
  import java.io.File

  def createCaseBaskets(fileRDD: org.apache.spark.rdd.RDD[String], caseNumber: Int): org.apache.spark.rdd.RDD[Set[String]] = {
    // create RDD of baskets:
    val basketRDD = fileRDD
      .map(line => line.split(",").dropRight(1))
      .mapPartitionsWithIndex((idx, iter) => if (idx == 0) iter.drop(1) else iter)
      .groupBy(row => if (caseNumber == 1) row(0) else if (caseNumber == 2) row(1))
      .values
      .map(basket => basket.map(pair => if (caseNumber == 1) pair(1) else pair(0)).toSet)
    basketRDD
  }

  def createBasketsIntRDDandLookUp(sc: org.apache.spark.SparkContext, filepath: String, caseNumber: Int) : Tuple2[HashMap[Int, String], org.apache.spark.rdd.RDD[Set[Int]]] = {
    val basketsRDD = createCaseBaskets(sc.textFile(filepath), caseNumber)
    val intLookUpTablePrimer = HashMap.empty[String, Int]
    var counter = 0
    var basketsIntArray = basketsRDD.collect().map(basket => {
      var basketIntSet = mutSet.empty[Int]
      val rowList = basket.toList.distinct
      for (elem <- rowList) {
        if (intLookUpTablePrimer.contains(elem)) {
          val stringNum = intLookUpTablePrimer(elem)
          basketIntSet += stringNum
        } else {
          intLookUpTablePrimer += (elem -> counter)
          val stringNum = counter
          basketIntSet += stringNum
          counter += 1
        }
      }
    basketIntSet.toSet})
    val intLookUpTable : HashMap[Int, String] = for ((k,v) <- intLookUpTablePrimer) yield (v, k)
    val basketsIntRDD : org.apache.spark.rdd.RDD[Set[Int]] = sc.parallelize(basketsIntArray)
    (intLookUpTable, basketsIntRDD)
  }

  def createFrequentSingles(basketsIterator: Iterator[Set[Int]], ps: Float)  : mutSet[Set[Int]] = {
    val singleCounts = HashMap.empty[Int, Int]
    basketsIterator.foreach(basket => {
      basket.foreach(single => {
        if (singleCounts.contains(single)) {
          singleCounts(single) += 1
        } else {
          singleCounts += (single -> 1)
        }
      })
    })
    var frequentSinglesCandidates = mutSet.empty[Set[Int]]
    for ((k, v) <- singleCounts) {
      if (v >= ps) {
        frequentSinglesCandidates += Set(k)
      }
    }
    frequentSinglesCandidates
  }

  def initializeApriori(baskets: Iterator[Set[Int]], ps: Float): Iterator[(Set[Int], Int)] = {
    // duplicate iterator:
    var (basketsIterator1, basketsIterator2) = baskets.duplicate
    var k = 2
    var resultsList: ListBuffer[(Set[Int], Int)] = ListBuffer.empty
    var frequentSinglesCandidates = createFrequentSingles(basketsIterator1, ps)
    frequentSinglesCandidates.foreach(single => {
      val tempTuple = (single, 1)
      resultsList += tempTuple
    })
    var prevCandidateItemsets = frequentSinglesCandidates
    while (prevCandidateItemsets.nonEmpty) {
      val (basketsIteratorTemp, generalizedBasketsIterator) = basketsIterator2.duplicate
      basketsIterator2 = basketsIteratorTemp
      var frequentKItemsetsCandidates = runGeneralizedApriori(generalizedBasketsIterator, prevCandidateItemsets, ps, k)
      if (frequentKItemsetsCandidates.nonEmpty) {
        frequentKItemsetsCandidates.foreach(kItemset => {
          val tempTuple = (kItemset, 1)
          resultsList += tempTuple
        })
      }

      k = k + 1
      prevCandidateItemsets = frequentKItemsetsCandidates
    }
    resultsList.iterator
  }

  def runGeneralizedApriori(baskets: Iterator[Set[Int]], prevFrequentItemsetCandidates: mutSet[Set[Int]], ps: Float, k: Int): mutSet[Set[Int]] = {
    var frequentSingles = mutSet.empty[Int]
    for (itemsets <- prevFrequentItemsetCandidates) {
      for (item <- itemsets) {
        frequentSingles += item
      }
    }
    var possibleFrequentKItemsets = mutSet.empty[Set[Int]]
    for (set <- prevFrequentItemsetCandidates) {
      for (single <- frequentSingles) {
        val possibleKItemset = set + single
        if (possibleKItemset.size == k) {
          var check = true
          if (k > 2) {
            val subItemsets = possibleKItemset.subsets(k - 1)
            for (subset <- subItemsets) {
              if (!prevFrequentItemsetCandidates.contains(subset)) {
                check = false
              }
            }
          }
          if (check) {
            possibleFrequentKItemsets += possibleKItemset
          }
        }
      }
    }

    val kItemsetCounts = HashMap.empty[Set[Int], Int]
    for (basket <- baskets) {
      for (kItemset <- possibleFrequentKItemsets) {
        if (kItemset.subsetOf(basket)) {
          if (kItemsetCounts.contains(kItemset)) {
            kItemsetCounts(kItemset) += 1
          } else {
            kItemsetCounts += (kItemset -> 1)
          }
        }
      }
    }
    var frequentKItemsetsCandidates = mutSet.empty[Set[Int]]
    for ((kItemset, count) <- kItemsetCounts) {
      if (count >= ps) {
        frequentKItemsetsCandidates += kItemset
      }
    }
    frequentKItemsetsCandidates
  }

  def main(args: Array[String]) {
    // set up Spark Session:
    val spark = SparkSession.builder()
      .appName("Spark Local")
      .config("spark.master", "local")
      .getOrCreate()
    val sc = spark.sparkContext
    // parse inputs:
    val caseNumber = args(0).toInt //
    val filepath = args(1)
    val filename = filepath.split("\\.")(0).split("/").last.capitalize
    val support = args(2).toInt
    // convert strings to integers and create integer lookup table:
    val (intLookUpTable, basketsIntRDD) = createBasketsIntRDDandLookUp(sc, filepath, caseNumber)
    // implement SON:
    val numChunks = basketsIntRDD.getNumPartitions
    val p = 1 / numChunks
    val ps = p * support
    // map phase 1 - implement a priori algorithm:
    var mapPhase1 = basketsIntRDD.mapPartitions(baskets => initializeApriori(baskets, ps))
    // reduce phase 1 - collect results from map phase 1 to get set of candidate frequent itemsets:
    var reducePhase1 = mapPhase1.reduceByKey((a: Int, b: Int) => a + b)
    var candidates = reducePhase1.map(tuple => tuple._1).collect().toList
    // map phase 2 - count occurence of candidate itemsets in original dataset:
    var mapPhase2 = basketsIntRDD.flatMap(basket => {
      var candidateCounts = HashMap.empty[Set[Int], Int]
      for (candidate <- candidates) {
        if (candidate.subsetOf(basket)) {
          if (candidateCounts.contains(candidate)) {
            candidateCounts(candidate) += 1
          } else {
            candidateCounts += (candidate -> 1)
          }
        }
      }
      candidateCounts
    })
    // reduce phase 2 - aggregate results of map phase 2 to find true frequent itemsets:
    var reducePhase2 = mapPhase2.reduceByKey((a: Int, b: Int) => a + b)
    var reducePhase2filter = reducePhase2.filter(itemTuple => itemTuple._2 >= support)
    // translate integer IDs back into string IDs:
    val frequentItemsetsRDD = reducePhase2filter.map(itemsetInt => {
        var itemsetString = mutSet.empty[String]
        for (itemInt <- itemsetInt._1) {
            itemsetString += intLookUpTable(itemInt)
          }
        itemsetString
        })
    // sort results:
    var sortedResults = frequentItemsetsRDD
      .map(itemset => itemset.toList.sorted)
      .groupBy(itemset => itemset.size)
      .sortBy(group => group._1)
      .collect()
      .map(group => {
          group._2
            .toList
            .map(itemset => {
              itemset.map(item => "'" + item + "'")
                .mkString("(", ", ", ")")
            })
            .sorted
      })
      .toList
    // write results to output file:
    val outfile = "Adam_Vaccaro_SON_" + filename + ".case" + caseNumber.toString + "-" + support.toString + ".txt"
    val pw = new PrintWriter(new File(outfile))
    for (itemsetList <- sortedResults) {
      val line = itemsetList.mkString(", ")
      pw.write(line + "\n")
    }
    pw.close()
  }
}
