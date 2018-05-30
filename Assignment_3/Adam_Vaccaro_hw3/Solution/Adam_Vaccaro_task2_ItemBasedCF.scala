/** INF 553 - Homework 3: User-based CF
  * Adam Vaccaro
  */

import java.io.{File, PrintWriter}
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.HashMap

object ItemBasedCF {
  def main(args: Array[String]): Unit = {
    val startTime = System.currentTimeMillis() / 1000

    // Parse inputs:
    val trainingPath = args(0)
    val similarityPath = args(1)
    val testingPath = args(2)
    val outputPath = args(3)

    // set up Spark Session:
    val spark = SparkSession.builder()
      .appName("Spark Local")
      .config("spark.master", "local")
      .getOrCreate()
    val sc = spark.sparkContext
    //sc.setLogLevel("ERROR")


    // Load and format raw training data
    val trainingDataRaw = sc.textFile(trainingPath)
    val header = trainingDataRaw.first() //extract header
    val trainingData = trainingDataRaw
      .filter(row => row != header)
      .map(_.split(',') match { case Array(userID, productID, rating, timestamp) =>
        (userID.toInt, productID.toInt, rating.toInt - 3)
      })
      .repartition(8)

    // Load and format test data
    val testDataRaw = sc.textFile(testingPath)
    val testDataHeader = testDataRaw.first()
    val testData: RDD[((Int, Int), Int)] =  testDataRaw
      .filter(row => row != testDataHeader)
      .map(_.split(',') match { case Array(userID, productID, rating) =>
        ((userID.toInt, productID.toInt), rating.toInt)
      })
    val testDataPairs = testData.map { case ((userID, productID), rating) => (userID, productID) }

    // Remove rows in the training data that are found in the test data:
    val testDataList = testDataPairs.collect.toList
    val filteredTrainingData = trainingData.filter(rating => {
      var shouldInclude = true
      testDataList.foreach(testRating => {
        if (rating._1 == testRating._1 && rating._2 == testRating._2) {
          shouldInclude = false
        }
      })
      shouldInclude
    })

    // Load in results CSV:
    val similarityDataRaw = sc.textFile(similarityPath)
    val similarityData = similarityDataRaw
      .map(line => line.split(", "))
      .map(row => row.map(elem => elem.toFloat))
    val similarPairs = ListBuffer.empty[Set[Int]]
    for (row <- similarityData.collect()) {
      val tempSet = Set(row(0).toInt, row(1).toInt)
      similarPairs += tempSet
    }


    // Create dictionary: UserID => { ProductID => Rating }
    val userHashMap = HashMap.empty[Int, HashMap[Int, Int]]
    // Create dictionary: ProductID => {UserID => Rating}
    val productHashMap = HashMap.empty[Int, HashMap[Int, Int]]
    // and map from ProductID => UserIDs
    val productToUsersMap = HashMap.empty[Int, Set[Int]]
    // and map from UserID => ProductIDs
    val userToProductsMap = HashMap.empty[Int, Set[Int]]
    for (tuple3 <- filteredTrainingData.collect.toList) {
      val userID = tuple3._1
      val productID = tuple3._2
      val rating = tuple3._3

      if (!userHashMap.contains(userID)) {
        userHashMap += (userID -> HashMap.empty[Int, Int])
      }
      userHashMap(userID) += (productID -> rating)

      if (!productHashMap.contains(productID)) {
        productHashMap += (productID -> HashMap.empty[Int, Int])
      }
      productHashMap(productID) += (userID -> rating)

      if (productToUsersMap.contains(productID)) {
        productToUsersMap(productID) += userID
      } else {
        productToUsersMap += (productID -> Set(userID))
      }

      if (userToProductsMap.contains(userID)) {
        userToProductsMap(userID) += productID
      } else {
        userToProductsMap += (userID -> Set(productID))
      }
    }


    // Create list of distinct products:
    val uniqueProductsList = filteredTrainingData
      .map(tuple3 => tuple3._2)
      .distinct
      .collect
      .toList


    val predictions = runPredictions(testDataPairs, productHashMap, userHashMap, userToProductsMap, similarPairs)//, corrHashMap)

    // Join original ratings with predictions
    val ratingsWithPredictions = testData.join(predictions)

    // Calculate RMSE and bucket predictions by accuracy:
    var n : Double = 0
    var rmseSum : Double = 0
    var accuracyBuckets : HashMap[String, Int] = new HashMap()
    accuracyBuckets += (">=0 and <1" -> 0)
    accuracyBuckets += (">=1 and <2" -> 0)
    accuracyBuckets += (">=2 and <3" -> 0)
    accuracyBuckets += (">=3 and <4" -> 0)
    accuracyBuckets += (">=4" -> 0)
    ratingsWithPredictions.collect.toList.foreach(entry => {
      // each entry is of the form ((userID, productID), (rating, prediction))
      val prediction = entry._2._2
      val rating = entry._2._1
      val absoluteDifference = Math.abs(prediction - rating)
      rmseSum += (absoluteDifference * absoluteDifference)
      n += 1
      if (absoluteDifference >= 0 && absoluteDifference < 1) {
        accuracyBuckets(">=0 and <1") += 1
      } else if (absoluteDifference >= 1 && absoluteDifference < 2) {
        accuracyBuckets(">=1 and <2") += 1
      } else if (absoluteDifference >= 2 && absoluteDifference < 3) {
        accuracyBuckets(">=2 and <3") += 1
      } else if (absoluteDifference >= 3 && absoluteDifference < 4) {
        accuracyBuckets(">=3 and <4") += 1
      } else {
        accuracyBuckets(">=4") += 1
      }
    })

    // Write results to file:
    val sortedPredictions = predictions.sortBy(prediction => prediction._1)
      .map(tuple => (tuple._1._1, tuple._1._2, tuple._2))
      .collect()
      .toList
    val pw = new PrintWriter(new File(outputPath))
    for (tuple <- sortedPredictions) {
      val line = tuple.productIterator.mkString(", ")
      pw.write(line + "\n")
    }
    pw.close()

    // Print accuracy metrics to console:
    val bucketOrder = List(">=0 and <1", ">=1 and <2", ">=2 and <3", ">=3 and <4", ">=4")
    for (bucket <- bucketOrder) {
      val count = accuracyBuckets(bucket)
      println(s"$bucket: $count")
    }
    val rmse: Double = Math.sqrt(rmseSum / n)
    println(s"RMSE: $rmse")
    val endTime = System.currentTimeMillis() / 1000
    val durationSeconds = endTime - startTime
    println(s"Time: $durationSeconds seconds")
  }

  def calculatePearsons(productID1: Int,
                        productID2: Int,
                        productHashMap: HashMap[Int, HashMap[Int, Int]]): Double = {
    val productProfile1 = productHashMap(productID1)
    val productProfile2 = productHashMap(productID2)

    val commonUsers = productProfile1.keySet.intersect(productProfile2.keySet)
    if (commonUsers.size == 0) {
      return 0.0
    }

    // Calculate the average rating across common users for each product
    var productSum1 = 0
    var productSum2 = 0
    commonUsers.foreach(userID => {
      productSum1 += productProfile1(userID)
      productSum2 += productProfile2(userID)
    })
    val meanProduct1 = productSum1 / commonUsers.size
    val meanProduct2 = productSum2 / commonUsers.size

    var numerator = 0
    var denominator1 = 0
    var denominator2 = 0
    commonUsers.foreach(userID => {
      val meanDifference1 = productProfile1(userID) - meanProduct1
      val meanDifference2 = productProfile2(userID) - meanProduct2

      numerator += meanDifference1 * meanDifference2

      denominator1 += meanDifference1 * meanDifference1

      denominator2 += meanDifference2 * meanDifference2
    })

    var correlation = numerator.toDouble/(Math.sqrt(denominator1) * Math.sqrt(denominator2))

    if (correlation.isNaN) {
      correlation = 0.0
    }

    correlation
  }

  def runPredictions(testDataPairs: RDD[(Int, Int)],
                     productHashMap: HashMap[Int, HashMap[Int, Int]],
                     userHashMap: HashMap[Int, HashMap[Int, Int]],
                     userToProductsMap: HashMap[Int, Set[Int]],
                     similarPairs: ListBuffer[Set[Int]]
                     //corrHashMap: HashMap[Set[Int], Double]
                    ): RDD[((Int, Int), Double)] = {
    val predictions = testDataPairs.map(pair => {
      val activeUser = pair._1
      val productID = pair._2
      val k = 3
      val neighborhood = findNeighborhood(activeUser, productID, userToProductsMap, productHashMap, similarPairs)//, corrHashMap)
        .filter(!_._2.isNaN)
        .filter(_._2 != 0.0)
        .filter(_._1 != productID)
        .sortBy(-1 * _._2)
        .take(k)

      val pred = calculatePrediction(activeUser, productID, neighborhood, userHashMap) + 3

      ((activeUser, productID), pred)
    })

    predictions
  }

  def findNeighborhood(activeUser: Int,
                       productID: Int,
                       userToProductsMap: HashMap[Int, Set[Int]],
                       productHashMap: HashMap[Int, HashMap[Int, Int]],
                       similarPairs: ListBuffer[Set[Int]]
                       //corrHashMap: HashMap[Set[Int], Double]
                      ): ListBuffer[Tuple2[Int, Double]] = {
    val productsRatedSet = userToProductsMap(activeUser)
    val similarItems = ListBuffer.empty[Int]
    for (set <- similarPairs) {
      if (set.contains(productID)) {
        for (item <- set) {
          if (item != productID) {
            similarItems += item
          }
        }
      }
    }

    val neighborhood = ListBuffer.empty[Tuple2[Int, Double]]

    //for (item <- productsRatedSet) {
      //val correlation = corrHashMap(Set(productID, item))
    for (item <- similarItems) {
      if (productsRatedSet.contains(item)) {
        val correlation = calculatePearsons(productID, item, productHashMap)
        val tempTuple = (item, correlation)

        neighborhood += tempTuple
      }
    }
    neighborhood
  }

  def calculatePrediction(activeUser: Int,
                          productID: Int,
                          neighborhood: ListBuffer[Tuple2[Int, Double]],
                          userHashMap: HashMap[Int, HashMap[Int, Int]]
                         ): Double = {
    val activeRatings = userHashMap(activeUser)
    val activeMean = activeRatings.foldLeft(0)(_ + _._2) / activeRatings.size

    val denominator = neighborhood
      .map(tuple => Math.abs(tuple._2))
      .sum

    val numerator = neighborhood
      .map(neighbor => {
        val neighborID = neighbor._1
        val neighborCorr = neighbor._2

        val neighborRatings = userHashMap(activeUser)
        //val neighborRatingsWithoutProduct = neighborRatings.filter(_._1 != productID)

        //val neighborMean = neighborRatingsWithoutProduct.foldLeft(0)(_ + _._2) / neighborRatings.size
        val neighborRating = neighborRatings(neighborID)

        (neighborRating ) * neighborCorr
      })
      .sum

    var prediction = numerator/denominator

    if (prediction.isNaN) {
      prediction = activeMean
    }

    prediction
  }
}