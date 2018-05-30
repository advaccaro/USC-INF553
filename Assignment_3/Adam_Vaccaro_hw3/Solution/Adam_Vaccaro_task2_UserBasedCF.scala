/** INF 553 - Homework 3: User-based CF
  * Adam Vaccaro
  */

import java.io.{File, PrintWriter}
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.HashMap

object UserBasedCF {
  def main(args: Array[String]): Unit = {
    val startTime = System.currentTimeMillis() / 1000

    // Parse inputs:
    val trainingPath = args(0)
    val testingPath = args(1)
    val outputPath = args(2)

    // set up Spark Session:
    val spark = SparkSession.builder()
      .appName("Spark Local")
      .config("spark.master", "local")
      .getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    // Load and format training data
    val trainingDataRaw = sc.textFile(trainingPath)
    val header = trainingDataRaw.first() //extract header
    val trainingData = trainingDataRaw
      .filter(row => row != header)
      .map(_.split(',') match { case Array(userID, productID, rating, timestamp) =>
        (userID.toInt, productID.toInt, rating.toInt - 3)
      })

    // Load and format test data
    val testDataRaw = sc.textFile(testingPath)
    val testDataHeader = testDataRaw.first()
    val testData = testDataRaw
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

    // Create dictionary: UserID => { ProductID => Rating }
    val userHashMap = HashMap.empty[Int, HashMap[Int, Int]]
    // and map from ProductID => UserIDs
    val productToUsersMap = HashMap.empty[Int, Set[Int]]
    for (tuple3 <- filteredTrainingData.collect.toList) {
      val userID = tuple3._1
      val productID = tuple3._2
      val rating = tuple3._3

      if (!userHashMap.contains(userID)) {
        userHashMap += (userID -> HashMap.empty[Int, Int])
      }
      userHashMap(userID) += (productID -> rating)

      if (productToUsersMap.contains(productID)) {
        productToUsersMap(productID) += userID
      } else {
        productToUsersMap += (productID -> Set(userID))
      }
    }

    // Create set and list of distinct users:
    val uniqueUsersList = filteredTrainingData
      .map(tuple3 => tuple3._1)
      .distinct
      .collect
      .toList

    // Calculate Pearson's -> ((user1, user2), corr):
    val corrHashMap = HashMap.empty[Set[Int], Double]
    for (user1 <- uniqueUsersList) {
      for (user2 <- uniqueUsersList) {
        if (user1 != user2) {
          val userSet = Set(user1, user2)
          if (!corrHashMap.contains(userSet)) {
            val corr = calculatePearsons(user1, user2, userHashMap)
            val key = Set(user1, user2)
            corrHashMap += (key -> corr)
          }
        }
      }
    }
    val predictions = runPredictions(testDataPairs, userHashMap, productToUsersMap, corrHashMap)

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

  def calculatePearsons(userID1: Int,
                        userID2: Int,
                        userHashMap: HashMap[Int, HashMap[Int, Int]]): Double = {
    val userProfile1 = userHashMap(userID1)
    val userProfile2 = userHashMap(userID2)

    val commonProducts = userProfile1.keySet.intersect(userProfile2.keySet)
    if (commonProducts.size == 0) {
      return 0.0
    }
    //    println("common product size", commonProducts.size)

    // Calculate the average rating across common products for each user
    var userSum1 = 0
    var userSum2 = 0
    commonProducts.foreach(productID => {
      userSum1 += userProfile1(productID)
      userSum2 += userProfile2(productID)
    })
    val meanUser1 = userSum1 / commonProducts.size
    val meanUser2 = userSum2 / commonProducts.size
    //    println("meanUser1", meanUser1)
    //    println("meanUser2", meanUser2)
    var numerator = 0
    var denominator1 = 0
    var denominator2 = 0
    commonProducts.foreach(productID => {
      val meanDifference1 = userProfile1(productID) - meanUser1
      val meanDifference2 = userProfile2(productID) - meanUser2

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
                     userHashMap: HashMap[Int, HashMap[Int, Int]],
                     productToUsersMap: HashMap[Int, Set[Int]],
                     corrHashMap: HashMap[Set[Int], Double]
                    ): RDD[((Int, Int), Double)] = {
    val predictions = testDataPairs.map(pair => {
      val activeUser = pair._1
      val productID = pair._2
      val k = 3
      val neighborhood = findNeighborhood(activeUser, productID, userHashMap, productToUsersMap, corrHashMap)
        .filter(!_._2.isNaN)
        .filter(_._2 != 0.0)
        .filter(_._1 != activeUser)
        .sortBy(-1 * _._2)
        .take(k)

      val pred = calculatePrediction(activeUser, productID, neighborhood, userHashMap) + 3

      ((activeUser, productID), pred)
    })

    predictions
  }

  def findNeighborhood(activeUser: Int,
                       productID: Int,
                       userHashMap: HashMap[Int, HashMap[Int, Int]],
                       productToUsersMap: HashMap[Int, Set[Int]],
                       corrHashMap: HashMap[Set[Int], Double]
                      ): ListBuffer[Tuple2[Int, Double]] = {
    val usersRatedSet = productToUsersMap(productID)

    val neighborhood = ListBuffer.empty[Tuple2[Int, Double]]

    for (userID <- usersRatedSet) {
      val correlation = corrHashMap(Set(userID, activeUser))
      val tempTuple = (userID, correlation)

      neighborhood += tempTuple
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

        val neighborRatings = userHashMap(neighborID)
        val neighborRatingsWithoutProduct = neighborRatings.filter(_._1 != productID)

        val neighborMean = neighborRatingsWithoutProduct.foldLeft(0)(_ + _._2) / neighborRatings.size
        val neighborRating = neighborRatings(productID)

        (neighborRating - neighborMean) * neighborCorr
      })
      .sum

    var prediction = activeMean + numerator/denominator

    if (prediction.isNaN) {
      prediction = activeMean
    }

    prediction
  }
}