/** INF 553 - Homework 3: Model-based CF
  * Adam Vaccaro
  */

import java.io.{File, PrintWriter}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.SparkSession
import scala.collection.mutable.HashMap



object ModelBasedCF {
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
        Rating(userID.toInt, productID.toInt, rating.toInt)
      })
    val trainingDataShifted = trainingData
      .map({ case Rating(userID, productID, rating) => Rating(userID, productID, rating - 3) })

    // Load and format test data
    val testDataRaw = sc.textFile(testingPath)
    val testDataHeader = testDataRaw.first()
    val testData = testDataRaw.filter(row => row != testDataHeader)
    val testDataRatings = testData.map(_.split(',') match { case Array(userID, productID, rating) =>
      Rating(userID.toInt, productID.toInt, rating.toInt)
    })
    val testDataShifted = testDataRatings.map { case Rating(userID, productID, rating) => (userID, productID, rating) }

    val testDataPairs = testDataRatings.map { case Rating(userID, productID, rating) => (userID, productID) }

    // Remove rows in the training data that are found in the test data:
    val testDataList = testDataPairs.collect.toList
    val filteredTrainingData = trainingDataShifted.filter(rating => {
      var shouldInclude = true
      testDataList.foreach(testRating => {
        if (rating.user == testRating._1 && rating.product == testRating._2) {
          shouldInclude = false
        }
      })
      shouldInclude
    })

    // Train predictive model:
    val rank = 6
    val numIterations = 13
    val lambda = 0.5
    val numBlocks = -1
    val seed = 3
    val model = ALS.train(filteredTrainingData, rank, numIterations, lambda, numBlocks, seed)

    // Make predictions
    val rawPredictions = model.predict(testDataPairs).map({
      case Rating(userID, productID, rating) => ((userID, productID), rating)
    })

    // The model will sometimes predict ratings < 1 or > 5. Since the scale is 1 to 5, these outside ratings should be
    // set to within the range.
    val predictions = rawPredictions.map(prediction => {
      val userID = prediction._1._1
      val productID = prediction._1._2
      val rating = prediction._2 + 3 //shift ratings back up to 1-5

      if (rating < 1) {
        ((userID, productID), 1.0)
      } else if (rating > 5) {
        ((userID, productID), 5.0)
      } else {
        ((userID, productID), rating)
      }
    })

    // Join original ratings with predictions:
    val ratingsWithPredictions = testDataRatings.map { case Rating(userID, productID, rating) =>
      ((userID, productID), rating)
    }.join(predictions)

    // Calculate RMSE and bucket predictions by accuracy:
    var n: Double = 0
    var rmseSum: Double = 0
    var accuracyBuckets: HashMap[String, Int] = new HashMap()
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
}

