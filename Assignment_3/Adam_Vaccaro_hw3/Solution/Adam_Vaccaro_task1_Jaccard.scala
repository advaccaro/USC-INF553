
/** INF 553 - Homework 3: LSH with Jaccard Similarity
  * Adam Vaccaro
  */

import org.apache.spark.sql.SparkSession
import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.{Set => mutSet}
import util.Random.nextInt
import java.io.PrintWriter
import java.io.File


object JaccardLSH {
  def main(args: Array[String]) = {
    val startTime = System.currentTimeMillis() / 1000
    // Parse inputs:
    val inputPath = args(0)
    val outputPath = args(1)
    // set up Spark Session:
    val spark = SparkSession.builder()
      .appName("Spark Local")
      .config("spark.master", "local")
      .getOrCreate()
    val sc = spark.sparkContext

    // Load in original CSV and construct binary characteristic matrix:
    val (characteristicMatrix, productsHashMap, uniqueProductsList, uniqueUsersList) = constructCharacteristicMatrix(sc.textFile(inputPath))
    val nUsers = characteristicMatrix.size
    val nProducts = characteristicMatrix(0).size

      // Compute hash values and build signature matrix:
      val (signatureMatrix, aValues, bValues) = createSignatureMatrix(characteristicMatrix, nProducts, nUsers)

      // Divide matrix into b bands and r rows b*r=n (number of hash functions) and generate candidate pairs:
      val n = nUsers
      val b = 20
      val r = 6
      var candidatePairsSet = generateCandidatePairs(signatureMatrix, b, r, n, nProducts)
      var nCandidates = candidatePairsSet.size

      // Find candidate pairs w/ Jaccard Similarity >= 0.5:
      var verifiedCandidatesSet = verifyCandidatePairs(candidatePairsSet, productsHashMap, uniqueProductsList, uniqueUsersList)
      var numVerified = verifiedCandidatesSet.size

    // Prepare output and print to text file:
    val resultsList = ListBuffer.empty[Tuple3[Int, Int, Float]]
    for (pair <- verifiedCandidatesSet) {
      val productID1 = pair._1.toInt
      val productID2 = pair._2.toInt
      val similarity = calculateJaccardSimilarity(productsHashMap, productID1, productID2)
      resultsList += Tuple3(productID1, productID2, similarity)
    }

    var sortedResults = sc.parallelize(resultsList)
      .sortBy(tuple => (tuple._1, tuple._2))
      .collect()
      .toList

    // write results to output file:
    val pw = new PrintWriter(new File(outputPath))
    for (tuple <- sortedResults) {
      val line = tuple.productIterator.mkString(", ")
      pw.write(line + "\n")
    }
    pw.close()

    val endTime = System.currentTimeMillis() / 1000
    val durationSeconds = endTime - startTime
    println(s"Time: $durationSeconds seconds")
  }

  def constructCharacteristicMatrix(inputRDD: org.apache.spark.rdd.RDD[String]) : (Array[Array[Int]], HashMap[Int,mutSet[Int]], List[Int], List[Int])= {
    // Remove headers and timestamp column, then convert values to integers:
    val cleanRDD = inputRDD
      .map(line => line.split(",").dropRight(1))
      .mapPartitionsWithIndex((idx, iter) => if (idx == 0) iter.drop(1) else iter)
      .map(row => row.map(elem => elem.toInt))

    // Create input HashMap (key = (userID, productID), value = rating):
    val inputHashMap = HashMap.empty[(Int, Int), Int]
    for (row <- cleanRDD.collect()) {
      val key = (row(0), row(1))
      val value = row(2)
      inputHashMap += key -> value
    }
    // Create input HashMap of sets of users for each product (key = productID, value = set of userIDs):
    val productsHashMap = HashMap.empty[Int, mutSet[Int]]
    for ((key,value) <- inputHashMap) {
      val userID = key._1
      val productID = key._2
      if (productsHashMap.contains(productID)) {
        productsHashMap(productID) += userID
      } else {
        productsHashMap += (productID -> mutSet(userID))
      }
    }

    // Create sets of unique users and products & find their lengths:
    val uniqueUsersList = cleanRDD
      .map(row => row(0))
      .collect()
      .toList
      .distinct
      .sorted
    val usersLength = uniqueUsersList.size
    val uniqueProductsList = cleanRDD
      .map(row => row(1))
      .collect()
      .toList
      .distinct
      .sorted
    val productsLength = uniqueProductsList.size
    // Construct empty characteristic matrix:
    val characteristicMatrix = Array.ofDim[Int](usersLength, productsLength)
    // Fill in characteristic matrix:
    for ((key,value) <- inputHashMap) {
      val userID = key._1
      //println(userID)
      val userIndex = uniqueUsersList.indexOf(userID)
      val productID = key._2
      val productIndex = uniqueProductsList.indexOf(productID)
      characteristicMatrix(userIndex)(productIndex) = 1
    }
    (characteristicMatrix, productsHashMap, uniqueProductsList, uniqueUsersList)
  }

  def createSignatureMatrix(characteristicMatrix: Array[Array[Int]], nProducts: Int, nUsers: Int) : Tuple3[Array[Array[Int]],Seq[Int], Seq[Int]] = {
    val mValue = nUsers
    // Set values for a and b:
    //var aValues = Seq.fill(10)(nextInt)
    val aValues = Seq(-1815999779, 1708993772, 345725741, 933671472, -1951422079, -1507771117, 1132938908, -773625781, 908472598, -1553381221)
    val aLength = aValues.size
    //var bValues = Seq.fill(12)(nextInt)
    val bValues = Seq(784363670, 790094779, 1894764623, -695585368, 1838309072, -2064030279, -898179995, 1493790503, -1481517415, -676623121, 348448404, 133119795)
    val bLength = bValues.size
    var listAandB = new ListBuffer[Tuple2[Int, Int]]
    for (i <- 0 until aLength) {
      for (j <- 0 until bLength) {
        val newTuple = (aValues(i), bValues(j))
        listAandB += newTuple
      }
    }
    // Create matrix of hash values:
    val nHashes = aLength * bLength
    val hashValuesMatrix = Array.ofDim[Int](nUsers, nHashes)
    val rowNumbers = List.range(0,nUsers)
    for (i <- 0 until nUsers) {
      for (j <- 0 until nHashes) {
        val a = listAandB(j)._1
        val b = listAandB(j)._2
        val hashedValue = (a*i + b) % mValue
        hashValuesMatrix(i)(j) = hashedValue
      }
    }
    // Create signature matrix:
    val signatureMatrix: Array[Array[Int]] = Array.fill(nHashes, nProducts)(10000)
    for (rowInd <- 0 until nUsers) {
      for (colInd <- 0 until nProducts) {
        if (characteristicMatrix(rowInd)(colInd) == 1) {
          for (hashInd <- 0 until nHashes) {
            if (hashValuesMatrix(rowInd)(hashInd) < signatureMatrix(hashInd)(colInd)) {
              signatureMatrix(hashInd)(colInd) = hashValuesMatrix(rowInd)(hashInd)
            }
          }
        }
      }
    }
    (signatureMatrix, aValues, bValues)
  }

  def generateCandidatePairs(signatureMatrix: Array[Array[Int]], b: Int, r: Int, n: Int, nProducts: Int) : mutSet[Tuple2[Int, Int]] = {
    var candidatePairs = mutSet.empty[Tuple2[Int, Int]]
    for (bandIdx <- 0 until b) {
      val rowIdxs = r*bandIdx until r*(bandIdx+1)
      var startColInd = 0
      while (startColInd < nProducts-1) {
        var checkColInd = startColInd + 1
        while (checkColInd < nProducts) {
          var checkEqual = true
          while (checkEqual == true) {
            var checkRowInd = rowIdxs(0)
            while (checkRowInd <= rowIdxs.last && checkEqual) {
              if (signatureMatrix(checkRowInd)(startColInd) != signatureMatrix(checkRowInd)(checkColInd)) {
                checkEqual = false
              } else {
                  if (checkRowInd == rowIdxs.last) {
                    candidatePairs += Tuple2(startColInd, checkColInd)
                    checkEqual = false
                  }
                checkRowInd += 1
              }
            }
          }
          checkColInd += 1
        }
        startColInd += 1
      }
    }
    candidatePairs
  }

  def verifyCandidatePairs(candidatePairsSet: mutSet[Tuple2[Int, Int]], productsHashMap: HashMap[Int,mutSet[Int]], uniqueProductList: List[Int], uniqueUsersList: List[Int]) : mutSet[Tuple2[Int, Int]] = {
    val verifiedCandidatesSet = mutSet.empty[Tuple2[Int, Int]]
    for (pair <- candidatePairsSet) {
      val colInd1 = pair._1
      val productID1 = uniqueProductList(colInd1)
      val colInd2 = pair._2
      val productID2 = uniqueProductList(colInd2)
      val jaccardSimilarity = calculateJaccardSimilarity(productsHashMap, productID1, productID2)
      if (jaccardSimilarity >= 0.5) {
        verifiedCandidatesSet += Tuple2(productID1, productID2)
      }
    }
    verifiedCandidatesSet
  }

  def calculateJaccardSimilarity(productsHashMap: HashMap[Int,mutSet[Int]], productID1: Int, productID2: Int) : Float = {
    // Calculate Jaccard similarity:
    val userSet1 = productsHashMap(productID1)
    val userSet2 = productsHashMap(productID2)
    val combinedUserSet = userSet1 union userSet2
    val jaccardDen = combinedUserSet.size
    var jaccardNum = 0
    for (userID <- userSet1) {
      if (userSet2.contains(userID)) {
        jaccardNum += 1
      }
    }
    val jaccardSimilarity = jaccardNum.toFloat/jaccardDen
    jaccardSimilarity
  }
}
