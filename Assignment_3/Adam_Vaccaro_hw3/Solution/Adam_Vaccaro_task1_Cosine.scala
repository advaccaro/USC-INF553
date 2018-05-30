/** INF 553 - Homework 3: LSH with Cosine Similarity
  * Adam Vaccaro
  */

package com.soundcloud.lsh

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.{Set => mutSet}
import java.io.PrintWriter
import java.io.File

object CosineLSH {

  def main(args: Array[String]) = {
    // Parse inputs:
    val inputPath = args(0)
    val outputPath = args(1)

    // set up Spark Session:
    val numPartitions = 8
    val spark = SparkSession.builder()
      .appName("Spark Local")
      .config("spark.master", "local")
      .getOrCreate()
    val storageLevel = StorageLevel.MEMORY_AND_DISK

    val sc = spark.sparkContext

    // Load in original CSV and construct binary characteristic matrix:
    val (characteristicMatrix, productsHashMap, uniqueProductsList, uniqueUsersList) = constructCharacteristicMatrix(sc.textFile(inputPath))
    val nProducts = characteristicMatrix(0).size

    // transpose and get into desired format:
    val transposed = characteristicMatrix.transpose
    val tempList = ListBuffer.empty[Tuple2[Int, Array[Int]]]
    for (i <- 0 until nProducts) {
      val features = transposed(i)
      val productID = uniqueProductsList(i)
      val tuple = Tuple2(productID, features)
      tempList += tuple
    }
    val data = sc.parallelize(tempList)

    // create an unique id for each product by zipping with the RDD index
    val indexed = data.zipWithIndex.persist(storageLevel)

    // create indexed row matrix where every row represents one product
    val rows = indexed.map {
      case ((word, features), index) =>
        IndexedRow(index, Vectors.dense(features.map(elem => elem.toDouble)))
    }

    // store index for later re-mapping (index to word)
    val index = indexed.map {
      case ((word, features), index) =>
        (index, word)
    }.persist(storageLevel)

    // create an input matrix from all rows and run lsh on it
    val matrix = new IndexedRowMatrix(rows)

    val lsh = new Lsh(
      minCosineSimilarity = 0.5,
      dimensions = 100,
      numNeighbours = 200,
      numPermutations = 10,
      partitions = numPartitions,
      storageLevel = storageLevel
    )

    val similarityMatrix = lsh.join(matrix)

    // remap both ids back to words
    val remapFirst = similarityMatrix.entries.keyBy(_.i).join(index).values

    val remapSecond = remapFirst.keyBy { case (entry, word1) => entry.j }.join(index).values.map {
      case ((entry, word1), word2) =>
        (word1, word2, entry.value)
    }

    val sortedResults = remapSecond
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
    sc.stop()
  }

  def constructCharacteristicMatrix(inputRDD: org.apache.spark.rdd.RDD[String]): (Array[Array[Int]], HashMap[Int, mutSet[Int]], List[Int], List[Int]) = {
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
    for ((key, value) <- inputHashMap) {
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
    for ((key, value) <- inputHashMap) {
      val userID = key._1
      //println(userID)
      val userIndex = uniqueUsersList.indexOf(userID)
      val productID = key._2
      val productIndex = uniqueProductsList.indexOf(productID)
      characteristicMatrix(userIndex)(productIndex) = 1
    }
    (characteristicMatrix, productsHashMap, uniqueProductsList, uniqueUsersList)
  }
}
