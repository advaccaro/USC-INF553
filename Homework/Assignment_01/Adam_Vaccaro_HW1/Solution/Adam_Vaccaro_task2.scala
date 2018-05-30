import java.io.{File, PrintWriter}

/** INF 553 - Homework 1: Task 2
  * Adam Vaccaro
  */

object task2 {
  import org.apache.spark.sql.SparkSession
  import java.io.File
  import java.io.PrintWriter
  def main(args: Array[String]) {
    // set up Spark session:
    val spark = SparkSession.builder()
      .appName("Spark Local")
      .config("spark.master", "local")
      .getOrCreate()
    val sc = spark.sparkContext
    // get filepath and path to output directory:
    val reviewspath = args(0)
    val metapath = args(1)
    val outpath = args(2)
    // read raw data into dataframe:
    val df_reviews = spark.read.json(reviewspath)
    val df_meta = spark.read.json(metapath)
    // extract relevant data:
    val df_brand = df_meta.select("asin", "brand").where("brand != 'null' AND brand != ''")
    val df_rev = df_reviews.select("asin", "overall")
    // join on asin:
    val df_joined = df_rev.join(df_brand, Seq("asin"))
    // group by brand:
    val brandGrouped = df_joined.groupBy("brand")
    // calculate average rating:
    val brandReduced = brandGrouped.agg(Map("overall" -> "mean"))
    // format results:
    val brandResult = brandReduced
      .toDF()
      .orderBy("brand")
    val renamedResult = brandResult
      .withColumnRenamed("avg(overall)", "rating_avg")
      .toDF().coalesce(1)
    // write to csv:
    renamedResult
      .write
      .format("csv")
      .option("header", "true")
      .save(outpath)
  }
}

