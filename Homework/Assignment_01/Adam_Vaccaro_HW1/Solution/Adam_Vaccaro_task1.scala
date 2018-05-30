/** INF 553 - Homework 1: Task 1
  * Adam Vaccaro
  */

object task1 {
  import java.io.File
  import java.io.PrintWriter
  import org.apache.spark.sql.SparkSession
  def main(args: Array[String]) {
    // set up Spark session:
    val spark = SparkSession.builder()
      .appName("Spark Local")
      .config("spark.master", "local")
      .getOrCreate()
    val sc = spark.sparkContext
    // get filepath and path to output directory:
    val filepath = args(0)
    val outpath = args(1)
    // read raw data into dataframe:
    val df_raw = spark.read.json(filepath)
    // extract variables of interest:
    val df_var = df_raw.select("asin", "overall")
    // group by asin:
    val asinGrouped = df_var.groupBy("asin")
    // calculate average rating:
    val asinReduced = asinGrouped.agg(Map("overall" -> "mean"))
    // format results:
    val asinResult = asinReduced
      .toDF()
      .orderBy("asin")
    val df_results = asinResult
        .withColumnRenamed("avg(overall)", "rating_avg")
      .toDF()
      .coalesce(1)
    val rdd_str = df_results.rdd.map(r => r.mkString(",")).collect()
    // write to csv:
    val pw = new PrintWriter(new File(outpath))
    val nl = sys.props("line.separator")
    pw.write("asin,rating_avg".concat(nl))
    for (r <- rdd_str) {
      pw.write(r.concat(nl))
    }
    pw.close()

    /*df_results
      .write
      .option("header", "true")
     .csv(outpath)*/
  }
}
