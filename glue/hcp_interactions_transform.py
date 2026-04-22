import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.window import Window

args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME", "input_path", "output_path"]
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

input_path = args["input_path"]
output_path = args["output_path"]

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .csv(input_path)
)

cleaned_type = (
    F.when(F.lower(F.trim(F.col("interaction_type"))).isin("call", "phone call"), F.lit("Call"))
     .when(F.lower(F.trim(F.col("interaction_type"))).isin("email", "e-mail"), F.lit("Email"))
     .when(F.lower(F.trim(F.col("interaction_type"))).isin("meeting", "in-person meeting"), F.lit("Meeting"))
     .otherwise(F.lit("Other"))
)

df_clean = (
    df.withColumn("interaction_time", F.to_timestamp("interaction_ts"))
      .withColumn("ingestion_time", F.to_timestamp("ingest_ts"))
      .withColumn("cleaned_interaction_type", cleaned_type)
)

dup_window = Window.partitionBy(
    "hcp_id",
    "rep_id",
    "interaction_type",
    "interaction_ts"
)

df_clean = df_clean.withColumn(
    "is_duplicate_flag",
    F.when(F.count(F.lit(1)).over(dup_window) > 1, F.lit(1)).otherwise(F.lit(0))
)

df_clean = (
    df_clean.withColumn("etl_loaded_at", F.current_timestamp())
            .withColumn("source_file_path", F.input_file_name())
)

(
    df_clean.write
    .mode("overwrite")
    .parquet(output_path)
)

job.commit()

