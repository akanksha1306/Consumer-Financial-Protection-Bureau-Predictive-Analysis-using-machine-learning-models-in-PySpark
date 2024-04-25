from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler

from pyspark.ml.evaluation import BinaryClassificationEvaluator

import pandas as pd

# Read the JSON file 'complaints.json' into a DataFrame named 'raw_complaints'
raw_complaints = spark.read.json('5560_Complaints_DS/complaints.json')

# Rename the column 'company_response' to 'company_response_status'
#raw_complaints = raw_complaints.withColumnRenamed("company_response", "company_response_status")

# Rename the column 'timely' to 'timely_response'
#raw_complaints = raw_complaints.withColumnRenamed("timely", "timely_response")

# Select all columns from 'raw_complaints' DataFrame and limit the result to 100 rows
complaint_df = raw_complaints.select('*')

# Show the first 100 rows of the DataFrame 'complaint_df'
complaint_df.show(100)

complaint_df = complaint_df.drop("_corrupt_record")

complaint_df = complaint_df.rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0]).toDF()

# Create a function to replace blanks with "Not Available"
complaint_df = complaint_df.withColumn("sub_issue", when(complaint_df["sub_issue"] == "", "Not Available").otherwise(complaint_df["sub_issue"]))
complaint_df = complaint_df.withColumn("sub_product", when(complaint_df["sub_product"] == "", "Not Available").otherwise(complaint_df["sub_product"]))


