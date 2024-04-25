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


########## 
def has_blank_values(df, col_name):
    """Checks if a specific column in a PySpark DataFrame has any blank values.

    Args:
        df (pyspark.sql.DataFrame): The DataFrame to check.
        col_name (str): The name of the column to examine.

    Returns:
        bool: True if the column has at least one blank value, False otherwise.
    """

    return df.filter(df[col_name].isEmpty()).count() > 0

# Example usage with single column
if has_blank_values(complaint_df, "date_recieved"):
    print("The 'my_column' column has blank values.")
else:
    print("The 'my_column' column does not have blank values.")
