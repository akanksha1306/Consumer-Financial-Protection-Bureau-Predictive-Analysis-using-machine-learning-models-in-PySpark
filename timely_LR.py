from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StringType
from pyspark.sql.functions import year, month, dayofmonth
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col

from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler

from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

import pandas as pd

PYSPARK_CLI = True
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

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

# drop 1 row having timely = None and Create a dataframe for prediction of timely_response
df_timely = complaint_df.filter(col("timely") != "")

# Cast date_sent_to_company to a suitable type 'timestamp'
df_timely = df_timely.withColumn("date_sent_to_company", col("date_sent_to_company").cast(TimestampType()))

# Extracting year, month, and day from 'date_sent_to_company' column
df_timely = df_timely.withColumn("year", year("date_sent_to_company")) \
                     .withColumn("month", month("date_sent_to_company")) \
                     .withColumn("day", dayofmonth("date_sent_to_company"))

# Define features_for_model directly with data types
features_for_model = ["company", "product", "issue", "state"]

# Create a list of stages for the pipeline
stages = []

# Stage 3: String indexing for categorical features
for col_name in features_for_model[:4]:
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index", handleInvalid="keep")
    stages.append(indexer)

# Stage 4: Assemble features
assembler = VectorAssembler(inputCols=[f"{col_name}_index" for col_name in features_for_model[:4]] + ["year", "month", "day"], outputCol="assembledFeatures")
stages.append(assembler)

# Stage 5: String indexing for label
label_indexer = StringIndexer(inputCol="timely", outputCol="label")
stages.append(label_indexer)

# Stage 4: Logistic Regression model
lr = LogisticRegression(featuresCol="assembledFeatures", labelCol="label")
stages.append(lr)

# Oversample the minority class (assuming "No" is the minority)
negative_df = df_timely.filter(col("timely") == "No")

# Calculate the fraction to achieve a more balanced ratio
# For example, if you want a 1:1 ratio, set fraction = number of "Yes" instances / number of "No" instances
balanced_ratio = df_timely.filter(col("timely") == "Yes").count() / negative_df.count()
oversampled_negative_df = negative_df.sample(withReplacement=True, fraction=balanced_ratio)

# Combine oversampled negatives with original data (assuming positive is the majority)
df_timely_balanced = df_timely.filter(col("timely") == "Yes").union(oversampled_negative_df)


# Split data into training and testing sets
train, test = df_timely_balanced.randomSplit([0.7, 0.3], seed=42)

# Combine stages into a pipeline
pipeline = Pipeline(stages=stages)

# Define evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# Define paramGrid
#paramGrid = ParamGridBuilder() \
#  .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
#  .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
#  .build()

from pyspark.ml.tuning import RandomParamGridBuilder

paramGrid = RandomParamGridBuilder() \
  .addGrid(lr.regParam, [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]) \
  .addGrid(lr.elasticNetParam, [0.0, 0.25, .5, 0.75, 1.0]) \
  .build()
  

# Create a CrossValidator
cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=5)

import time
# **Measure training time on the actual training set**
# Start time
start_time = time.time()

# Fit the model with cross-validation on the training set
model = cv.fit(train)

# End time
end_time = time.time()

# Calculate training time
training_time = end_time - start_time


# Calculate minutes and seconds
minutes = int(training_time // 60)
seconds = int(training_time % 60)

# Format the time
training_time_formatted = "{:02d}:{:02d}".format(minutes, seconds)

# Print training time
print("Training time:", training_time_formatted)

# Make predictions on the test set (use the actual test set)
predictions = model.transform(test)

predicted = predictions.select("assembledFeatures", "prediction", "probability", "label")

predicted.show(100, truncate=False)

tp = float(predicted.filter("prediction == 1.0 AND label == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND label == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND label == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND label == 1").count())

auc = evaluator.evaluate(predictions)

metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn)),
 ("AUC", auc)],["metric", "value"])

metrics.show()