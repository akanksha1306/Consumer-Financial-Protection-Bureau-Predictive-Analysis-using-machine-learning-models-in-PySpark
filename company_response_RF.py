from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler

from pyspark.ml.evaluation import BinaryClassificationEvaluator

import pandas as pd

# Read the JSON file 'complaints.json' into a DataFrame named 'raw_complaints'
raw_complaints = spark.read.json('/user/dvaishn2/5560_Complaints_DS/complaints.json')

# Select necessary columns and drop corrupt records
complaint_df = raw_complaints.select('company', 'product', 'company_response').filter(raw_complaints['_corrupt_record'].isNull())

complaint_df = complaint_df.filter(~(isnull(col("company")) | (trim(col("company")) == "")))
complaint_df = complaint_df.filter(~(isnull(col("product")) | (trim(col("product")) == "")))
complaint_df = complaint_df.filter(~(isnull(col("company_response")) | (trim(col("company_response")) == "")))

# Show the first 10 rows of the DataFrame 'complaint_df'
complaint_df.show(10)

from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql.functions import lit
from pyspark.ml import Transformer
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.functions import col, isnull, trim
from pyspark.ml.feature import IndexToString
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.feature import HashingTF


# Load dataset (assuming `complaint_df` is already defined)
df_company_response = complaint_df


# Calculate the frequency of each company
company_frequency = df_company_response.groupBy("company").agg(count("*").alias("frequency"))

# Join the frequency DataFrame with the original DataFrame on the company column
df_response_with_frequency = df_company_response.join(company_frequency, on="company", how="left")

# Show the first few rows of the DataFrame with company frequencies
df_response_with_frequency.show(10)

# Use the frequency column as a feature for modeling
features = ["product", "frequency"] 
target = "company_response"


from pyspark.storagelevel import StorageLevel
 
df_response_with_frequency.persist(StorageLevel.MEMORY_ONLY)
  
# String indexing for target variable
target_indexer = StringIndexer(inputCol="company_response", outputCol="indexed_company_response")

indexer_product = StringIndexer(inputCol="product", outputCol="indexed_product")

df_response_with_frequency = df_response_with_frequency.drop('company')

# Create VectorAssembler to combine the indexed product and hashed company features
assembler = VectorAssembler(inputCols=["indexed_product", "frequency"], outputCol="features")

# Create Random Forest model
rf = RandomForestClassifier(labelCol="indexed_company_response", featuresCol="features")

# Create a pipeline with the VectorAssembler and Random Forest model
pipeline = Pipeline(stages=[indexer_product, target_indexer, assembler, rf])

# Split the data into training and testing sets
train_data, test_data = df_response_with_frequency.randomSplit([0.7, 0.3], seed=42)


evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="indexed_company_response", metricName="accuracy")

# Define parameter grid for hyperparameter tuning
 
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20]) \
    .addGrid(rf.maxDepth, [2, 4, 6]) \
    .addGrid(rf.minInstancesPerNode, [1, 5, 10]) \
    .build()

# Define CrossValidator
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)
                          
#Training the model and Calculating its time
import time

# Start time
start_time = time.time() 

# Fit the cross validator to the training data
cvModel = crossval.fit(train_data)

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

# Make predictions on the test data using the best model
predictions = cvModel.transform(test_data)

# Evaluate model performance
accuracy_rf = evaluator.evaluate(predictions)
print("Accuracy (Random Forest):", accuracy_rf)

# Define the evaluator for precision
precision_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="indexed_company_response", metricName="weightedPrecision")

# Calculate precision
precision_rf = precision_evaluator.evaluate(predictions)
print("Precision (Random Forest):", precision_rf)

# Define the evaluator for recall
recall_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="indexed_company_response", metricName="weightedRecall")

# Calculate recall
recall_rf = recall_evaluator.evaluate(predictions)
print("Recall (Random Forest):", recall_rf)