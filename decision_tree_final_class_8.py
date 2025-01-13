# -*- coding: utf-8 -*-
"""This Spark code performs classification on a complaints dataset 
using a Desicion tree model trained with CrossValidator for hyperparameter tuning.
 
It calculates feature importances, evaluates model performance 
using various metrics (accuracy, precision, recall), 
and compares the results between CrossValidator and TrainValidationSplit approaches.

Data Loading and Preprocessing:
Feature Engineering:
Feature Selection and Preprocessing:
Model Building:"""

import pandas as pd

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.storagelevel import StorageLevel
from pyspark.sql.functions import lit
from pyspark.ml import Transformer
from pyspark.sql.functions import col
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.mllib.evaluation import MulticlassMetrics

# PYSPARK_CLI = True
# if PYSPARK_CLI:
    # sc = SparkContext.getOrCreate()
    # spark = SparkSession(sc)



#Data Loading and Preprocessing:
#-------------------------------
"""Read the JSON file 'complaints.json' into a DataFrame named 'raw_complaints'.
Select necessary columns ('company', 'product', 'company_response', 'issue').
Filter out corrupt records based on the '_corrupt_record' column.
Remove rows with missing or empty values in 'company', 'product', or 'company_response'."""



# Read the JSON file 'complaints.json' into a DataFrame named 'raw_complaints'
raw_complaints = spark.read.json('/user/dvaishn2/5560_Complaints_DS/complaints.json')

# Select necessary columns and drop corrupt records
complaint_df = raw_complaints.select('company', 'product', 'company_response' , 'issue').filter(raw_complaints['_corrupt_record'].isNull())

complaint_df = complaint_df.filter(~(isnull(col("company")) | (trim(col("company")) == "")))
complaint_df = complaint_df.filter(~(isnull(col("product")) | (trim(col("product")) == "")))
complaint_df = complaint_df.filter(~(isnull(col("company_response")) | (trim(col("company_response")) == "")))

# Show the first 10 rows of the DataFrame 'complaint_df'
complaint_df.show(10)

# Load dataset 
df_company_response = complaint_df




#Feature Engineering:
#----------------------
"""Calculate frequency of each company (company_frequency).
Calculate frequency of each issue (issue_frequency).
Join the frequency DataFrames with the original DataFrame on 'company' and 'issue' columns, respectively."""




# Calculate the frequency of each company
company_frequency = df_company_response.groupBy("company").agg(count("*").alias("frequency_company"))

# Join the frequency DataFrame with the original DataFrame on the company column
df_response_with_frequency = df_company_response.join(company_frequency, on="company", how="left")

# Calculate the frequency of each issue (corrected to avoid duplicate calculation)
issue_frequency = df_company_response.groupBy("issue").agg(count("*").alias("frequency_issue"))

# Join the issue frequency DataFrame with the existing DataFrame on the issue column
df_response_with_frequency = df_response_with_frequency.join(issue_frequency, on="issue", how="left")

# Show the result
df_response_with_frequency.show(10)




#Feature Selection and Preprocessing:
#------------------------------------
"""Define features (product, frequency_company, frequency_issue) and target (company_response).
Perform string indexing for the target variable and product using StringIndexer.
Create a VectorAssembler to combine indexed features into a single feature vector."""




# Use the frequency column as a feature for modeling
features = ["product", "frequency_company", "frequency_issue"] 
target = "company_response"

from pyspark.storagelevel import StorageLevel

df_response_with_frequency.persist(StorageLevel.MEMORY_ONLY)

# String indexing for target variable
target_indexer = StringIndexer(inputCol="company_response", outputCol="indexed_company_response")

indexer_product = StringIndexer(inputCol="product", outputCol="indexed_product" , handleInvalid="skip")

df_response_with_frequency = df_response_with_frequency.drop('company', 'issue')

# Create VectorAssembler to combine the indexed product and hashed company features
assembler = VectorAssembler(inputCols=["indexed_product", "frequency_company", "frequency_issue"], outputCol="features")


# Create Decision Tree model
dt = DecisionTreeClassifier(labelCol="indexed_company_response", featuresCol="features")




# Balancing the data_set: 
#------------------------ 
"""# Balancing the Dataset

To ensure balanced representation of each response type in the dataset, the following steps are performed:

### Define DataFrames for each response type
### Calculate current counts for each response type
- Counts the number of complaints in each category.

### Calculate Oversampling Factors
- Calculates the oversampling/undersampling factor for each category to achieve a target count of 15,000 samples.

### Calculate Sampling Fractions
- Calculates sampling fractions to achieve the target count for each category.

### Sample Each Category
- Samples each category DataFrame with the calculated fraction to achieve the target count.
- The sampling is performed with replacement to balance the dataset.
- Union the sampled DataFrames to create a balanced dataset.

### Display Balanced Dataset
- Shows the count of each category in the balanced data.
- Displays the first 20 rows of the balanced dataset."""



# Define DataFrames for each response type
closed_with_explanation = df_response_with_frequency.filter(df_response_with_frequency["company_response"] == "Closed with explanation")
closed_with_non_monetary_relief = df_response_with_frequency.filter(df_response_with_frequency["company_response"] == "Closed with non-monetary relief")
in_progress = df_response_with_frequency.filter(df_response_with_frequency["company_response"] == "In progress")
closed_with_monetary_relief = df_response_with_frequency.filter(df_response_with_frequency["company_response"] == "Closed with monetary relief")
closed_without_relief = df_response_with_frequency.filter(df_response_with_frequency["company_response"] == "Closed without relief")
closed = df_response_with_frequency.filter(df_response_with_frequency["company_response"] == "Closed")
untimely_response = df_response_with_frequency.filter(df_response_with_frequency["company_response"] == "Untimely response")
closed_with_relief = df_response_with_frequency.filter(df_response_with_frequency["company_response"] == "Closed with relief")


# Calculate current counts for each response type
# Calculate current counts for each response type
counts = {
    "Closed with explanation": closed_with_explanation.count(),
    "Closed with non-monetary relief": closed_with_non_monetary_relief.count(),
    "In progress": in_progress.count(),
    "Closed with monetary relief": closed_with_monetary_relief.count(),
    "Closed without relief": closed_without_relief.count(),
    "Closed": closed.count(),
    "Untimely response": untimely_response.count(),
    "Closed with relief": closed_with_relief.count()
}


# Calculate the oversampling factor for each category to achieve 15000 samples
target_count = 15000
oversampling_factors = {response: target_count / count for response, count in counts.items()}

# Calculate sampling fractions to achieve the target count for each category
sampling_fractions = {category: target_count / count for category, count in counts.items()}

# Create an empty DataFrame to hold the balanced data
balanced_data = spark.createDataFrame([], df_response_with_frequency.schema)

# Sample each category to achieve the target count
for category, count in counts.items():
    # Sample the category DataFrame with the calculated fraction
    sampled_df = df_response_with_frequency.filter(df_response_with_frequency["company_response"] == category)\
                                          .sample(withReplacement=True, fraction=sampling_fractions[category], seed=42)
    # Union the sampled DataFrame with the balanced data
    balanced_data = balanced_data.union(sampled_df)   


# Show the count of each category in the balanced data
balanced_data.groupBy("company_response").count().orderBy("company_response").show()



#pipeline:
#------------

# Create a pipeline with the VectorAssembler and Decision Tree model
pipeline = Pipeline(stages=[indexer_product, target_indexer, assembler, dt])

# Split the data into training and testing sets
train_data, test_data = balanced_data.randomSplit([0.7, 0.3], seed=42)

train_rows = train_data.count()
test_rows = test_data.count()

# Print the counts
print("Training Rows:", train_rows, " Testing Rows:", test_rows)




#Model Building:
#---------------
"""Create a Desiscion Tree model (Decisiontree Classifier).
Define a pipeline comprising the feature preprocessing stages and the model.
Set up a ParamGridBuilder for hyperparameter tuning"""

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="indexed_company_response", metricName="accuracy")


paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [3, 5, 7]) \
    .addGrid(dt.minInstancesPerNode, [1, 5, 10]) \
    .build()
    
    
# paramGrid = ParamGridBuilder() \
    # .addGrid(dt.maxDepth, [3, 5]) \
    # .addGrid(dt.minInstancesPerNode, [1, 5]) \
    # .addGrid(dt.maxBins, [32, 64]) \
    # .addGrid(dt.minInfoGain, [0.0, 0.1]) \
    # .addGrid(dt.impurity, ['gini', 'entropy']) \
    # .build()



# Define CrossValidator
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)




#Model Traning:
# Training the model and calculating its time

import time

# Start time
start_time = time.time() 

# Fit the cross validator to the training data
cvModel = crossval.fit(train_data)

# End time
end_time = time.time()

print("Model trained!")

# Calculate training time
training_time = end_time - start_time

# Calculate minutes and seconds
minutes = int(training_time // 60)
seconds = int(training_time % 60)

# Format the time
training_time_formatted = "{:02d}:{:02d}".format(minutes, seconds)

# Print training time
print("Training time CrossValidator:", training_time_formatted)




# Feature Importance crossvalidation:
#------------------------------------


# Get the fitted model from CrossValidator
bestModel = cvModel.bestModel

# Access the feature importances from the Random Forest model within the pipeline
feature_importances = bestModel.stages[-1].featureImportances  # Assuming Random Forest is the last stage

# Get feature names from the VectorAssembler
feature_names = assembler.getInputCols()

# Create a DataFrame of feature importances
featureImp = pd.DataFrame(list(zip(feature_names, feature_importances)), columns=["feature", "importance"])

# Sort the DataFrame by importance (descending order)
featureImp = featureImp.sort_values(by="importance", ascending=False)

# Print the DataFrame with feature importance
print("\nFeature Importance:")
print(featureImp.round(2).to_string(index=False))



#Test the Data : 
#----------------


# Make predictions on the test data using the best model
predictions = cvModel.transform(test_data)

from pyspark.sql.types import FloatType
#important: need to cast to float type, and order by prediction, else it won't work
preds_and_labels = predictions.select(['prediction','indexed_company_response'])\
                              .withColumn('indexed_company_response', col('indexed_company_response')\
                              .cast(FloatType()))\
                              .orderBy('prediction')
    

from pyspark.mllib.evaluation import MulticlassMetrics
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple)) 



#Result:crossvalidation
#----------------------

confusion_matrix = metrics.confusionMatrix().toArray()

print(metrics.confusionMatrix().toArray())

  
import numpy as np   
            
tps = np.diag(confusion_matrix)
fps = np.sum(confusion_matrix, axis=0) - tps
fns = np.sum(confusion_matrix, axis=1) - tps
tns = np.sum(confusion_matrix) - tps - fps - fns

# Calculate precision, recall, and accuracy for each class
precision = tps / (tps + fps)
recall = tps / (tps + fns)
accuracy = (tps + tns) /(tps + tns + fps + fns)

"""Class 1: "Closed with explanation"
Class 2: "Closed with non-monetary relief"
Class 3: "In progress"
Class 4: "Closed with monetary relief"
Class 5: "Closed without relief"
Class 6: "Closed"
Class 7: "Untimely response"
Class 8: "Closed with relief" """

class_names = {
    1: "Closed with explanation",
    2: "Closed with non-monetary relief",
    3: "In progress",
    4: "Closed with monetary relief",
    5: "Closed without relief",
    6: "Closed",
    7: "Untimely response",
    8: "Closed with relief"
}

for i in range(len(precision)):
    class_name = class_names[i + 1]
    print(f"{class_name}:")
    print(f"Precision: {precision[i]:.2f}")
    print(f"Recall: {recall[i]:.2f}")
    print(f"Accuracy: {accuracy[i]:.2f}")
    print()

confusion_matrix_2d = confusion_matrix.tolist()

# Print the 2D list
print(confusion_matrix_2d)
  



#Train Validation
#------------------    
    
# Define TrainValidationSplit



from pyspark.ml.tuning import TrainValidationSplit
trainval = TrainValidationSplit(estimator=pipeline,
                                 estimatorParamMaps=paramGrid,
                                 evaluator=evaluator,
                                 trainRatio=0.8) 
                                 


    
#Training the model and Calculating its time
import time

# Start time
start_time = time.time() 

# Fit the cross validator to the training data
tvModel = trainval.fit(train_data)

# End time
end_time = time.time()

print("Model trained!")


# Calculate training time
training_time = end_time - start_time

# Calculate minutes and seconds
minutes = int(training_time // 60)
seconds = int(training_time % 60)

# Format the time
training_time_formatted = "{:02d}:{:02d}".format(minutes, seconds)

# Print training time
print("Training time TrainValidator:", training_time_formatted)

# Make predictions on the test data using the best model
predictions = tvModel.transform(test_data)


#Feature Importance:trainvalidation
#----------------------------------

# Get the fitted model from CrossValidator
bestModel = tvModel.bestModel

# Access the feature importances from the Random Forest model within the pipeline
feature_importances_tv = bestModel.stages[-1].featureImportances  # Assuming Random Forest is the last stage

# Get feature names from the VectorAssembler
feature_names_tv = assembler.getInputCols()

# Create a DataFrame of feature importances
featureImp_tv = pd.DataFrame(list(zip(feature_names_tv, feature_importances_tv)), columns=["feature", "importance"])

# Sort the DataFrame by importance (descending order)
featureImp_tv = featureImp_tv.sort_values(by="importance", ascending=False)

# Print the DataFrame with feature importance
print("\nFeature Importance:")
print(featureImp_tv.round(2).to_string(index=False))


#Result:TrainValidationSplit
#----------------------------


from pyspark.sql.types import FloatType
#important: need to cast to float type, and order by prediction, else it won't work
preds_and_labels = predictions.select(['prediction','indexed_company_response'])\
                              .withColumn('indexed_company_response', col('indexed_company_response')\
                              .cast(FloatType()))\
                              .orderBy('prediction')
    


metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple)) 

confusion_matrix = metrics.confusionMatrix().toArray()

print(metrics.confusionMatrix().toArray())

import numpy as np   
            
tps = np.diag(confusion_matrix)
fps = np.sum(confusion_matrix, axis=0) - tps
fns = np.sum(confusion_matrix, axis=1) - tps
tns = np.sum(confusion_matrix) - tps - fps - fns

# Calculate precision, recall, and accuracy for each class
precision = tps / (tps + fps)
recall = tps / (tps + fns)
accuracy = (tps + tns) / np.sum(confusion_matrix)

"""Class 1: "Closed with explanation"
Class 2: "Closed with non-monetary relief"
Class 3: "In progress"
Class 4: "Closed with monetary relief"
Class 5: "Closed without relief"
Class 6: "Closed"
Class 7: "Untimely response"
Class 8: "Closed with relief" """



# Define a dictionary mapping class indices to class names
class_names = {
    1: "Closed with explanation",
    2: "Closed with non-monetary relief",
    3: "In progress",
    4: "Closed with monetary relief",
    5: "Closed without relief",
    6: "Closed",
    7: "Untimely response",
    8: "Closed with relief"
}

for i in range(len(precision)):
    class_name = class_names[i + 1]
    print(f"{class_name}:")
    print(f"Precision: {precision[i]:.2f}")
    print(f"Recall: {recall[i]:.2f}")
    print(f"Accuracy: {accuracy[i]:.2f}")
    print()
    
confusion_matrix_2d = confusion_matrix.tolist()

# Print the 2D list
print(confusion_matrix_2d)
    





    
    
    
