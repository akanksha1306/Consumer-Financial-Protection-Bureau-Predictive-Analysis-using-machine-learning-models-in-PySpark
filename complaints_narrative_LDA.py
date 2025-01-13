# Import necessary functions and types from PySpark
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from pyspark.sql.functions import year, month, dayofmonth, count, col, when, lit, expr, explode, udf, trim
from pyspark.sql.types import StringType, TimestampType, ArrayType, DoubleType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, RegexTokenizer
from pyspark.ml.clustering import LDA, BisectingKMeans
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
import re

# Set up Spark session
PYSPARK_CLI = True
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# Data Pre-processing
# -------------------

# Read the JSON file 'complaints.json' into a DataFrame named 'raw_complaints'
raw_complaints = spark.read.json('project/complaints.json')

# Select necessary columns and drop corrupt records
df_nlp = raw_complaints.select('complaint_what_happened', 'date_received') \
                             .filter(raw_complaints['_corrupt_record'].isNull()) \
                             .filter(trim(col('complaint_what_happened')) != '') \
                             .filter(col('complaint_what_happened').isNotNull())

# Cast date_received to TimestampType
df_nlp = df_nlp.withColumn("date_received", col("date_received").cast(TimestampType()))

# Extract year, month, and day from date_received
df_nlp = df_nlp.withColumn("year", year("date_received")) \
                     .withColumn("month", month("date_received")) \
                     .withColumn("day", dayofmonth("date_received"))

df_nlp = df_nlp.drop("date_received")

# Define a function to clean text and remove stop words
def clean_text(text):
    # Define regular expression pattern to match "xx" followed by one or more "x"s
    xx_pattern = r'\bxxxx+\b|\bXX+\b|\bxx+\b'
    # Combine the xx_pattern with the existing pattern
    pattern = fr'{xx_pattern}|[^a-zA-Z\s]|(?<![a-zA-Z])\b[a-zA-Z]\b'
    # Replace matched patterns with a space
    cleaned_text = re.sub(pattern, ' ', text.lower())
    # Split text into words and filter out empty strings and words with less than 2 characters
    cleaned_words = [word for word in cleaned_text.split() if len(word) > 1]
    return cleaned_words

# Apply text cleaning function to complaint_what_happened column
clean_text_udf = udf(clean_text, ArrayType(StringType()))
df_cleaned = df_nlp.withColumn("cleaned_complaint", clean_text_udf("complaint_what_happened"))

# Define stop words
stop_words = StopWordsRemover().getStopWords()

# Define and apply stop words remover
stop_words_remover = StopWordsRemover(inputCol="cleaned_complaint", outputCol="cleaned_complaint_without_stopwords", stopWords=stop_words)
df_cleaned = stop_words_remover.transform(df_cleaned)

df_cleaned = df_cleaned.drop("cleaned_complaint", "complaint_what_happened")

# Convert words into numerical features using CountVectorizer and IDF

# Initialize CountVectorizer
cv = CountVectorizer(inputCol="cleaned_complaint_without_stopwords", outputCol="raw_features")
cv_model = cv.fit(df_cleaned)
df_featurized = cv_model.transform(df_cleaned)

# Initialize IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_featurized)
df_features = idf_model.transform(df_featurized)


# Model Training
# --------------

# Train LDA model

num_topics = 25  # adjust this parameter
lda = LDA(k=num_topics, seed=123, optimizer="em", featuresCol="features")

# Measure training time
import time
start_time = time.time()

lda_model = lda.fit(df_features)

end_time = time.time()
training_time = end_time - start_time
minutes = int(training_time // 60)
seconds = int(training_time % 60)
logging.info("Training time: %02d:%02d" % (minutes, seconds))

# Evaluation
# ----------

ldatopics = lda_model.describeTopics()
ldatopics.show(25)

# Extract vocabulary from CountVectorizer model
vocab = cv_model.vocabulary
vocab_broadcast = sc.broadcast(vocab)

# Define function to map term IDs to words
def map_termID_to_Word(termIndices):
    words = []
    for termID in termIndices:
        words.append(vocab_broadcast.value[termID])
    return words

# Apply UDF to map term IDs to words in LDA topics
udf_map_termID_to_Word = udf(map_termID_to_Word , ArrayType(StringType()))
ldatopics_mapped = ldatopics.withColumn("topic_desc", udf_map_termID_to_Word(ldatopics.termIndices))
ldatopics_mapped.select(ldatopics_mapped.topic, ldatopics_mapped.topic_desc).show(50,False)

from pyspark.sql.functions import avg, expr

# Assign topics to each complaint
df_topics = lda_model.transform(df_features)
df_topics.select("month", "day", "topicDistribution").show(20)

logging.info("Training time: %02d:%02d" % (minutes, seconds))
