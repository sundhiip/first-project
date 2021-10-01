# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## SparkML
# MAGIC In this notebook, we'll use Spark for:
# MAGIC 
# MAGIC * Sentiment Analysis
# MAGIC * Natural Language Processing (NLP)
# MAGIC * Decision Trees
# MAGIC 
# MAGIC We will be using a dataset of roughly 50,000 IMDB reviews, which includes the English language text of that review and the rating associated with it (1-10). Based on the text of the review, we want to predict if the rating is "positive" or "negative".

# COMMAND ----------

# MAGIC %run "./Includes/Classroom Setup"

# COMMAND ----------

reviewsDF = spark.read.parquet("/mnt/training/movie-reviews/imdb/imdb_ratings_50k.parquet")
reviewsDF.createOrReplaceTempView("reviews")
display(reviewsDF)

# COMMAND ----------

# MAGIC %md
# MAGIC What does the distribution of scores look like?
# MAGIC 
# MAGIC HINT: Use `count()`

# COMMAND ----------

# ANSWER

tempDF = spark.sql("SELECT count(rating), rating FROM reviews GROUP BY rating ORDER BY rating")

display(tempDF)

# COMMAND ----------

# MAGIC %md
# MAGIC The authors of this dataset have removed the "neutral" ratings, which they defined as a rating of 5 or 6.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train-Test Split
# MAGIC 
# MAGIC We'll split our data into training and test samples. We will use 80% for training, and the remaining 20% for testing. We set a seed to reproduce the same results (i.e. if you re-run this notebook, you'll get the same results both times).

# COMMAND ----------

(trainDF, testDF) = reviewsDF.randomSplit([0.8, 0.2], seed=42)
trainDF.cache()
testDF.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's determine our baseline accuracy.

# COMMAND ----------

positiveRatings = trainDF.filter("rating >= 5").count()
totalRatings = trainDF.count()

print("Baseline accuracy: {0:.2f}%".format(positiveRatings/totalRatings*100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformers
# MAGIC 
# MAGIC A transformer takes in a DataFrame, and returns a new DataFrame with one or more columns appended to it. They implement a `.transform()` method.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's get started by using <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.RegexTokenizer" target="_blank">RegexTokenizer</a> to convert our review string into a list of tokens.

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer

tokenizer = (RegexTokenizer()
            .setInputCol("review")
            .setOutputCol("tokens")
            .setPattern("\\W+"))

tokenizedDF = tokenizer.transform(reviewsDF)
display(tokenizedDF.limit(5)) # Look at a few tokenized reviews

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC There are a lot of words that do not contain much information about the sentiment of the review (e.g. `the`, `a`, etc.). Let's remove these uninformative words using [StopWordsRemover](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.StopWordsRemover).

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover

remover = (StopWordsRemover()
          .setInputCol("tokens")
          .setOutputCol("stopWordFree"))

removedStopWordsDF = remover.transform(tokenizedDF)
display(removedStopWordsDF.limit(5)) # Look at a few tokenized reviews without stop words

# COMMAND ----------

# MAGIC %md
# MAGIC Where do the stop words actually come from? Spark includes a small English list as a default, which we're implicitly using here.

# COMMAND ----------

stopWords = remover.getStopWords()
stopWords

# COMMAND ----------

# MAGIC %md
# MAGIC Let's remove the `br` from our reviews.

# COMMAND ----------

remover.setStopWords(["br"] + stopWords)
removedStopWordsDF = remover.transform(tokenizedDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Estimators
# MAGIC 
# MAGIC Estimators take in a DataFrame, and return a model (a Transformer). They implement a `.fit()` method.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's apply a <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.CountVectorizer" target="_blank">CountVectorizer</a> model to convert our tokens into a vocabulary.

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer

counts = (CountVectorizer()
          .setInputCol("stopWordFree")
          .setOutputCol("features")
          .setVocabSize(1000))

countModel = counts.fit(removedStopWordsDF) # It's a model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC __Now let's adjust the label (target) values__
# MAGIC 
# MAGIC We want to group the reviews into "positive" or "negative" sentiment. So all of the star "levels" need to be collapsed into one of two groups. To accomplish this, we will use [Binarizer](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.Binarizer).

# COMMAND ----------

from pyspark.ml.feature import Binarizer

binarizer = (Binarizer()
            .setInputCol("rating")
            .setOutputCol("label")
            .setThreshold(5.0))

# COMMAND ----------

# MAGIC %md
# MAGIC Now we are going to use a Decision Tree model to fit to our dataset.

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline
# MAGIC 
# MAGIC Let's put all of these stages into a <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Pipeline" target="_blank">Pipeline</a>. This way, you don't have to remember all of the different steps you applied to the training set, and then apply the same steps to the test dataset. The pipeline takes care of that for you!

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline().setStages([tokenizer, remover, counts, binarizer, dtc])
pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC We can extract the stages from our Pipeline, such as the Decision Tree model.

# COMMAND ----------

decisionTree = pipelineModel.stages[-1]
print(decisionTree.toDebugString)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's save the pipeline model.

# COMMAND ----------

fileName = userhome + "/tmp/DT_Pipeline"
pipelineModel.write().overwrite().save(fileName)

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's load the `PipelineModel` back in.
# MAGIC 
# MAGIC **Note**: You need to know what type of model you're loading in.

# COMMAND ----------

from pyspark.ml import PipelineModel
# Load saved model
savedPipelineModel = PipelineModel.load(fileName)

# COMMAND ----------

resultDF = savedPipelineModel.transform(testDF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Evaluate
# MAGIC 
# MAGIC We are going to use <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.MulticlassClassificationEvaluator" target="_blank">MultiClassClassificationEvaluator</a> to evaluate our predictions (we are using MultiClass because the BinaryClassificationEvaluator does not support accuracy as a metric).

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Accuracy: %(result)s" % {"result": evaluator.evaluate(resultDF)})

# COMMAND ----------

# MAGIC %md
# MAGIC #### Confusion Matrix
# MAGIC 
# MAGIC Let's see if we had more False Positive or False Negatives.

# COMMAND ----------

display(resultDF.groupBy("label", "prediction").count())

# COMMAND ----------

# MAGIC %md
# MAGIC In the next notebook, we will see how to apply this pipeline to streaming data!

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2019 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>