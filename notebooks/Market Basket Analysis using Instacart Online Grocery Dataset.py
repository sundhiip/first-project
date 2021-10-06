# Databricks notebook source
# MAGIC %md
# MAGIC # Market Basket Analysis using Instacart Online Grocery Dataset
# MAGIC ## Which products will an Instacart consumer purchase again?
# MAGIC 
# MAGIC To showcase the concept of market basket analysis with the [Databricks Unified Analytics Platform](https://databricks.com/product/unified-analytics-platform), we will use the Instacart's [3 Million Instacart Orders, Open Sourced](https://www.instacart.com/datasets/grocery-shopping-2017) dataset.
# MAGIC 
# MAGIC > “The Instacart Online Grocery Shopping Dataset 2017”, Accessed from https://www.instacart.com/datasets/grocery-shopping-2017 on 01/17/2018. This anonymized dataset contains a sample of over 3 million grocery orders from more than 200,000 Instacart users.
# MAGIC For each user, we provide between 4 and 100 of their orders, with the sequence of products purchased in each order. We also provide the week and hour of day the order was placed, and a relative measure of time between orders.
# MAGIC 
# MAGIC Whether you shop from meticulously planned grocery lists or let whimsy guide your grazing, our unique food rituals define who we are. Instacart's grocery ordering and delivery app aims to make it easy to fill your refrigerator and pantry with your personal favorites and staples when you need them. After selecting products through the Instacart app, personal shoppers review your order and do the in-store shopping and delivery for you.
# MAGIC 
# MAGIC 
# MAGIC This notebook will show how you can predict which items a shopper will purchase whether they buy it again or recommend to try for the first time.
# MAGIC <img src="https://s3.us-east-2.amazonaws.com/databricks-dennylee/media/buy+it+again+or+recommend.png" width="1100"/>
# MAGIC 
# MAGIC 
# MAGIC *Source: [3 Million Instacart Orders, Open Sourced](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2)*

# COMMAND ----------

# DBTITLE 1,Data Engineering Pipeline
# MAGIC %md
# MAGIC ![](https://s3.us-east-2.amazonaws.com/databricks-dennylee/media/data-engineering-pipeline-3.png)
# MAGIC 
# MAGIC Data engineering pipelines are commonly comprised of these components:
# MAGIC 
# MAGIC * **Ingest Data**: Bringing in the data from your source systems; often involving ETL processes (though we will skip this step in this demo for brevity)
# MAGIC * **Explore Data**: Now that you have cleansed data, explore it so you can get some business insight
# MAGIC * **Train ML Model**: Execute FP-growth for frequent pattern mining
# MAGIC * **Review Association Rules**: Review the generated association rules

# COMMAND ----------

# MAGIC %md # Ingest Data

# COMMAND ----------

# DBTITLE 0,Ingest Data
# MAGIC %md 
# MAGIC First, download the [3 Million Instacart Orders, Open Sourced](https://www.instacart.com/datasets/grocery-shopping-2017) and upload it to `dbfs`; for more information, refer to [Importing Data](https://docs.databricks.com/user-guide/importing-data.html).
# MAGIC 
# MAGIC The following `dbutils filesystem (fs)` query displays the six files:
# MAGIC * `Orders`: 3.4M rows, 206K users
# MAGIC * `Products`: 50K rows
# MAGIC * `Aisles`: 134 rows 
# MAGIC * `Departments`: 21 rows
# MAGIC * `order_products__SET`: 30M+ rows where `SET` is defined as:
# MAGIC   * `prior`: 3.2M previous orders
# MAGIC   * `train`: 131K orders for your training dataset
# MAGIC   
# MAGIC Reference: [The Instacart Online Grocery Shopping Dataset 2017 Data Descriptions](https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b)
# MAGIC 
# MAGIC ### Important
# MAGIC You will need to **edit** the locations (in the examples below, we're using `/mnt/bhavin/mba/instacard/csv`) to where you had uploaded your data.

# COMMAND ----------

# DBTITLE 1,Review Ingested Files
# MAGIC %fs ls /mnt/bhavin/mba/instacart/csv

# COMMAND ----------

# DBTITLE 1,Review `orders.csv` file
# MAGIC %fs head dbfs:/mnt/bhavin/mba/instacart/csv/orders.csv

# COMMAND ----------

# DBTITLE 1,Create DataFrames
# Import Data
aisles = spark.read.csv("/mnt/bhavin/mba/instacart/csv/aisles.csv", header=True, inferSchema=True)
departments = spark.read.csv("/mnt/bhavin/mba/instacart/csv/departments.csv", header=True, inferSchema=True)
order_products_prior = spark.read.csv("/mnt/bhavin/mba/instacart/csv/order_products__prior.csv", header=True, inferSchema=True)
order_products_train = spark.read.csv("/mnt/bhavin/mba/instacart/csv/order_products__train.csv", header=True, inferSchema=True)
orders = spark.read.csv("/mnt/bhavin/mba/instacart/csv/orders.csv", header=True, inferSchema=True)
products = spark.read.csv("/mnt/bhavin/mba/instacart/csv/products.csv", header=True, inferSchema=True)

# Create Temporary Tables
aisles.createOrReplaceTempView("aisles")
departments.createOrReplaceTempView("departments")
order_products_prior.createOrReplaceTempView("order_products_prior")
order_products_train.createOrReplaceTempView("order_products_train")
orders.createOrReplaceTempView("orders")
products.createOrReplaceTempView("products")

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploratory Data Analysis
# MAGIC 
# MAGIC Explore your Instacart data using Spark SQL

# COMMAND ----------

# DBTITLE 1,Busiest day of the week
# MAGIC %sql
# MAGIC select 
# MAGIC   count(order_id) as total_orders, 
# MAGIC   (case 
# MAGIC      when order_dow = '0' then 'Sunday'
# MAGIC      when order_dow = '1' then 'Monday'
# MAGIC      when order_dow = '2' then 'Tuesday'
# MAGIC      when order_dow = '3' then 'Wednesday'
# MAGIC      when order_dow = '4' then 'Thursday'
# MAGIC      when order_dow = '5' then 'Friday'
# MAGIC      when order_dow = '6' then 'Saturday'              
# MAGIC    end) as day_of_week 
# MAGIC   from orders  
# MAGIC  group by order_dow 
# MAGIC  order by total_orders desc

# COMMAND ----------

# DBTITLE 1,Breakdown of Orders by Hour of the Day
# MAGIC %sql
# MAGIC select 
# MAGIC   count(order_id) as total_orders, 
# MAGIC   order_hour_of_day as hour 
# MAGIC   from orders 
# MAGIC  group by order_hour_of_day 
# MAGIC  order by order_hour_of_day

# COMMAND ----------

# DBTITLE 1,Max Products by Department
# MAGIC %sql
# MAGIC select countbydept.*
# MAGIC   from (
# MAGIC   -- from product table, let's count number of records per dept
# MAGIC   -- and then sort it by count (highest to lowest) 
# MAGIC   select department_id, count(1) as counter
# MAGIC     from products
# MAGIC    group by department_id
# MAGIC    order by counter asc 
# MAGIC   ) as maxcount
# MAGIC inner join (
# MAGIC   -- let's repeat the exercise, but this time let's join
# MAGIC   -- products and departments tables to get a full list of dept and 
# MAGIC   -- prod count
# MAGIC   select
# MAGIC     d.department_id,
# MAGIC     d.department,
# MAGIC     count(1) as products
# MAGIC     from departments d
# MAGIC       inner join products p
# MAGIC          on p.department_id = d.department_id
# MAGIC    group by d.department_id, d.department 
# MAGIC    order by products desc
# MAGIC   ) countbydept 
# MAGIC   -- combine the two queries's results by matching the product count
# MAGIC   on countbydept.products = maxcount.counter

# COMMAND ----------

# DBTITLE 1,Top 10 Popular Items
# MAGIC %sql
# MAGIC select count(opp.order_id) as orders, p.product_name as popular_product
# MAGIC   from order_products_prior opp, products p
# MAGIC  where p.product_id = opp.product_id 
# MAGIC  group by popular_product 
# MAGIC  order by orders desc 
# MAGIC  limit 10

# COMMAND ----------

# DBTITLE 1,Shelf Space by Department
# MAGIC %sql
# MAGIC select d.department, count(distinct p.product_id) as products
# MAGIC   from products p
# MAGIC     inner join departments d
# MAGIC       on d.department_id = p.department_id
# MAGIC  group by d.department
# MAGIC  order by products desc
# MAGIC  limit 10

# COMMAND ----------

# MAGIC %md ## Organize and View Shopping Basket

# COMMAND ----------

# DBTITLE 1,Organize Shopping Basket
# Organize the data by shopping basket
from pyspark.sql.functions import collect_set, col, count
rawData = spark.sql("select p.product_name, o.order_id from products p inner join order_products_train o where o.product_id = p.product_id")
baskets = rawData.groupBy('order_id').agg(collect_set('product_name').alias('items'))
baskets.createOrReplaceTempView('baskets')

# COMMAND ----------

# DBTITLE 1,View Shopping Basket
display(baskets)

# COMMAND ----------

# MAGIC %md # Train ML Model 
# MAGIC 
# MAGIC To understand the frequency of items are associated with each other (e.g. peanut butter and jelly), we will use association rule mining for market basket analysis.  [Spark MLlib](http://spark.apache.org/docs/latest/mllib-guide.html) implements two algorithms related to frequency pattern mining (FPM): [FP-growth](https://spark.apache.org/docs/latest/mllib-frequent-pattern-mining.html#fp-growth) and [PrefixSpan](https://spark.apache.org/docs/latest/mllib-frequent-pattern-mining.html#prefixspan). The distinction is that FP-growth does not use order information in the itemsets, if any, while PrefixSpan is designed for sequential pattern mining where the itemsets are ordered. We will use FP-growth as the order information is not important for this use case.
# MAGIC 
# MAGIC > Note, we will be using the `Scala API` so we can configure `setMinConfidence`.

# COMMAND ----------

# DBTITLE 1,Use FP-growth
# MAGIC %scala
# MAGIC import org.apache.spark.ml.fpm.FPGrowth
# MAGIC 
# MAGIC // Extract out the items 
# MAGIC val baskets_ds = spark.sql("select items from baskets").as[Array[String]].toDF("items")
# MAGIC 
# MAGIC // Use FPGrowth
# MAGIC val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(0.001).setMinConfidence(0)
# MAGIC val model = fpgrowth.fit(baskets_ds)

# COMMAND ----------

# DBTITLE 1,Most Frequent Itemsets
# MAGIC %scala
# MAGIC // Display frequent itemsets
# MAGIC val mostPopularItemInABasket = model.freqItemsets
# MAGIC mostPopularItemInABasket.createOrReplaceTempView("mostPopularItemInABasket")

# COMMAND ----------

# MAGIC %sql
# MAGIC select items, freq from mostPopularItemInABasket where size(items) > 2 order by freq desc limit 20

# COMMAND ----------

# MAGIC %md # Review Association Rules
# MAGIC 
# MAGIC In addition to `freqItemSets`, the `FP-growth` model also generates `association rules`.  For example, if a shopper purchases *peanut butter* , what is the likelihood that they will also purchase *jelly*.  For more information, a good reference is Susan Li's [A Gentle Introduction on Market Basket Analysis — Association Rules](https://towardsdatascience.com/a-gentle-introduction-on-market-basket-analysis-association-rules-fa4b986a40ce)

# COMMAND ----------

# DBTITLE 1,View Generated Association Rules
# MAGIC %scala
# MAGIC // Display generated association rules.
# MAGIC val ifThen = model.associationRules
# MAGIC ifThen.createOrReplaceTempView("ifThen")

# COMMAND ----------

# MAGIC %sql
# MAGIC select antecedent as `antecedent (if)`, consequent as `consequent (then)`, confidence from ifThen order by confidence desc limit 20