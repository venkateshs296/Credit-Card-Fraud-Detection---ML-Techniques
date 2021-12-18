# Databricks notebook source
# MAGIC %md
# MAGIC # Project: Fraud Detection in Financial Transactions using Machine Learning Techniques in Spark
# MAGIC ## Code: Custom Decision Tree in Spark
# MAGIC ## Team Members:
# MAGIC   1. Manasa M Bhat ( mmb190005 )
# MAGIC   2. Manneyaa Jayasanker ( mxj180040 )
# MAGIC   3. Swathi Poseti ( sxp190117 )
# MAGIC   4. Venkatesh Sankar ( vxs200014 )

# COMMAND ----------

# MAGIC %md
# MAGIC ### ReadMe Section
# MAGIC The following dependencies must be installed in the Databricks cluster before executing the code
# MAGIC 1. networkx
# MAGIC    - Click on Libraries tab in the Cluster. 
# MAGIC    - Click on 'Install New' button
# MAGIC    - Select Library Source as Upload and Library Type as Python Whl
# MAGIC    - Download the .whl file from this [link](https://pypi.org/project/networkx/#files)
# MAGIC    - Upload the .whl file and click on Install
# MAGIC 2. pygraphviz 
# MAGIC    - Run the Command 2 and 3 in the notebook as shown below. For more details refer [databricks installation guide](https://kb.databricks.com/libraries/install-pygraphviz.html)
# MAGIC    
# MAGIC The input file is hosted in UTD Box and the link is used below.
# MAGIC Once the above installations are done. The notebook can be imported in Databricks and executed.

# COMMAND ----------

# MAGIC %sh sudo apt-get install -y python3-dev graphviz libgraphviz-dev pkg-config

# COMMAND ----------

# MAGIC %sh pip install pygraphviz

# COMMAND ----------

import networkx as nx
import math
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import QuantileDiscretizer, Bucketizer
from pyspark.sql.functions import col
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import copy
from pyspark import SparkFiles

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Data from data source file in UTD Box
# MAGIC * Source Dataset link: https://www.kaggle.com/ealaxi/paysim1
# MAGIC * UTD Box link for the same dataset: https://utdallas.box.com/shared/static/xu02vmnwqctry6187lg7pnhm0dvj7l3r.csv

# COMMAND ----------

fraud_data_url = "https://utdallas.box.com/shared/static/xu02vmnwqctry6187lg7pnhm0dvj7l3r.csv"

spark.sparkContext.addFile(fraud_data_url)

fraud_dataset = spark.read.csv("file://"+SparkFiles.get("xu02vmnwqctry6187lg7pnhm0dvj7l3r.csv"), header=True, inferSchema= True)

# COMMAND ----------

#fraud_dataset = spark.read.option("header","true").csv("/FileStore/tables/FraudDataset.csv")
double_attr = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
for attr in double_attr:
  fraud_dataset = fraud_dataset.withColumn(attr, fraud_dataset[attr].cast(DoubleType()))
int_attr = ["isFraud"]
for attr in int_attr:
  fraud_dataset = fraud_dataset.withColumn(attr, fraud_dataset[attr].cast(IntegerType()))

# COMMAND ----------

data = fraud_dataset.select(["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest","isFraud"]) 
data.cache()
data.printSchema()

# COMMAND ----------

display(data)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Split data into train and test sets

# COMMAND ----------

# Initially split our dataset between training and test datasets
(train_data, test_data) = data.randomSplit([0.8, 0.2], seed=13241)

# Cache the training and test datasets
train_data.cache()
test_data.cache()

# Print out dataset counts
print("Total rows: %s, Training rows: %s, Test rows: %s" % (data.count(), train_data.count(), test_data.count()))

# COMMAND ----------

print("Fraudulent transactions in train data:", train_data.filter(F.col("isFraud") == 1).count())
print("Fraudulent transactions in test data:", test_data.filter(F.col("isFraud") == 1).count())

# COMMAND ----------

attr_names = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
discrete_attr_list = ["type"]
# number of buckets for continuous parameters
bucket_count = 10
depth_limit = 12
nodes_limit = 400

# COMMAND ----------

#Decision tree representation initialization
decision_tree = nx.DiGraph()

# COMMAND ----------

def calculateEntropyNew(x,y):
  total_count = x + y
  px = x/total_count
  py = y/total_count
  res = round(px * math.log2(px) * -1,5) + round(py * math.log2(py) * -1,5)
  return res * total_count

def conditional(x, total_count):
  if type(x[1]) == int:
    return 0.0
  else:
    return x[1]/total_count
  
def entropyTerm(x, total_count):
  p = x[1]/total_count
  res = round(p * math.log2(p) * -1,5) #optional rounding, remove if not needed
  return (x[0],res)

def calculate_entropy(counts_data,total_count):
  if counts_data.count() == 1:
    return 0.0
  else:
    return counts_data.rdd.map(lambda x: entropyTerm(x,total_count)).reduce(lambda x,y: x[1]+y[1])

# COMMAND ----------

def calculate_info_gain(data, used_attr):
    attr_info_gain = dict()
    # Get the data count for each class: fraud and non-fraud
    counts_data = data.select(["isFraud"]).groupBy("isFraud").count()
    total_count = counts_data.rdd.reduce(lambda x,y: x[1]+y[1])
    entropy = calculate_entropy(counts_data,total_count)
    
    for attr in attr_names:
      if attr not in used_attr:
        if attr not in discrete_attr_list:
          buckets_data = find_buckets(data,attr)
          attr_entropy = buckets_data.groupBy("buckets", "isFraud").count().select(["buckets","count"]).rdd.reduceByKey(calculateEntropyNew).map(lambda x: conditional(x,total_count)).reduce(lambda x,y: x+y)
          attr_info_gain[attr] = entropy - attr_entropy
        else:
          attr_entropy = data.groupBy(attr, "isFraud").count().select([attr,"count"]).rdd.reduceByKey(calculateEntropyNew).map(lambda x: conditional(x,total_count)).reduce(lambda x,y: x+y)
          attr_info_gain[attr] = entropy - attr_entropy
#     for key in attr_info_gain:
#       print(attr_info_gain[key])
    return attr_info_gain

# COMMAND ----------

def find_buckets(data,attr):
  qds = QuantileDiscretizer(numBuckets=bucket_count, inputCol=attr, outputCol="buckets", relativeError=0.01, handleInvalid="error")
  bucketizer = qds.fit(data)
  return bucketizer.setHandleInvalid("skip").transform(data)

# COMMAND ----------

def find_max_gain_attr(info_gain_dict):
  return max(info_gain_dict, key=info_gain_dict.get)

# COMMAND ----------

def build_tree(train_data, used_attr, distinct_attr_val):
    count_classes = train_data.select(["isFraud"]).groupBy("isFraud").count()
    
    #Leaf 1: if there is only one class in the data, it is a leaf
    #Leaf 2: if all attributes have been used in the decision tree, it is a leaf 
    #        find the class with majority count and make that value as the leaf node value
    if count_classes.count() == 1 or len(used_attr) >= depth_limit or decision_tree.order() > nodes_limit:
        class_val = count_classes.rdd.max(lambda x: x[1])[0]
        decision_tree.add_edges_from([(distinct_attr_val,class_val)])
        return
    else:
        used_attr_cp = copy.deepcopy(used_attr)
        # get attribute with highest gain
        info_gain_dict = calculate_info_gain(train_data,used_attr)
        if bool(info_gain_dict):
          best_attr = find_max_gain_attr(info_gain_dict)
        else:
          best_attr = -1
        #print("best_attr "+ str(best_attr))  
        #Leaf 3: if no more attributes to be found or no best attribute - not sure if this will occur
        if best_attr == -1:
          class_val = count_classes.rdd.max(lambda x: x[1])[0]
          decision_tree.add_edges_from([(distinct_attr_val,class_val)])
          return
        
        used_attr_cp.append(best_attr)
       
        #Get unique values or all possible range for best_attr
        #attr_vals = get_unique_values(train_data,best_attr)
    # split to multiple sets based on values - use where condition here - no need to pass to other build_Tree calls
        if distinct_attr_val != '': #not root call
          decision_tree.add_edges_from([(distinct_attr_val,best_attr)])
          
        if best_attr not in discrete_attr_list:
          buckets_data = find_buckets(train_data,best_attr)
          i = 0
          for i in range(bucket_count):
            data_i = buckets_data.filter(col("buckets") == i).select(list(set(buckets_data.columns) - {'buckets'}))
            data_i.cache()
            if data_i.count() == 0:
              continue
            row = data_i.select([best_attr]).agg(F.min(data_i[best_attr]),F.max(data_i[best_attr])).rdd.collect()
            attr_val = str(row[0][0]) +" to "+ str(row[0][1])
            decision_tree.add_edges_from([(best_attr,attr_val)])
            build_tree(data_i,used_attr_cp,attr_val)
        else:
          attr_vals = train_data.select([best_attr]).distinct().rdd.collect()
          for attr_val in attr_vals:
            data_i = train_data.filter(train_data[best_attr] == attr_val[0])
            data_i.cache()
            decision_tree.add_edges_from([(best_attr,attr_val[0])])
            build_tree(data_i,used_attr_cp,attr_val[0])
            
    return


# COMMAND ----------

decision_tree = nx.DiGraph()
print("Starting build tree...")
print("with parameters: ")
print(" no. of buckets for continuous variables: ", bucket_count)
print(" nodes limit: ", nodes_limit)
print(" depth limit: ", depth_limit)
build_tree(train_data, [], '')
print("Decision tree has been built.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experiments
# MAGIC 1. Vary the hyper parameters: buckets for continuous variables, depth and number of nodes and find the impact on decision tree creation
# MAGIC 2. Split train and test data into different ratios and find the metrics: accuracy, precision, recall, F1score, confusion matrix and area under ROC.
# MAGIC 3. Comparison between custom decision tree implementation and MLLib classifiers
# MAGIC 4. Feature ranking based on information gain

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC ### Experiment 1. Vary the hyper parameters: buckets for continuous variables, depth and number of nodes and find the impact on decision tree creation

# COMMAND ----------

#bucket = 10, nodes <= 400 , max nodes depth = 12, 80:20 split
from networkx.drawing.nx_agraph import graphviz_layout
fig = plt.gcf()
fig.set_size_inches(50, 50)
pos = graphviz_layout(decision_tree, prog="dot")
nx.draw(decision_tree,pos,with_labels=True,arrows=False)

# COMMAND ----------

#bucket = 5, nodes <= 200 , max nodes depth = 12, 90:10 split
from networkx.drawing.nx_agraph import graphviz_layout
fig = plt.gcf()
fig.set_size_inches(50, 50)
pos = graphviz_layout(decision_tree, prog="dot")
nx.draw(decision_tree,pos,with_labels=True,arrows=False)

# COMMAND ----------

#bucket = 5, nodes <= 50 , max nodes depth = 6, 90:10 split
from networkx.drawing.nx_agraph import graphviz_layout
fig = plt.gcf()
fig.set_size_inches(50, 50)
pos = graphviz_layout(decision_tree, prog="dot")
nx.draw(decision_tree,pos,with_labels=True,arrows=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Calculate predictions for Fraud detection

# COMMAND ----------

tuple_order=["type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]
root = [n for n,d in decision_tree.in_degree() if d==0][0]

def prediction(x, decision_tree):
  if len(x) == 7:
    cur_node = root
  else:
    cur_node = x[7]
  
  if cur_node != 0 and cur_node != 1:
    idx = tuple_order.index(cur_node)
    if cur_node in discrete_attr_list:
      ngbr_list = [n for n in decision_tree.neighbors(cur_node)]
      for val in ngbr_list:
        if val == x[idx]:
          cur_node = [n for n in decision_tree.neighbors(val)][0]
          break
    else:
      ngbr_list = [n for n in decision_tree.neighbors(cur_node)]
      for val in ngbr_list:
        ranges = val.split(" ")
        if x[idx] >= float(ranges[0].strip()) and x[idx] <= float(ranges[2].strip()):
          cur_node = [n for n in decision_tree.neighbors(val)][0]
          break
  if cur_node == 0 or cur_node == 1:
    cur_node = float(cur_node)
  return (x[0],x[1],x[2],x[3],x[4],x[5],x[6],cur_node)


# COMMAND ----------

def compute_prediction(data, decision_tree):
  total_itr = 2 * depth_limit
  data_rdd = data.rdd
  for i in range(total_itr):
    data_rdd = data_rdd.map(lambda x: prediction(x,decision_tree))
  
  pred = data_rdd.toDF(["type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","isFraud","predictedFraud"])
  return pred.filter(F.col("predictedFraud").isNotNull())

# COMMAND ----------

test_pred = compute_prediction(test_data,decision_tree)

# COMMAND ----------

display(test_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Metrics 1: Area under ROC

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluatorROC = BinaryClassificationEvaluator()

# COMMAND ----------

#train_roc = evaluatorROC.evaluate(train_pred)
test_roc = evaluatorROC.evaluate(test_pred,{evaluatorROC.metricName: 'areaUnderROC', evaluatorROC.labelCol : 'isFraud', evaluatorROC.rawPredictionCol : 'predictedFraud'})

#print("Area under ROC for train data: ", train_roc)
print("Area under ROC for test data:", round(test_roc,5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Metrics 2: Confusion Matrix

# COMMAND ----------

from pyspark.mllib.evaluation import MulticlassMetrics

predictionAndLabels = test_pred.select(["predictedFraud", "isFraud"]).withColumnRenamed("predictedFraud", "predict").withColumnRenamed("isFraud","label").rdd
# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels)

# COMMAND ----------

conf_mat =  test_pred.select(["predictedFraud", "isFraud"]).withColumnRenamed("predictedFraud", "predict").withColumnRenamed("isFraud","label").groupBy("predict","label").count()
print("Confusion Matrix")
display(conf_mat)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Metrics 3: Accuracy, Precision, Recall, F1score

# COMMAND ----------

tparray = conf_mat.filter((F.col("label") == 1) & (F.col("predict") == 1.0)).rdd.collect()
tp = tparray[0][2] if len(tparray) > 0 else 0.0

tnarray = conf_mat.filter((F.col("label") == 0) & (F.col("predict") == 0.0)).rdd.collect()
tn = tnarray[0][2] if len(tnarray) > 0 else 0.0

fparray = conf_mat.filter((F.col("label") == 0) & (F.col("predict") == 1.0)).rdd.collect()
fp = fparray[0][2] if len(fparray) > 0 else 0.0

fnarray = conf_mat.filter((F.col("label") == 1) & (F.col("predict") == 0.0)).rdd.collect()
fn = fnarray[0][2] if len(fnarray) > 0 else 0.0

# COMMAND ----------

print("True Positives: ", tp)
print("False Positives: ", fp)
print("True Negatives: ", tn)
print("False Negatives: ", fn)

# COMMAND ----------

precision = tp / (tp + fp) if tp+fp > 0 else 0.1
recall = tp / (tp + fn) if tp+fn > 0 else 0.1
beta = 0.0005
f_measure = (1 + beta*beta) * precision * recall / (beta*beta*precision + recall)
accuracy = (tp + tn) / (tp + tn + fp + fn) 
print("Precision" ,precision)
print("Recall", recall)
print("F1 measure", f_measure)
print("Accuracy", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment 4: Feature Ranking based on Information Gain

# COMMAND ----------

attr_gains = calculate_info_gain(fraud_dataset, [])
sorted_gains = dict(sorted(attr_gains.items(), key=lambda item: -item[1]))
print("Feature Ranking based on Info gain")
rank = 1
for key in sorted_gains:
  print("Rank ",rank,"-",key)
  rank = rank + 1

# COMMAND ----------

display(data.groupBy("type","isFraud").count())

# COMMAND ----------

display(data.groupBy("type","isFraud").count().filter(F.col("isFraud") == 1))

# COMMAND ----------

display(data.filter(F.col("isFraud") == 1).agg({"oldbalanceOrg": "avg"}))

# COMMAND ----------

display(data.groupBy("isFraud").count())
