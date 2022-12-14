# Importing PySpark libraries required for project:
import findspark
findspark.init()
findspark.find()
import pyspark
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline

# Importing numpy and pandas:
import numpy as np
import pandas as pd

# Importing metric functions:
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def labeled_point_convert(spark_context, features, labels, categorical = False):
    temp = []
    for x, y in zip(features, labels):        
        lbl_points = LabeledPoint(y, x)
        temp.append(lbl_points)
    return spark_context.parallelize(temp) 

config = pyspark.SparkConf().setAppName('winequality').setMaster('local')
spark_context = pyspark.SparkContext(config = config)
spark_sess = SparkSession(spark_context)

data = spark_sess.read.format("csv").load("TrainingDataset.csv", header = True, sep =";")
data.printSchema()
data.show()

for column_name in data.columns[1:-1]+['""""quality"""""']:
    data = data.withColumn(column_name, col(column_name).cast('float'))
data = data.withColumnRenamed('""""quality"""""', "label")

features = np.array(data.select(data.columns[1:-1]).collect())
label = np.array(data.select('label').collect())

vec_data = VectorAssembler(inputCols = data.columns[1:-1] , outputCol = 'features').transform(data)
vec_data = vec_data.select(['features','label'])

dset = labeled_point_convert(spark_context, features, label)


from sklearn.model_selection import train_test_split
train, test = dset.randomSplit([0.7, 0.3], seed = 2)

model = DecisionTree.trainClassifier(train, numClasses=10, categoricalFeaturesInfo={})
                                     
preds = model.predict(test.map(lambda l: l.features))

labels_preds = test.map(lambda lbl_point: lbl_point.label).zip(preds)

labels_preds_df = labels_preds.toDF()
label_pred = labels_preds.toDF(["label", "Prediction"])
label_pred.show()
label_pred_df = label_pred.toPandas()

print("Score for F1: ", f1_score(label_pred_df['label'], label_pred_df['Prediction'], average='micro'))

testing_error = labels_preds.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test.count())    
print('Test Error = ' + str(testing_error))

model.save(spark_context, 's3://winequal/trainingmodel.model')