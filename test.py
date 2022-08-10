# Importing PySpark libraries required for project:
import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
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
import sys

# Importing numpy and pandas:
import pandas as pd
import numpy as np

# Importing metric functions:
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def labeled_point_convert(spark_context, feats, labels, categorical = False):
    temp = []
    for x, y in zip(feats, labels):        
        lbl_points = LabeledPoint(y, x)
        temp.append(lbl_points)
    return spark_context.parallelize(temp)

config = pyspark.SparkConf().setAppName('winequality').setMaster('local')
spark_context = pyspark.SparkContext(config = config)
spark_sess = SparkSession(spark_context)

path  = sys.argv[1]
valid = spark.read.format("csv").load(path, header = True , sep=";")
valid.printSchema()
valid.show()

for column_name in valid.columns[1:-1]+['""""quality"""""']:
    valid = valid.withColumn(column_name, col(column_name).cast('float'))
valid = valid.withColumnRenamed('""""quality"""""', "label")

feats = np.array(valid.select(valid.columns[1:-1]).collect())
label = np.array(valid.select('label').collect())

vec_assembler = VectorAssembler(inputCols = val.columns[1:-1] , outputCol = 'features')
data_tr = vec_assembler.transform(valid)
data_tr = data_tr.select(['features','label'])

dset = labeled_point_convert(spark_context, feats, label)

model = RandomForestModel.load(spark_context, "/wineprediction/trainingmodel.model/")

preds = model.predict(dset.map(lambda l: l.feats))

labels_preds = dset.map(lambda lp: lp.label).zip(preds)

labels_preds_df = pd.DataFrame(data = labels_preds)
label_pred = labels_preds.toDF(["label", "Prediction"])
label_pred.show()
label_pred_df = label_pred.toPandas()

f1 = f1_score(label_pred_df['label'], label_pred_df['Prediction'], average='micro')
print("Score for F1: ", f1)
print(confusion_matrix(label_pred_df['label'],label_pred_df['Prediction']))
print(classification_report(label_pred_df['label'],label_pred_df['Prediction']))
print("Accuracy" , accuracy_score(label_pred_df['label'], label_pred_df['Prediction']))

error_test = labels_preds.filter(
    lambda lp: lp[0] != lp[1]).count() / float(dataset.count())    
print('Test Error = ' + str(error_test))