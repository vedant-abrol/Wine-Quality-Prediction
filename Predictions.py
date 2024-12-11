import quinn
import requests

from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler
from pyspark.ml.classification import LogisticRegression

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

sc = SparkContext('local')
spark = SparkSession(sc)

train_df = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/home/ubuntu/CS643-AWS-ProgAssgn-2/TrainingDataset.csv')
test_df = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/home/ubuntu/CS643-AWS-ProgAssgn-2/ValidationDataset.csv')

print("Data loaded into Spark.")
print(train_df.toPandas().head())

def remove_quotations(s):
    return s.replace('"', '')

train_df = quinn.with_columns_renamed(remove_quotations)(train_df)
train_df = train_df.withColumnRenamed('quality', 'label')

test_df = quinn.with_columns_renamed(remove_quotations)(test_df)
test_df = test_df.withColumnRenamed('quality', 'label')

print("Data has been formatted.")
print(train_df.toPandas().head())

assembler = VectorAssembler(
    inputCols=["fixed acidity",
               "volatile acidity",
               "citric acid",
               "residual sugar",
               "chlorides",
               "free sulfur dioxide",
               "total sulfur dioxide",
               "density",
               "pH",
               "sulphates",
               "alcohol"],
                outputCol="inputFeatures")

scaler = Normalizer(inputCol="inputFeatures", outputCol="features")

## Only use LogisticRegression because this was the best model decided on
lr = LogisticRegression()

pipeline1 = Pipeline(stages=[assembler, scaler, lr])

paramgrid = ParamGridBuilder().build()

evaluator = MulticlassClassificationEvaluator(metricName="f1")

crossval = CrossValidator(estimator=pipeline1,  
                         estimatorParamMaps=paramgrid,
                         evaluator=evaluator, 
                         numFolds=3
                        )

cvModel1 = crossval.fit(train_df) 
print("F1 Score for Our Model: ", evaluator.evaluate(cvModel1.transform(test_df)))
