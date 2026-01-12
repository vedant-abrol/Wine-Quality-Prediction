#!/usr/bin/env python3
"""
Wine Quality Prediction - Docker Inference Script
==================================================

This script is optimized for running inside a Docker container.
It trains a model and makes predictions on wine data.

The script expects data files to be mounted at /data/:
    - /data/TrainingDataset.csv
    - /data/ValidationDataset.csv

Author: Vedant Abrol
Course: CS643 - Cloud Computing
"""

import os
import logging

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Normalizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

import quinn

# =============================================================================
# CONFIGURATION
# =============================================================================

# Docker mount paths
DATA_DIR = os.environ.get("DATA_DIR", "/data")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "TrainingDataset.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "ValidationDataset.csv")

# Feature columns used for prediction
FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_spark_session() -> SparkSession:
    """Create and configure a Spark session for local execution."""
    logger.info("Initializing Spark session...")
    
    spark = SparkSession.builder \
        .appName("WineQualityPrediction_Docker") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session created successfully")
    
    return spark


def load_dataset(spark: SparkSession, file_path: str, dataset_name: str):
    """Load a CSV dataset into a Spark DataFrame."""
    logger.info(f"Loading {dataset_name}: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = spark.read.format('csv') \
        .options(header='true', inferSchema='true', sep=';') \
        .load(file_path)
    
    row_count = df.count()
    logger.info(f"  → Loaded {row_count:,} rows with {len(df.columns)} columns")
    
    return df


def clean_column_names(df):
    """Remove quotation marks from column names."""
    def remove_quotes(s):
        return s.replace('"', '')
    return quinn.with_columns_renamed(remove_quotes)(df)


def prepare_data(df, label_column: str = "quality"):
    """Prepare the dataset for training/prediction."""
    df = clean_column_names(df)
    df = df.withColumnRenamed(label_column, 'label')
    return df


def build_pipeline():
    """Build the ML pipeline with feature processing and classifier."""
    
    # Feature assembly
    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="inputFeatures",
        handleInvalid="skip"
    )
    
    # Feature normalization
    normalizer = Normalizer(
        inputCol="inputFeatures",
        outputCol="features",
        p=2.0
    )
    
    # Logistic Regression classifier
    classifier = LogisticRegression(maxIter=100)
    
    # Build pipeline
    pipeline = Pipeline(stages=[assembler, normalizer, classifier])
    
    return pipeline


def train_model(pipeline, train_df):
    """Train the model using cross-validation."""
    logger.info("Training model with cross-validation...")
    
    param_grid = ParamGridBuilder().build()
    
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    
    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3
    )
    
    model = crossval.fit(train_df)
    logger.info("  → Model training complete!")
    
    return model


def evaluate_model(model, test_df):
    """Evaluate model performance on test data."""
    predictions = model.transform(test_df)
    
    # F1 Score
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    f1_score = f1_evaluator.evaluate(predictions)
    
    # Accuracy
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = acc_evaluator.evaluate(predictions)
    
    return predictions, f1_score, accuracy


def display_results(predictions, f1_score, accuracy):
    """Display prediction results and metrics."""
    
    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 60)
    
    # Show sample predictions
    logger.info("\nSample Predictions:")
    logger.info("-" * 40)
    
    sample = predictions.select("label", "prediction") \
        .limit(10) \
        .collect()
    
    correct = 0
    for i, row in enumerate(sample, 1):
        actual = int(row["label"])
        predicted = int(row["prediction"])
        match = "✓" if actual == predicted else "✗"
        if actual == predicted:
            correct += 1
        logger.info(f"  {i:2}. Actual: {actual} | Predicted: {predicted} | {match}")
    
    # Summary metrics
    logger.info("\n" + "=" * 60)
    logger.info("MODEL PERFORMANCE METRICS")
    logger.info("=" * 60)
    logger.info(f"  F1 Score:  {f1_score:.4f}")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info("=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution workflow for Docker container."""
    
    logger.info("=" * 60)
    logger.info("WINE QUALITY PREDICTION - DOCKER CONTAINER")
    logger.info("=" * 60)
    logger.info(f"Data directory: {DATA_DIR}")
    
    # Initialize Spark
    spark = create_spark_session()
    
    try:
        # Load data
        logger.info("-" * 60)
        logger.info("STEP 1: Loading Data")
        logger.info("-" * 60)
        
        train_df = load_dataset(spark, TRAINING_DATA_PATH, "Training Dataset")
        test_df = load_dataset(spark, TEST_DATA_PATH, "Test Dataset")
        
        # Prepare data
        logger.info("-" * 60)
        logger.info("STEP 2: Preparing Data")
        logger.info("-" * 60)
        
        train_df = prepare_data(train_df)
        test_df = prepare_data(test_df)
        logger.info("Data preparation complete")
        
        # Build and train model
        logger.info("-" * 60)
        logger.info("STEP 3: Training Model")
        logger.info("-" * 60)
        
        pipeline = build_pipeline()
        model = train_model(pipeline, train_df)
        
        # Evaluate
        logger.info("-" * 60)
        logger.info("STEP 4: Evaluating Model")
        logger.info("-" * 60)
        
        predictions, f1_score, accuracy = evaluate_model(model, test_df)
        
        # Display results
        display_results(predictions, f1_score, accuracy)
        
        logger.info("\n✓ Prediction workflow completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error("Make sure to mount data directory: docker run -v /path/to/data:/data ...")
        raise
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise
        
    finally:
        spark.stop()
        logger.info("Spark session closed")


if __name__ == "__main__":
    main()
