#!/usr/bin/env python3
"""
Wine Quality Prediction - Inference Script
==========================================

This script loads a trained model and makes predictions on new wine data.
It can either load a pre-trained model or train a new one if no saved model exists.

Usage:
    python Predictions.py                           # Use default paths
    python Predictions.py --test-data custom.csv   # Custom test data

Author: Vedant Abrol
Course: CS643 - Cloud Computing
"""

import os
import sys
import argparse
import logging

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Normalizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml import Pipeline

import quinn

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default file paths
DEFAULT_TRAINING_DATA = "TrainingDataset.csv"
DEFAULT_TEST_DATA = "ValidationDataset.csv"
DEFAULT_MODEL_PATH = "wine_quality_model"

# Feature columns (must match training)
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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Wine Quality Prediction - Make predictions on wine data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--training-data",
        type=str,
        default=DEFAULT_TRAINING_DATA,
        help=f"Path to training data CSV (default: {DEFAULT_TRAINING_DATA})"
    )
    
    parser.add_argument(
        "--test-data",
        type=str,
        default=DEFAULT_TEST_DATA,
        help=f"Path to test data CSV (default: {DEFAULT_TEST_DATA})"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to saved model directory (default: {DEFAULT_MODEL_PATH})"
    )
    
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if saved model exists"
    )
    
    return parser.parse_args()


def create_spark_session(app_name: str = "WineQualityPrediction") -> SparkSession:
    """Create and configure a Spark session."""
    logger.info("Initializing Spark session...")
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"Spark session created: {app_name}")
    
    return spark


def load_dataset(spark: SparkSession, file_path: str, dataset_name: str):
    """Load a CSV dataset into a Spark DataFrame."""
    logger.info(f"Loading {dataset_name}: {file_path}")
    
    df = spark.read.format('csv') \
        .options(header='true', inferSchema='true', sep=';') \
        .load(file_path)
    
    logger.info(f"  → Loaded {df.count():,} rows")
    return df


def clean_column_names(df):
    """Remove quotation marks from column names."""
    def remove_quotes(s):
        return s.replace('"', '')
    return quinn.with_columns_renamed(remove_quotes)(df)


def prepare_data(df, label_column: str = "quality"):
    """Prepare the dataset for prediction."""
    df = clean_column_names(df)
    df = df.withColumnRenamed(label_column, 'label')
    return df


def model_exists(model_path: str) -> bool:
    """Check if a saved model exists at the given path."""
    return os.path.exists(model_path) and os.path.isdir(model_path)


def load_model(model_path: str):
    """Load a previously saved model."""
    logger.info(f"Loading saved model from: {model_path}")
    model = CrossValidatorModel.load(model_path)
    logger.info("  → Model loaded successfully!")
    return model


def train_model(spark: SparkSession, training_path: str):
    """Train a new model if no saved model exists."""
    logger.info("No saved model found. Training new model...")
    
    # Load and prepare training data
    train_df = load_dataset(spark, training_path, "Training Data")
    train_df = prepare_data(train_df)
    
    # Build pipeline
    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="inputFeatures",
        handleInvalid="skip"
    )
    
    normalizer = Normalizer(
        inputCol="inputFeatures",
        outputCol="features",
        p=2.0
    )
    
    lr = LogisticRegression(maxIter=100)
    pipeline = Pipeline(stages=[assembler, normalizer, lr])
    
    # Cross-validation
    param_grid = ParamGridBuilder().build()
    evaluator = MulticlassClassificationEvaluator(metricName="f1")
    
    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3
    )
    
    # Train
    logger.info("Training Logistic Regression model...")
    model = crossval.fit(train_df)
    logger.info("  → Training complete!")
    
    return model


def evaluate_model(model, test_df):
    """Evaluate the model and return metrics."""
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


def display_sample_predictions(predictions, num_samples: int = 10):
    """Display sample predictions."""
    logger.info(f"\nSample Predictions (first {num_samples} rows):")
    logger.info("-" * 50)
    
    sample = predictions.select("label", "prediction", "features") \
        .limit(num_samples) \
        .collect()
    
    for i, row in enumerate(sample, 1):
        actual = int(row["label"])
        predicted = int(row["prediction"])
        match = "✓" if actual == predicted else "✗"
        logger.info(f"  {i:2}. Actual: {actual} | Predicted: {predicted} | {match}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main prediction workflow."""
    
    # Parse arguments
    args = parse_arguments()
    
    logger.info("=" * 60)
    logger.info("WINE QUALITY PREDICTION - INFERENCE")
    logger.info("=" * 60)
    
    # Initialize Spark
    spark = create_spark_session("CS643_Wine_Quality_Prediction")
    
    try:
        # Load or train model
        logger.info("-" * 60)
        logger.info("STEP 1: Loading Model")
        logger.info("-" * 60)
        
        if model_exists(args.model_path) and not args.force_retrain:
            model = load_model(args.model_path)
        else:
            model = train_model(spark, args.training_data)
            # Save the newly trained model
            logger.info(f"Saving model to: {args.model_path}")
            model.write().overwrite().save(args.model_path)
        
        # Load test data
        logger.info("-" * 60)
        logger.info("STEP 2: Loading Test Data")
        logger.info("-" * 60)
        
        test_df = load_dataset(spark, args.test_data, "Test Data")
        test_df = prepare_data(test_df)
        
        # Make predictions
        logger.info("-" * 60)
        logger.info("STEP 3: Making Predictions")
        logger.info("-" * 60)
        
        predictions, f1_score, accuracy = evaluate_model(model, test_df)
        
        # Display results
        logger.info("-" * 60)
        logger.info("STEP 4: Results")
        logger.info("-" * 60)
        
        display_sample_predictions(predictions)
        
        # Summary
        logger.info("=" * 60)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Test Data: {args.test_data}")
        logger.info(f"Total Samples: {predictions.count():,}")
        logger.info(f"F1 Score: {f1_score:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("=" * 60)
        
        return f1_score
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise
        
    finally:
        spark.stop()
        logger.info("Spark session closed")


if __name__ == "__main__":
    main()
