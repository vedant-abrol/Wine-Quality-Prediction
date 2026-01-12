#!/usr/bin/env python3
"""
Wine Quality Prediction - Model Training Script
================================================

This script trains machine learning models to predict wine quality based on
chemical properties. It compares Logistic Regression and Random Forest
classifiers, selects the best performing model, and saves it for later use.

Author: Vedant Abrol
Course: CS643 - Cloud Computing
"""

import os
import sys
import logging
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, Normalizer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

import quinn

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths - Update these based on your environment
TRAINING_DATA_PATH = "TrainingDataset.csv"
VALIDATION_DATA_PATH = "ValidationDataset.csv"
MODEL_OUTPUT_PATH = "wine_quality_model"

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

# Model training parameters
CROSS_VALIDATION_FOLDS = 3

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

def create_spark_session(app_name: str = "WineQualityTraining") -> SparkSession:
    """
    Create and configure a Spark session.
    
    Args:
        app_name: Name of the Spark application
        
    Returns:
        Configured SparkSession instance
    """
    logger.info("Initializing Spark session...")
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    
    # Set log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info(f"Spark session created: {app_name}")
    return spark


def load_dataset(spark: SparkSession, file_path: str, dataset_name: str):
    """
    Load a CSV dataset into a Spark DataFrame.
    
    Args:
        spark: Active SparkSession
        file_path: Path to the CSV file
        dataset_name: Name for logging purposes
        
    Returns:
        Spark DataFrame with loaded data
    """
    logger.info(f"Loading {dataset_name} from: {file_path}")
    
    df = spark.read.format('csv') \
        .options(header='true', inferSchema='true', sep=';') \
        .load(file_path)
    
    row_count = df.count()
    logger.info(f"  → Loaded {row_count:,} rows with {len(df.columns)} columns")
    
    return df


def clean_column_names(df):
    """
    Remove quotation marks from column names.
    
    Args:
        df: Input DataFrame with potentially quoted column names
        
    Returns:
        DataFrame with cleaned column names
    """
    def remove_quotes(s):
        return s.replace('"', '')
    
    return quinn.with_columns_renamed(remove_quotes)(df)


def prepare_data(df, label_column: str = "quality"):
    """
    Prepare the dataset for training by cleaning and renaming columns.
    
    Args:
        df: Raw DataFrame
        label_column: Name of the target column
        
    Returns:
        Cleaned DataFrame with 'label' column
    """
    # Clean column names
    df = clean_column_names(df)
    
    # Rename target column to 'label' for MLlib compatibility
    df = df.withColumnRenamed(label_column, 'label')
    
    return df


def build_pipeline(classifier):
    """
    Build a machine learning pipeline with feature assembly, scaling, and classifier.
    
    Args:
        classifier: MLlib classifier instance (e.g., LogisticRegression)
        
    Returns:
        Configured Pipeline instance
    """
    # Assemble features into a single vector
    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="inputFeatures",
        handleInvalid="skip"
    )
    
    # Normalize features to unit norm
    normalizer = Normalizer(
        inputCol="inputFeatures",
        outputCol="features",
        p=2.0  # L2 normalization
    )
    
    # Build pipeline
    pipeline = Pipeline(stages=[assembler, normalizer, classifier])
    
    return pipeline


def train_and_evaluate(pipeline, train_df, validation_df, model_name: str):
    """
    Train a model using cross-validation and evaluate on validation set.
    
    Args:
        pipeline: ML Pipeline to train
        train_df: Training DataFrame
        validation_df: Validation DataFrame
        model_name: Name of the model for logging
        
    Returns:
        Tuple of (trained_model, f1_score)
    """
    logger.info(f"Training {model_name}...")
    
    # Set up cross-validation
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
        numFolds=CROSS_VALIDATION_FOLDS,
        parallelism=2
    )
    
    # Train the model
    start_time = datetime.now()
    cv_model = crossval.fit(train_df)
    training_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"  → Training completed in {training_time:.2f} seconds")
    
    # Evaluate on validation set
    predictions = cv_model.transform(validation_df)
    f1_score = evaluator.evaluate(predictions)
    
    # Calculate additional metrics
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = accuracy_evaluator.evaluate(predictions)
    
    logger.info(f"  → F1 Score: {f1_score:.4f}")
    logger.info(f"  → Accuracy: {accuracy:.4f}")
    
    return cv_model, f1_score


def save_model(model, output_path: str):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model to save
        output_path: Directory path for saving the model
    """
    logger.info(f"Saving model to: {output_path}")
    model.write().overwrite().save(output_path)
    logger.info("  → Model saved successfully!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main training workflow."""
    
    logger.info("=" * 60)
    logger.info("WINE QUALITY PREDICTION - MODEL TRAINING")
    logger.info("=" * 60)
    
    # Initialize Spark
    spark = create_spark_session("CS643_Wine_Quality_Training")
    
    try:
        # Load datasets
        logger.info("-" * 60)
        logger.info("STEP 1: Loading Data")
        logger.info("-" * 60)
        
        train_df = load_dataset(spark, TRAINING_DATA_PATH, "Training Dataset")
        validation_df = load_dataset(spark, VALIDATION_DATA_PATH, "Validation Dataset")
        
        # Prepare data
        logger.info("-" * 60)
        logger.info("STEP 2: Preparing Data")
        logger.info("-" * 60)
        
        train_df = prepare_data(train_df)
        validation_df = prepare_data(validation_df)
        
        logger.info("Data preparation completed")
        logger.info(f"Features: {FEATURE_COLUMNS}")
        
        # Train models
        logger.info("-" * 60)
        logger.info("STEP 3: Training Models")
        logger.info("-" * 60)
        
        # Model 1: Logistic Regression
        lr_pipeline = build_pipeline(LogisticRegression(maxIter=100))
        lr_model, lr_f1 = train_and_evaluate(
            lr_pipeline, train_df, validation_df, "Logistic Regression"
        )
        
        # Model 2: Random Forest
        rf_pipeline = build_pipeline(RandomForestClassifier(numTrees=100))
        rf_model, rf_f1 = train_and_evaluate(
            rf_pipeline, train_df, validation_df, "Random Forest"
        )
        
        # Compare and select best model
        logger.info("-" * 60)
        logger.info("STEP 4: Model Selection")
        logger.info("-" * 60)
        
        logger.info("Model Comparison:")
        logger.info(f"  • Logistic Regression F1: {lr_f1:.4f}")
        logger.info(f"  • Random Forest F1:       {rf_f1:.4f}")
        
        if lr_f1 >= rf_f1:
            best_model = lr_model
            best_name = "Logistic Regression"
            best_score = lr_f1
        else:
            best_model = rf_model
            best_name = "Random Forest"
            best_score = rf_f1
        
        logger.info(f"\n✓ Selected Model: {best_name} (F1: {best_score:.4f})")
        
        # Save the best model
        logger.info("-" * 60)
        logger.info("STEP 5: Saving Model")
        logger.info("-" * 60)
        
        save_model(best_model, MODEL_OUTPUT_PATH)
        
        # Summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best Model: {best_name}")
        logger.info(f"F1 Score: {best_score:.4f}")
        logger.info(f"Model saved to: {MODEL_OUTPUT_PATH}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
        
    finally:
        spark.stop()
        logger.info("Spark session closed")


if __name__ == "__main__":
    main()
