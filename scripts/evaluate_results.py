#!/usr/bin/env python3
"""Script to evaluate scored profiles against human labels."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.evaluation.metrics import ModelEvaluator
from src.processing.json_parser import parse_model_outputs_to_dataframe
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions against human labels"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to scored CSV file (output from run_scoring.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save wide evaluation table (optional)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to configuration file",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("Starting evaluation")
    
    # Load configuration
    config = load_config(args.config)
    
    # Load scored data
    logger.info(f"Loading scored data from {args.input}")
    df = pd.read_csv(args.input)
    
    # Parse JSON outputs to structured columns
    ai_cols = parse_model_outputs_to_dataframe(
        df[config.data.output_column],
        prefix="ai_",
    )
    
    # Merge with original data
    wide_df = pd.concat([df.reset_index(drop=True), ai_cols], axis=1)
    
    # Evaluate
    evaluator = ModelEvaluator(
        ai_quality_col="ai_quality",
        human_label_col=config.data.human_label_column,
    )
    
    # Set output path
    output_path = args.output
    if output_path is None and config.evaluation.save_wide_table:
        output_path = args.input.parent / f"{args.input.stem}_evaluated.csv"
    
    wide_df, metrics = evaluator.evaluate_dataframe(
        wide_df,
        save_path=output_path,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTotal Samples: {metrics.total_samples}")
    print("\nConfusion Matrix:")
    print(metrics.confusion_matrix_df())
    print("\nMetrics:")
    for metric_name, value in metrics.summary_dict().items():
        print(f"  {metric_name.replace('_', ' ').title()}: {value:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()