#!/usr/bin/env python3
"""Script to run profile quality scoring on a dataset."""

import argparse
import logging
from pathlib import Path

from src.config import load_config
from src.processing.batch_processor import ProfileBatchProcessor
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main entry point for scoring script."""
    parser = argparse.ArgumentParser(
        description="Score job seeker profiles for quality"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to process (for testing)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path",
    )
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Use multi-model processing (process with all enabled models)",
    )
    parser.add_argument(
        "--no-parse",
        action="store_true",
        help="Skip parsing JSON outputs into separate columns",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file=args.log_file)
    logger.info("Starting profile quality scoring")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Unified processing through all enabled models
    processor = ProfileBatchProcessor(config)
    logger.info(processor.get_model_summary())
    
    # Process data
    processor.process_csv(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        parse_outputs=not args.no_parse,
    )
    
    logger.info("Scoring complete!")


if __name__ == "__main__":
    main()