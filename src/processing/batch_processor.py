"""Batch processing orchestration for profile analysis."""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm.auto import tqdm

from ..config import AppConfig, ModelConfig
from ..constants import make_model_columns
from ..model.pipeline import ProfileAnalysisPipeline
from .json_parser import parse_model_outputs_to_dataframe

logger = logging.getLogger(__name__)


class ProfileBatchProcessor:
    """Unified processor supporting single or multiple models."""

    def __init__(self, config: AppConfig):
        """Initialize the processor with application config."""
        self.config = config
        self.enabled_models = config.enabled_models

        if len(self.enabled_models) == 0:
            raise ValueError("No enabled models found in configuration")

    def get_model_summary(self) -> str:
        """Get a summary of enabled models."""
        summary = f"Enabled Models ({len(self.enabled_models)}):\n"
        for idx, model in enumerate(self.enabled_models, 1):
            summary += f"  {idx}. {model.name}: {model.model_id}\n"
        return summary

    def _process_with_model(
        self,
        df: pd.DataFrame,
        model_config: ModelConfig,
        input_col: str,
        parse_outputs: bool,
    ) -> pd.DataFrame:
        """Process dataframe with a single model and optionally parse outputs."""
        result_df = df.copy()

        logger.info(f"\n{'='*60}")
        logger.info(
            f"Processing Model {model_config.name} | ID: {model_config.model_id}"
        )
        logger.info(f"{'='*60}\n")

        cols = make_model_columns(model_config.name)
        output_col = cols["output"]

        pipeline: Optional[ProfileAnalysisPipeline] = None
        try:
            # Initialize pipeline
            pipeline = ProfileAnalysisPipeline(
                model_config=model_config,
                generation_config=self.config.generation,
            )
            pipeline.initialize()

            # Extract texts and identify non-empty ones
            texts = result_df[input_col].astype(str).tolist()
            is_nonempty = [bool(text.strip()) for text in texts]

            # Initialize output storage
            raw_outputs: list[Optional[str]] = [None] * len(texts)
            nonempty_indices = [i for i, ok in enumerate(is_nonempty) if ok]

            logger.info(
                f"Found {len(nonempty_indices)} non-empty profiles to process"
            )

            # Batch processing loop
            batch_size = self.config.generation.batch_size
            for batch_start in tqdm(
                range(0, len(nonempty_indices), batch_size),
                desc=f"{model_config.name}: Processing batches",
            ):
                batch_indices = nonempty_indices[batch_start : batch_start + batch_size]
                batch_texts = [texts[idx] for idx in batch_indices]

                # Generate assessments
                batch_outputs = pipeline.generate_batch(batch_texts)

                # Store results
                for idx, output in zip(batch_indices, batch_outputs):
                    raw_outputs[idx] = output

            # Attach raw outputs
            result_df[output_col] = raw_outputs

            # Parse outputs if requested
            if parse_outputs:
                logger.info(f"Parsing outputs for model '{model_config.name}'")
                parsed_df = parse_model_outputs_to_dataframe(
                    result_df[output_col],
                    prefix=f"ai_{model_config.name}_",
                )
                for col in parsed_df.columns:
                    result_df[col] = parsed_df[col]

            logger.info(f"✅ Model '{model_config.name}' processing complete!")

        except Exception as e:
            # Fail fast with context for clearer error surfacing
            raise RuntimeError(
                f"Failed to process model '{model_config.name}': {e}"
            ) from e
        finally:
            # Cleanup pipeline to free memory
            try:
                if pipeline is not None:
                    pipeline.cleanup()
            except Exception as cleanup_error:
                logger.warning(
                    f"Cleanup warning for '{model_config.name}': {cleanup_error}"
                )

        return result_df

    def process_dataframe(
        self,
        df: pd.DataFrame,
        input_column: Optional[str] = None,
        parse_outputs: bool = True,
    ) -> pd.DataFrame:
        """Process a DataFrame through all enabled models.

        Args:
            df: Input DataFrame with profile texts.
            input_column: Column name with "About Me" text.
            parse_outputs: Whether to parse JSON outputs into separate columns.

        Returns:
            DataFrame with outputs from all models.
        """
        if df is None or len(df) == 0:
            return df

        input_col = input_column or self.config.data.dataset_column
        if input_col not in df.columns:
            raise ValueError(f"Input column '{input_col}' not found in DataFrame")

        logger.info(
            f"Processing {len(df)} profiles through {len(self.enabled_models)} model(s)"
        )

        result_df = df.copy()
        for model_config in self.enabled_models:
            result_df = self._process_with_model(
                result_df,
                model_config,
                input_col,
                parse_outputs,
            )

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ All {len(self.enabled_models)} model(s) completed!")
        logger.info(f"{'='*60}\n")

        return result_df

    def process_csv(
        self,
        input_path: Path,
        output_path: Path,
        limit: Optional[int] = None,
        parse_outputs: bool = True,
    ) -> pd.DataFrame:
        """Process profiles from CSV through all models and save results."""
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)

        if limit:
            logger.info(f"Limiting to first {limit} rows")
            df = df.head(limit)

        # Process through all models
        result_df = self.process_dataframe(df, parse_outputs=parse_outputs)

        # Save results
        logger.info(f"Saving results to {output_path}")
        result_df.to_csv(output_path, index=False)
        logger.info("Results saved successfully!")

        return result_df


 