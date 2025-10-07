"""Batch processing orchestration for profile analysis."""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm.auto import tqdm

from ..config import AppConfig, ModelConfig
from ..model.pipeline import ProfileAnalysisPipeline
from .json_parser import parse_model_outputs_to_dataframe

logger = logging.getLogger(__name__)


class ProfileBatchProcessor:
    """Orchestrates batch processing of job seeker profiles."""
    
    def __init__(
        self,
        pipeline: ProfileAnalysisPipeline,
        config: AppConfig,
    ):
        """Initialize the batch processor.
        
        Args:
            pipeline: Initialized LLM pipeline.
            config: Application configuration.
        """
        self.pipeline = pipeline
        self.config = config
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        input_column: Optional[str] = None,
        output_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Process a DataFrame of profiles through the LLM.
        
        Args:
            df: Input DataFrame with profile texts.
            input_column: Column name with "About Me" text.
            output_column: Column name for model outputs.
            
        Returns:
            DataFrame with added output column.
        """
        input_col = input_column or self.config.data.dataset_column
        output_col = output_column or self.config.data.output_column
        
        if input_col not in df.columns:
            raise ValueError(f"Input column '{input_col}' not found in DataFrame")
        
        logger.info(f"Processing {len(df)} profiles from column '{input_col}'")
        
        # Extract texts and identify non-empty ones
        texts = df[input_col].astype(str).tolist()
        is_nonempty = [bool(text.strip()) for text in texts]
        
        # Initialize output storage
        raw_outputs = [None] * len(texts)
        nonempty_indices = [i for i, ok in enumerate(is_nonempty) if ok]
        
        logger.info(
            f"Found {len(nonempty_indices)} non-empty profiles to process"
        )
        
        # Batch processing loop
        batch_size = self.config.generation.batch_size
        
        for batch_start in tqdm(
            range(0, len(nonempty_indices), batch_size),
            desc="Processing batches",
        ):
            batch_indices = nonempty_indices[batch_start : batch_start + batch_size]
            batch_texts = [texts[idx] for idx in batch_indices]
            
            # Generate assessments
            batch_outputs = self.pipeline.generate_batch(batch_texts)
            
            # Store results
            for idx, output in zip(batch_indices, batch_outputs):
                raw_outputs[idx] = output
        
        # Add to DataFrame
        result_df = df.copy()
        result_df[output_col] = raw_outputs
        
        logger.info(f"Processing complete. Results in column '{output_col}'")
        
        return result_df
    
    def process_csv(
        self,
        input_path: Path,
        output_path: Path,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Process profiles from CSV file and save results.
        
        Args:
            input_path: Path to input CSV file.
            output_path: Path for output CSV file.
            limit: Optional limit on number of rows to process.
            
        Returns:
            Processed DataFrame.
        """
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        if limit:
            logger.info(f"Limiting to first {limit} rows")
            df = df.head(limit)
        
        # Process
        result_df = self.process_dataframe(df)
        
        # Save
        logger.info(f"Saving results to {output_path}")
        result_df.to_csv(output_path, index=False)
        
        return result_df


class MultiModelBatchProcessor:
    """Orchestrates batch processing across multiple models."""
    
    def __init__(self, config: AppConfig):
        """Initialize multi-model processor.
        
        Args:
            config: Application configuration with multiple models.
        """
        self.config = config
        self.enabled_models = config.get_enabled_models()
        
        if len(self.enabled_models) == 0:
            raise ValueError("No enabled models found in configuration")
        
        logger.info(f"MultiModelBatchProcessor initialized with {len(self.enabled_models)} model(s)")
    
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
        result_df = df.copy()
        input_col = input_column or self.config.data.dataset_column
        
        logger.info(f"Processing {len(df)} profiles through {len(self.enabled_models)} model(s)")
        
        # Process each enabled model sequentially
        for idx, model_config in enumerate(self.enabled_models, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing Model {idx}/{len(self.enabled_models)}: {model_config.name}")
            logger.info(f"Model ID: {model_config.model_id}")
            logger.info(f"{'='*60}\n")
            
            try:
                # Initialize pipeline for this model
                pipeline = ProfileAnalysisPipeline(
                    model_config=model_config,
                    generation_config=self.config.generation,
                )
                pipeline.initialize()
                
                # Create single-model processor
                processor = ProfileBatchProcessor(pipeline, self.config)
                
                # Generate model-specific output column name
                output_col = f"{self.config.data.output_column}_{model_config.name}"
                
                # Process data
                result_df = processor.process_dataframe(
                    result_df,
                    input_column=input_col,
                    output_column=output_col,
                )
                
                # Parse outputs if requested
                if parse_outputs:
                    logger.info(f"Parsing outputs for model '{model_config.name}'")
                    parsed_df = parse_model_outputs_to_dataframe(
                        result_df[output_col],
                        prefix=f"ai_{model_config.name}_",
                    )
                    # Add parsed columns to result
                    for col in parsed_df.columns:
                        result_df[col] = parsed_df[col]
                
                logger.info(f"✅ Model '{model_config.name}' processing complete!")
                
            except Exception as e:
                logger.error(f"❌ Error processing model '{model_config.name}': {e}")
                # Add empty columns for this model
                output_col = f"{self.config.data.output_column}_{model_config.name}"
                result_df[output_col] = None
                raise
                
            finally:
                # Always cleanup to free memory
                try:
                    pipeline.cleanup()
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup warning for '{model_config.name}': {cleanup_error}")
        
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
        """Process profiles from CSV through all models and save results.
        
        Args:
            input_path: Path to input CSV file.
            output_path: Path for output CSV file.
            limit: Optional limit on number of rows to process.
            parse_outputs: Whether to parse JSON outputs into separate columns.
            
        Returns:
            Processed DataFrame with all model outputs.
        """
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
        logger.info(f"Results saved successfully!")
        
        return result_df
    
    def get_model_summary(self) -> str:
        """Get a summary of enabled models.
        
        Returns:
            Formatted string with model information.
        """
        summary = f"Enabled Models ({len(self.enabled_models)}):\n"
        for idx, model in enumerate(self.enabled_models, 1):
            summary += f"  {idx}. {model.name}: {model.model_id}\n"
        return summary