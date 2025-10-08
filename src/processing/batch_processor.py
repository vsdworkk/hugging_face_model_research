"""Batch processing orchestration for profile analysis."""
from typing import Optional, List

import pandas as pd
from tqdm.auto import tqdm

from ..config import AppConfig, ModelConfig
from ..constants import make_model_columns
from ..model.pipeline import ProfileAnalysisPipeline
from .json_parser import parse_model_outputs_to_dataframe


class ProfileBatchProcessor:
    """Runs one or more models over a DataFrame of 'About Me' texts."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.enabled_models: List[ModelConfig] = config.enabled_models
        if not self.enabled_models:
            raise ValueError("No enabled models found in configuration")

    def get_model_summary(self) -> str:
        lines = [f"Enabled Models ({len(self.enabled_models)}):"]
        lines += [f"  {i}. {m.name}: {m.model_id}" for i, m in enumerate(self.enabled_models, 1)]
        return "\n".join(lines)

    def _process_with_model(
        self,
        df: pd.DataFrame,
        model_config: ModelConfig,
        input_col: str,
        parse_outputs: bool,
    ) -> pd.DataFrame:
        result_df = df.copy()
        cols = make_model_columns(model_config.name)
        output_col = cols["output"]

        pipeline: Optional[ProfileAnalysisPipeline] = None
        try:
            pipeline = ProfileAnalysisPipeline(
                model_config=model_config,
                generation_config=self.config.generation,
            )
            pipeline.initialize()

            texts = result_df[input_col].astype(str).tolist()
            nonempty_idx = [i for i, txt in enumerate(texts) if txt.strip()]

            raw_outputs: list[Optional[str]] = [None] * len(texts)
            bs = self.config.generation.batch_size

            for start in tqdm(range(0, len(nonempty_idx), bs), desc=f"{model_config.name}: batches"):
                idxs = nonempty_idx[start:start + bs]
                batch_texts = [texts[i] for i in idxs]

                batch_out = pipeline.generate_batch(batch_texts)
                for i, out in zip(idxs, batch_out):
                    raw_outputs[i] = out

            result_df[output_col] = raw_outputs

            if parse_outputs:
                parsed_df = parse_model_outputs_to_dataframe(result_df[output_col], prefix=f"ai_{model_config.name}_")
                # align columns
                for col in parsed_df.columns:
                    result_df[col] = parsed_df[col]

        except Exception as e:
            raise RuntimeError(f"Failed to process model '{model_config.name}': {e}") from e
        finally:
            if pipeline is not None:
                try:
                    pipeline.cleanup()
                except Exception:
                    pass

        return result_df

    def process_dataframe(
        self,
        df: pd.DataFrame,
        input_column: Optional[str] = None,
        parse_outputs: bool = True,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        input_col = input_column or self.config.data.dataset_column
        if input_col not in df.columns:
            raise ValueError(f"Input column '{input_col}' not found in DataFrame")

        out = df.copy()
        for m in self.enabled_models:
            out = self._process_with_model(out, m, input_col, parse_outputs)

        return out
