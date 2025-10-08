"""LLM pipeline initialization and management."""
import logging
from typing import Optional

import torch
from transformers import Pipeline, pipeline

from ..config import ModelConfig, GenerationConfig
from .prompts import build_profile_analysis_messages

logger = logging.getLogger(__name__)


def _resolve_torch_dtype(dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    m = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return m.get(dtype_str, "auto")


class ProfileAnalysisPipeline:
    """Manages LLM pipeline for profile quality analysis."""

    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig):
        self.model_config = model_config
        self.generation_config = generation_config
        self.model_name = model_config.name
        self._pipe: Optional[Pipeline] = None

    def initialize(self) -> None:
        logger.info("Loading model '%s': %s", self.model_name, self.model_config.model_id)

        hub_kwargs = {}
        if self.model_config.hf_token:
            # transformers >=4.37 prefers 'token', older versions used 'use_auth_token'
            hub_kwargs["token"] = self.model_config.hf_token

        self._pipe = pipeline(
            task="text-generation",
            model=self.model_config.model_id,
            torch_dtype=_resolve_torch_dtype(self.model_config.torch_dtype),
            device_map=self.model_config.device_map,
            **hub_kwargs,
        )

        tok = self._pipe.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        tok.padding_side = "left"

        logger.info("Pipeline '%s' initialized", self.model_name)

    def cleanup(self) -> None:
        if self._pipe is not None:
            logger.info("Cleaning up model '%s'", self.model_name)
            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")

    @property
    def pipe(self) -> Pipeline:
        if self._pipe is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self._pipe

    def build_prompt(self, about_text: str) -> str:
        messages = build_profile_analysis_messages(about_text)
        return self.pipe.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def generate_batch(self, about_texts: list[str]) -> list[str]:
        prompts = [self.build_prompt(t) for t in about_texts]
        # batching is handled by ProfileBatchProcessor; we call once per batch here
        outputs = self.pipe(
            prompts,
            do_sample=self.generation_config.do_sample,
            num_beams=self.generation_config.num_beams,
            max_new_tokens=self.generation_config.max_new_tokens,
            return_full_text=False,
        )
        return [o[0]["generated_text"].strip() for o in outputs]

    def test_generation(self, test_prompt: str = "How old is the sun?") -> str:
        """Quick functional test; returns raw text."""
        prompt = self.build_prompt(test_prompt)
        out = self.pipe(prompt, max_new_tokens=64, return_full_text=False)
        return out[0]["generated_text"]
