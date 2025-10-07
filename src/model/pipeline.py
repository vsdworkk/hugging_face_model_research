"""LLM pipeline initialization and management."""

import logging
from typing import Any, Optional

import torch
from transformers import Pipeline, pipeline

from ..config import ModelConfig, GenerationConfig
from .prompts import build_profile_analysis_messages

logger = logging.getLogger(__name__)


class ProfileAnalysisPipeline:
    """Manages LLM pipeline for profile quality analysis."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
    ):
        """Initialize the analysis pipeline.
        
        Args:
            model_config: Model configuration settings.
            generation_config: Text generation configuration.
        """
        self.model_config = model_config
        self.generation_config = generation_config
        self.model_name = model_config.name
        self._pipe: Optional[Pipeline] = None
        
    def initialize(self) -> None:
        """Load and initialize the LLM pipeline."""
        logger.info(f"Loading model '{self.model_name}': {self.model_config.model_id}")
        
        self._pipe = pipeline(
            "text-generation",
            model=self.model_config.model_id,
            torch_dtype=self.model_config.torch_dtype,
            device_map=self.model_config.device_map,
            token=self.model_config.hf_token,
        )
        
        # Configure tokenizer for batch processing
        tokenizer = self._pipe.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.padding_side = "left"
        
        logger.info(f"Pipeline '{self.model_name}' initialized successfully")
    
    def cleanup(self) -> None:
        """Cleanup pipeline and free GPU memory."""
        if self._pipe is not None:
            logger.info(f"Cleaning up model '{self.model_name}'")
            
            # Delete pipeline components
            del self._pipe
            self._pipe = None
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
            
            logger.info(f"Model '{self.model_name}' cleaned up successfully")
    
    @property
    def pipe(self) -> Pipeline:
        """Get the initialized pipeline, raising if not ready."""
        if self._pipe is None:
            raise RuntimeError(
                "Pipeline not initialized. Call initialize() first."
            )
        return self._pipe
    
    def build_prompt(self, about_text: str) -> str:
        """Build chat-formatted prompt for a single profile text.
        
        Args:
            about_text: Profile "About Me" section text.
            
        Returns:
            Formatted prompt string ready for generation.
        """
        messages = build_profile_analysis_messages(about_text)
        return self.pipe.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    
    def generate_batch(self, about_texts: list[str]) -> list[str]:
        """Generate quality assessments for a batch of profiles.
        
        Args:
            about_texts: List of "About Me" section texts.
            
        Returns:
            List of generated JSON-formatted assessment strings.
        """
        prompts = [self.build_prompt(text) for text in about_texts]
        
        outputs = self.pipe(
            prompts,
            do_sample=self.generation_config.do_sample,
            num_beams=self.generation_config.num_beams,
            max_new_tokens=self.generation_config.max_new_tokens,
            batch_size=self.generation_config.batch_size,
            return_full_text=False,
            handle_long_generation=self.generation_config.handle_long_generation,
        )
        
        return [output[0]["generated_text"].strip() for output in outputs]
    
    def test_generation(self, test_prompt: str = "How old is the sun?") -> Any:
        """Run a simple test generation to verify pipeline works.
        
        Args:
            test_prompt: Simple test question.
            
        Returns:
            Generation output.
        """
        messages = [{"role": "user", "content": test_prompt}]
        return self.pipe(messages)