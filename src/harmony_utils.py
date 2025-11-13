"""Harmony utilities for gpt-oss model integration."""
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ReasoningEffort
)

# Hardcoded Harmony settings for profile analysis
HARMONY_SETTINGS = {
    "reasoning_level": "high",
    "include_reasoning": False,
    "include_commentary": False,
    "safety_filter": True,
    "encoding": "HARMONY_GPT_OSS",
    "channels": {
        "analysis": True,
        "commentary": False,
        "final": True
    }
}

DEFAULT_CACHE_DIR = Path(
    os.environ.get("TIKTOKEN_CACHE_DIR", "/dbfs/FileStore/harmony_encodings")
).expanduser()


def ensure_harmony_cache_dir() -> Path:
    """Ensure a writable cache directory exists for Harmony vocab files."""
    cache_dir = Path(
        os.environ.get("TIKTOKEN_CACHE_DIR", str(DEFAULT_CACHE_DIR))
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
    return cache_dir




def build_harmony_conversation(system_prompt: str, user_text: str) -> Conversation:
    """
    Build a Harmony conversation from system prompt and user text.
    
    Args:
        system_prompt: The analysis instructions
        user_text: The profile text to analyze
        
    Returns:
        Harmony Conversation object
    """
    system_content = (
        SystemContent.new()
        .with_model_identity("You are ChatGPT, a large language model trained by OpenAI.")
        .with_reasoning_effort(ReasoningEffort.LOW)  # Faster for classification tasks
        .with_conversation_start_date("2025-11-12")
        .with_knowledge_cutoff("2024-06")
        .with_required_channels(["analysis", "final"])  # Only need final JSON output
    )
    
    developer_content = DeveloperContent.new().with_instructions(system_prompt)
    
    return Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, system_content),
        Message.from_role_and_content(Role.DEVELOPER, developer_content),
        Message.from_role_and_content(Role.USER, f"About me:\n{user_text}")
    ])


def render_harmony_prompt(conversation: Conversation) -> Tuple[List[int], List[int]]:
    """
    Render Harmony conversation to token IDs.
    
    Args:
        conversation: Harmony conversation object
        
    Returns:
        Tuple of (prompt_token_ids, stop_token_ids)
    """
    ensure_harmony_cache_dir()
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    stop_ids = encoding.stop_tokens_for_assistant_actions()
    return prompt_ids, stop_ids
 

def get_harmony_stop_tokens() -> List[int]:
    """Return the EOS/stop token IDs for Harmony assistant completions."""
    ensure_harmony_cache_dir()
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding.stop_tokens_for_assistant_actions()
 

def parse_harmony_response(completion_tokens: List[int]) -> Optional[str]:
    """
    Parse Harmony response tokens to extract final channel content.
    
    Args:
        completion_tokens: Generated token IDs from model
        
    Returns:
        Final response content as plain text or None if parsing fails
    """
    try:
        ensure_harmony_cache_dir()
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        messages = encoding.parse_messages_from_completion_tokens(
            completion_tokens, Role.ASSISTANT
        )
        
        # Extract only final channel messages
        for message in messages:
            if hasattr(message, 'channel') and message.channel == 'final':
                content = message.content
                
                # Handle TextContent objects - extract the actual text
                if hasattr(content, '__iter__') and not isinstance(content, str):
                    # If content is a list of TextContent objects
                    text_parts = []
                    for item in content:
                        if hasattr(item, 'text'):
                            text_parts.append(item.text)
                        else:
                            text_parts.append(str(item))
                    return ''.join(text_parts)
                elif hasattr(content, 'text'):
                    # If content is a single TextContent object
                    return content.text
                else:
                    # If content is already a string
                    return str(content)
                
        return None
    except Exception:
        return None


def is_harmony_model(model_config: Dict[str, Any]) -> bool:
    """Check if model configuration indicates a Harmony model."""
    return model_config.get('is_harmony', False)
