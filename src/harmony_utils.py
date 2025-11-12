"""Harmony utilities for gpt-oss model integration."""
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


def load_harmony_encoder():
    """Load the Harmony encoding for gpt-oss models."""
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


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
        .with_reasoning_effort(ReasoningEffort.HIGH)
        .with_conversation_start_date("2025-11-12")
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
    encoding = load_harmony_encoder()
    prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    stop_ids = encoding.stop_tokens_for_assistant_actions()
    return prompt_ids, stop_ids


def parse_harmony_response(completion_tokens: List[int]) -> Optional[str]:
    """
    Parse Harmony response tokens to extract final channel content.
    
    Args:
        completion_tokens: Generated token IDs from model
        
    Returns:
        Final response content or None if parsing fails
    """
    try:
        encoding = load_harmony_encoder()
        messages = encoding.parse_messages_from_completion_tokens(
            completion_tokens, Role.ASSISTANT
        )
        
        # Extract only final channel messages
        for message in messages:
            if hasattr(message, 'channel') and message.channel == 'final':
                return message.content
                
        return None
    except Exception:
        return None


def is_harmony_model(model_config: Dict[str, Any]) -> bool:
    """Check if model configuration indicates a Harmony model."""
    return model_config.get('is_harmony', False)
