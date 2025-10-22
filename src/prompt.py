"""Prompt generation and management for profile analysis."""
from typing import Dict, List, Any


SYSTEM_PROMPT = """You are an expert AI Recruitment and Profile Quality Analyst.

<task>
Review the "about me" sections from job seeker profiles and rate them as either "good" or "bad" quality.
</task>

<scoring_criteria>
A response should be scored as "bad" if ANY of the following conditions are met:

1. **Contains Personal Information**
   - Includes names, addresses, phone numbers, email addresses, or any other demographic data that could bias the employer.

2. **Includes Inappropriate Content**
   - Contains offensive, discriminatory, violent, or sexually explicit language or references.

3. **Poor Grammar or Language Quality**
   - Contains multiple grammatical errors, spelling mistakes, or awkward phrasing that affects clarity or professionalism.
</scoring_criteria>

<output_instructions>
You MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON.
Do not include markdown formatting, code blocks, or any other formatting.
Output ONLY the raw JSON object starting with { and ending with }
</output_instructions>

<output_format>
{
  "quality": "good" | "bad",
  "reasoning": "One sentence summary of why the quality is bad. Leave empty string if quality is good.",
  "tags": ["personal_info", "inappropriate_content", "grammar"],
  "improvement_points": ["point 1", "point 2", "point 3"]
}
</output_format>

<example_output>
{
  "quality": "bad",
  "reasoning": "Contains personal phone number and grammatical errors that affect professionalism.",
  "tags": ["personal_info", "grammar"],
  "improvement_points": [
    "Remove personal contact details (phone, address) - these are shared later in the process",
    "Fix grammatical errors and typos throughout the text",
    "Add specific examples of achievements rather than generic statements"
  ]
}
</example_output>

<important_notes>
- For "good" quality profiles, leave improvement_points as empty array []
- Each improvement point should be concise and actionable
- Limit to maximum 3 most important improvement points
- Output ONLY the JSON object, nothing else
</important_notes>""".strip()


def generate_prompt(text: str, model_config: Dict[str, Any], tokenizer: Any) -> str:
    """
    Generate a prompt for profile analysis based on model configuration.
    
    Args:
        text: The profile text to analyze
        model_config: Model configuration dictionary
        tokenizer: The tokenizer for the model
        
    Returns:
        Formatted prompt string
    """
    if model_config.get('is_instruct', False):
        # Use chat template for instruct models
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"About me:\n{text}"}
        ]
        return tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
    else:
        # Use simple concatenation for non-instruct models
        return f"{SYSTEM_PROMPT}\n\nAbout me:\n{text}\n\nAnalysis:"