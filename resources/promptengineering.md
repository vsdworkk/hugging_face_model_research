
# Prompt Engineering Best Practices

## Use XML Tags to Structure Your Prompts

When your prompts involve multiple components like context, instructions, and examples, XML tags can be a game-changer. They help AI parse your prompts more accurately, leading to higher-quality outputs.

**XML tip**: Use tags like `<instructions>`, `<example>`, and `<formatting>` to clearly separate different parts of your prompt. This prevents the AI from mixing up instructions with examples or context.

### Why use XML tags?

-   **Clarity**: Clearly separate different parts of your prompt and ensure your prompt is well structured.
-   **Accuracy**: Reduce errors caused by the AI misinterpreting parts of your prompt.
-   **Flexibility**: Easily find, add, remove, or modify parts of your prompt without rewriting everything.
-   **Parseability**: Having the AI use XML tags in its output makes it easier to extract specific parts of its response by post-processing.

### Tagging Best Practices

1.  **Be consistent**: Use the same tag names throughout your prompts, and refer to those tag names when talking about the content (e.g, `Using the contract in <contract> tags...`).
2.  **Nest tags**: You should nest tags `<outer><inner></inner></outer>` for hierarchical content.

**Power user tip**: Combine XML tags with other techniques like multishot prompting (`<examples>`) or chain of thought (`<thinking>`, `<answer>`). This creates super-structured, high-performance prompts.

## Giving the AI a Role with a System Prompt

When using an AI, you can dramatically improve its performance by using the `system` parameter to give it a role. This technique, known as role prompting, is the most powerful way to use system prompts with an AI.

The right role can turn an AI from a general assistant into your virtual domain expert!

### Why use role prompting?

-   **Enhanced accuracy**: In complex scenarios like legal analysis or financial modeling, role prompting can significantly boost AI performance.
-   **Tailored tone**: Whether you need a CFO’s brevity or a copywriter’s flair, role prompting adjusts the AI's communication style.
-   **Improved focus**: By setting the role context, the AI stays more within the bounds of your task’s specific requirements.

## Chain Complex Prompts for Stronger Performance

While these tips apply broadly to all AI models, you can find prompting tips specific to extended thinking models.

When working with complex tasks, an AI can sometimes drop the ball if you try to handle everything in a single prompt. Chain of thought (CoT) prompting is great, but what if your task has multiple distinct steps that each require in-depth thought?

Enter prompt chaining: breaking down complex tasks into smaller, manageable subtasks.

### Why chain prompts?

1.  **Accuracy**: Each subtask gets the AI’s full attention, reducing errors.
2.  **Clarity**: Simpler subtasks mean clearer instructions and outputs.
3.  **Traceability**: Easily pinpoint and fix issues in your prompt chain.

## Prompting Techniques for Extended Thinking

### Essential Tips for Long Context Prompts

-   **Put longform data at the top**: Place your long documents and inputs (~20K+ tokens) near the top of your prompt, above your query, instructions, and examples. This can significantly improve AI performance across all models.
-   **Structure document content and metadata with XML tags**: When using multiple documents, wrap each document in `<document>` tags with `<document_content>` and `<source>` (and other metadata) subtags for clarity.

### Use General Instructions First, Then Troubleshoot

An AI often performs better with high-level instructions to just think deeply about a task rather than step-by-step prescriptive guidance. The model’s creativity in approaching problems may exceed a human’s ability to prescribe the optimal thinking process.

For example, instead of:

**User:**

```text
Think through this math problem step by step: 
1. First, identify the variables
2. Then, set up the equation
3. Next, solve for x
...
```

Consider:

**User:**

```text
Please think about this math problem thoroughly and in great detail. 
Consider multiple approaches and show your complete reasoning.
Try different methods if your first approach doesn't work.
```

That said, the AI can still effectively follow complex structured execution steps when needed. The model can handle even longer lists with more complex instructions than previous versions. We recommend that you start with more generalized instructions, then read the AI’s thinking output and iterate to provide more specific instructions to steer its thinking from there.

### Maximizing Instruction Following with Extended Thinking

The AI shows significantly improved instruction following when extended thinking is enabled. The model typically:

1.  Reasons about instructions inside the extended thinking block
2.  Executes those instructions in the response

To maximize instruction following:

-   Be clear and specific about what you want
-   For complex instructions, consider breaking them into numbered steps that the AI should work through methodically
-   Allow the AI enough budget to process the instructions fully in its extended thinking

### Using Extended Thinking to Debug and Steer the AI’s Behavior

You can use the AI’s thinking output to debug the AI’s logic, although this method is not always perfectly reliable.

To make the best use of this methodology, we recommend the following tips:

-   We don’t recommend passing the AI’s extended thinking back in the user text block, as this doesn’t improve performance and may actually degrade results.
-   Prefilling extended thinking is explicitly not allowed, and manually changing the model’s output text that follows its thinking block is likely going to degrade results due to model confusion.

When extended thinking is turned off, standard `assistant` response text prefill is still allowed.

### Making the Best of Long Outputs and Longform Thinking

For dataset generation use cases, try prompts such as “Please create an extremely detailed table of…” for generating comprehensive datasets.

For use cases such as detailed content generation where you may want to generate longer extended thinking blocks and more detailed responses, try these tips:

-   Increase both the maximum extended thinking length AND explicitly ask for longer outputs
-   For very long outputs (20,000+ words), request a detailed outline with word counts down to the paragraph level. Then ask the AI to index its paragraphs to the outline and maintain the specified word counts

We do not recommend that you push the AI to output more tokens for outputting tokens’ sake. Rather, we encourage you to start with a small thinking budget and increase as needed to find the optimal settings for your use case.

### Have the AI Reflect on and Check Its Work

You can use simple natural language prompting to improve consistency and reduce errors:

1.  Ask the AI to verify its work with a simple test before declaring a task complete
2.  Instruct the model to analyze whether its previous step achieved the expected result
3.  For coding tasks, ask the AI to run through test cases in its extended thinking
