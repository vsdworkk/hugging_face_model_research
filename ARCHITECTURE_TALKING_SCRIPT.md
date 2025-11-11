# Talking Script: System Architecture Section

## Slide 4: Architecture Overview

**[Start by pointing to the diagram]**

"Let me walk you through the high-level architecture of our system. As you can see, we've designed this with a modular, component-based approach.

**[Point to config.yaml]**

"Everything starts with our configuration file - `config.yaml`. This is where we define which models we want to test, their settings, quantization levels, and processing parameters. The beauty of this approach is that you can enable or disable models, adjust batch sizes, or change quantization settings - all without touching a single line of code. This makes it easy for non-technical team members to experiment with different configurations.

**[Point to analyzer.py]**

"The heart of our system is `analyzer.py` - this is our orchestration engine. It reads the configuration, loads the models one at a time, manages the entire processing pipeline, handles batch processing, and parses the JSON outputs from the models. Think of it as the conductor of an orchestra - it coordinates everything but delegates specific tasks to specialized components.

**[Point to prompt.py and evaluate.py]**

"Below the analyzer, we have two specialized modules. On the left, `prompt.py` handles all the prompt engineering - it takes our system prompt and formats it correctly for each model type. On the right, `evaluate.py` handles all the metrics calculation and model comparison after we've got our predictions.

**[Emphasize the design]**

"The key architectural decision here was modularity. Each component has a single, well-defined responsibility. This separation of concerns makes the codebase maintainable, testable, and easy to extend. If we want to add a new model, we just update the config. If we want to change how we evaluate models, we only touch `evaluate.py`. This modular design has been crucial for our development process."

---

## Slide 5: Core Components

**[Transition]**

"Let me dive deeper into what each of these components actually does.

**[Point to analyzer.py]**

"Starting with `analyzer.py` - this is our orchestration engine. It's responsible for four main things:

First, **model pipeline management**. It loads models from Hugging Face, handles authentication, applies quantization if needed, and manages device placement - whether that's CPU or GPU.

Second, **batch processing logic**. Instead of processing profiles one at a time, which would be incredibly slow, we process them in batches. The batch size is configurable, so you can tune it based on your hardware constraints.

Third, **memory management**. This is critical - after each model finishes processing, we explicitly delete it and clear the GPU cache. This allows us to process multiple large models sequentially without running out of memory.

And fourth, **JSON parsing with fallback strategies**. Models don't always output perfect JSON - sometimes they include markdown, sometimes explanatory text. Our parser tries multiple strategies to extract the JSON, and gracefully handles failures.

**[Point to prompt.py]**

"Next, `prompt.py` handles all our prompt engineering. It contains our system prompt, which defines the task, scoring criteria, and output format. But more importantly, it handles the formatting differences between model types. Instruct models like Llama-3.2-Instruct use chat templates with system and user messages, while other models might use simple text concatenation. This module abstracts away those differences so the rest of the system doesn't need to care.

**[Point to evaluate.py]**

"`evaluate.py` is our metrics engine. It takes the model predictions and compares them against human labels. It calculates all the standard binary classification metrics - precision, recall, F1-score, accuracy - and generates comparison tables so we can easily see which model performs best. It also handles edge cases like missing predictions or invalid labels gracefully.

**[Point to config.yaml]**

"Finally, `config.yaml` is our configuration layer. It's a simple YAML file that defines which models to test, their quantization settings, batch sizes, token limits, and file paths. This configuration-driven approach means we can run different experiments without code changes - just modify the config and rerun."

---

## Slide 6: Data Flow

**[Transition]**

"Now let me walk you through the actual data flow - what happens when you run the system.

**[Point to step 1]**

"Step one: **Load Data**. We start with a CSV file that contains profile text - typically in an 'about_me' column - and human quality labels. These labels are our ground truth - they tell us which profiles are actually good or bad.

**[Point to step 2]**

"Step two: **Load Models**. Based on what's enabled in `config.yaml`, we load models from Hugging Face. We load them one at a time - not all at once - to manage memory. Each model gets configured with its quantization settings, device placement, and authentication tokens if needed.

**[Point to step 3]**

"Step three: **Generate Prompts**. For each profile, we combine our system prompt with the profile text. If it's an instruct model, we format it using the model's chat template. If it's not, we use simple concatenation. This gives us a properly formatted prompt that the model can understand.

**[Point to step 4]**

"Step four: **Batch Inference**. This is where the efficiency comes in. Instead of sending one profile at a time to the model, we send batches - typically 10 profiles at once. The model processes them in parallel, which is much faster. We configure the maximum number of tokens to generate, and we use deterministic sampling so results are reproducible.

**[Point to step 5]**

"Step five: **Parse Outputs**. The models return text, but we need structured JSON. Our parser tries multiple strategies - first a direct JSON parse, then regex extraction if that fails. If parsing fails completely, we handle it gracefully and continue processing other profiles.

**[Point to step 6]**

"Step six: **Evaluate Performance**. Once we have predictions from all models, we compare them against the human labels. We calculate precision, recall, F1-score, and accuracy for each model. Precision tells us how many of the profiles we flagged were actually bad - important for minimizing wasted reviewer time. Recall tells us how many bad profiles we caught - important for quality assurance.

**[Point to step 7]**

"Finally, step seven: **Export Results**. We save everything to timestamped CSV files. One file contains the full analysis with all predictions and reasoning. Another contains a comparison table showing which model performed best. These files are automatically saved to Azure Storage, so they're accessible for download and further analysis.

**[Wrap up]**

"This entire pipeline runs automatically once you execute the notebook. The system handles all the complexity - model loading, memory management, error handling - so you can focus on analyzing the results and choosing the best model for your use case."

---

## Slide 9: Challenge 3: Optimizing Unreliable Model Outputs

**[Transition from JSON parsing]**

"Now, JSON parsing helps us handle inconsistent outputs, but we also faced a broader challenge: how do you get reliable, high-quality outputs from smaller models in the first place?

**[Point to the problem]**

"Smaller models - we're talking about 1B to 3B parameter models here - they're less capable than their larger counterparts. They don't always follow instructions perfectly. They might ignore the JSON format requirement, produce low-quality reasoning, or give inconsistent results across similar inputs. And if you're using non-deterministic sampling, you might get different results every time you run the same profile, which makes evaluation really difficult.

**[Point to solution overview]**

"So we developed a multi-layered optimization strategy. We attack this problem from five different angles.

**[Point to deterministic sampling]**

"First, **deterministic sampling**. We set `do_sample=False` in our pipeline. This means the model always produces the same output for the same input. No randomness. This is crucial for evaluation - you need reproducible results to compare models fairly. It also helps smaller models because randomness can actually hurt their performance - they're more likely to produce bad outputs when sampling randomly.

**[Point to structured prompt engineering]**

"Second, **structured prompt engineering**. We don't just write a simple prompt. We use XML-style tags to structure the prompt - `<task>`, `<scoring_criteria>`, `<output_format>`. This helps models understand the structure better. We include detailed scoring criteria with examples. We have multiple reminders about the output format. And we include an example output right in the prompt - this is few-shot learning, showing the model exactly what we want.

**[Point to explicit output format]**

"Third, **explicit output format specification**. Look at our system prompt - we have an entire section called `<output_instructions>` that explicitly tells the model to output ONLY JSON, no markdown, no explanatory text, nothing else. We repeat this multiple times. And we include a complete example output showing the exact JSON structure we want. This redundancy is intentional - smaller models need multiple reminders to follow instructions.

**[Point to graceful degradation]**

"Fourth, **graceful degradation**. Even with all these strategies, sometimes models still fail. So we have our multi-strategy JSON parser that we covered earlier. If a model produces invalid output, we handle it gracefully - we don't crash, we just mark that prediction as invalid and continue processing. Our evaluation system filters out invalid predictions, so they don't skew our metrics.

**[Point to token limits]**

"Finally, **token limit management**. We set a maximum of 2000 new tokens. This prevents models from generating excessively long outputs, which increases the chance they'll drift away from the JSON format. It's a balance - we want enough tokens for complete reasoning, but not so many that the model loses focus.

**[Wrap up]**

"Together, these strategies work synergistically. Deterministic sampling gives us consistency. Structured prompts guide the model better. Multiple format reminders increase compliance. And graceful handling ensures the system keeps working even when models fail. The result? We can get reliable outputs even from smaller, less capable models - which is important because smaller models are faster, cheaper, and more accessible."

---

## Transition to Next Section

**[Bridge to challenges]**

"Now, building this architecture wasn't straightforward. We faced several significant challenges along the way, and I'd like to walk you through how we solved them..."

