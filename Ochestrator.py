# Databricks notebook source
# CELL 1: Setup
%pip install -r /Workspace/Users/vthedataeng@gmail.com/wfa_profile_analyzer/requirements.txt
dbutils.library.restartPython()


# COMMAND ----------

# CELL 2: Imports and Setup

# Add repo to Python path
import sys
sys.path.insert(0, '/Workspace/Repos/vthedataeng@gmail.com/wfa_profile_analyzer')

# Standard imports
import os
import pandas as pd
from pathlib import Path

# Import our modules
from src.config import AppConfig, ModelConfig, GenerationConfig, DataConfig, EvaluationConfig, load_config
from src.model.pipeline import ProfileAnalysisPipeline
from src.processing.batch_processor import ProfileBatchProcessor, MultiModelBatchProcessor
from src.processing.json_parser import parse_model_outputs_to_dataframe
from src.evaluation.metrics import ModelEvaluator
from src.utils.logger import setup_logging

# Setup logging
setup_logging()

# COMMAND ----------

# CELL 3: Load Configuration

from pathlib import Path

# Set your HuggingFace token
os.environ["HF_TOKEN"] = "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"

# Load config from YAML file
config_path = Path('/Workspace/Repos/vthedataeng@gmail.com/wfa_profile_analyzer/config/config.yaml')
config = load_config(config_path)

# Get enabled models
enabled_models = config.get_enabled_models()

print("✅ Configuration loaded from YAML!")
print(f"\n📊 Enabled Models: {len(enabled_models)}")
for idx, model in enumerate(enabled_models, 1):
    print(f"   {idx}. {model.name}: {model.model_id}")

print(f"\n⚙️  Generation Settings:")
print(f"   Batch size: {config.generation.batch_size}")
print(f"   Max new tokens: {config.generation.max_new_tokens}")

# COMMAND ----------

# CELL 4: Display Model Configuration Details

print("\n" + "="*70)
print("MODEL CONFIGURATION SUMMARY")
print("="*70)

for model in config.models:
    status = "✅ ENABLED" if model.enabled else "❌ DISABLED"
    print(f"\n{status} - {model.name}")
    print(f"   Model ID: {model.model_id}")
    print(f"   Device Map: {model.device_map}")
    print(f"   Torch Dtype: {model.torch_dtype}")

print("\n" + "="*70)

# COMMAND ----------

# CELL 5: Initialize Multi-Model Processor

print("🔄 Initializing Multi-Model Processor...")

# Create multi-model processor
multi_processor = MultiModelBatchProcessor(config)

# Display summary
print("\n" + multi_processor.get_model_summary())

print("✅ Multi-Model Processor ready!")

# COMMAND ----------

# CELL 6: Test Single Model (Optional)

# If you want to test a single model first, uncomment this section:

# print("🧪 Testing single model...")
# 
# # Get first enabled model
# first_model = enabled_models[0]
# 
# # Initialize pipeline
# test_pipeline = ProfileAnalysisPipeline(
#     model_config=first_model,
#     generation_config=config.generation
# )
# test_pipeline.initialize()
# 
# # Test generation
# test_result = test_pipeline.test_generation("What is 2+2?")
# print(f"\n✅ Test result: {test_result}")
# 
# # Cleanup
# test_pipeline.cleanup()
# print("✅ Test complete!")

# COMMAND ----------

# CELL 7: Process Data with All Enabled Models

# Load your data
# Replace with your actual data path
# data_path = "/dbfs/path/to/your/profiles.csv"
# df = pd.read_csv(data_path)

# Example: Create sample data for testing
sample_data = pd.DataFrame({
    'about_me': [
        "I am a software engineer with 5 years of experience in Python and Java.",
        "Passionate about data science and machine learning. Looking for new opportunities.",
        "Hi",  # Low quality example
        "Experienced project manager seeking remote work in tech industry.",
    ]
})

print(f"📊 Processing {len(sample_data)} profiles through {len(enabled_models)} model(s)...")

# Process through all enabled models
result_df = multi_processor.process_dataframe(
    sample_data,
    input_column='about_me',
    parse_outputs=True  # This will create separate columns for each model's outputs
)

print("\n✅ Processing complete!")
print(f"\n📊 Result columns: {list(result_df.columns)}")

# COMMAND ----------

# CELL 8: View Results

# Display results
print("\n" + "="*70)
print("PROCESSING RESULTS")
print("="*70)

# Show the dataframe
display(result_df)

# COMMAND ----------

# CELL 9: Compare Model Outputs (if multiple models enabled)

if len(enabled_models) > 1:
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # Get flag columns from each model
    flag_columns = [col for col in result_df.columns if col.endswith('_flag')]
    
    if flag_columns:
        print("\nFlag Predictions by Model:")
        comparison_df = result_df[['about_me'] + flag_columns].copy()
        display(comparison_df)
        
        # Count agreement between models
        if len(flag_columns) >= 2:
            print("\n📊 Model Agreement Analysis:")
            for i, row_idx in enumerate(result_df.index):
                flags = [result_df.loc[row_idx, col] for col in flag_columns]
                unique_flags = set([f for f in flags if pd.notna(f)])
                agreement = len(unique_flags) <= 1  # All models agree if only 1 unique value
                status = "✅ AGREE" if agreement else "⚠️  DISAGREE"
                print(f"   Profile {i+1}: {status} - Flags: {flags}")
else:
    print("\n💡 Only one model enabled. Enable more models in config.yaml to see comparisons!")

# COMMAND ----------

# CELL 10: Save Results (Optional)

# Save results to CSV
# output_path = "/dbfs/path/to/output/results.csv"
# result_df.to_csv(output_path, index=False)
# print(f"✅ Results saved to {output_path}")

# Or save to Delta table
# result_df.write.format("delta").mode("overwrite").saveAsTable("wfa_profile_analysis_results")
# print("✅ Results saved to Delta table")

print("💾 Ready to save results when you uncomment the save code above!")

# COMMAND ----------

# CELL 11: Model Performance Summary

print("\n" + "="*70)
print("PROCESSING SUMMARY")
print("="*70)

print(f"\n✅ Successfully processed {len(result_df)} profiles")
print(f"✅ Used {len(enabled_models)} model(s)")

for model in enabled_models:
    output_col = f"about_me_processed_{model.name}"
    flag_col = f"ai_{model.name}_flag"
    
    if output_col in result_df.columns:
        non_null = result_df[output_col].notna().sum()
        print(f"\n📊 Model: {model.name}")
        print(f"   Processed: {non_null}/{len(result_df)} profiles")
        
        if flag_col in result_df.columns:
            flag_counts = result_df[flag_col].value_counts()
            print(f"   Flag distribution:")
            for flag, count in flag_counts.items():
                print(f"      {flag}: {count}")

print("\n" + "="*70)
print("🎉 ALL DONE!")
print("="*70)
