"""Setup configuration for WFA Profile Analyzer package."""

from setuptools import setup, find_packages

setup(
    name="wfa-profile-analyzer",
    version="0.1.0",
    description="AI-powered quality analysis for Workforce Australia job seeker profiles",
    author="Your Team",
    author_email="team@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "transformers==4.55.1",
        "huggingface_hub==0.35.3",
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "accelerate>=0.20.0",
        "triton==3.4",
        "safetensors==0.6.2",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "black>=23.0.0", "ruff>=0.1.0"],
    },
)