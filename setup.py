#!/usr/bin/env python3
"""
Setup configuration for AI Safety Guardrails package.
"""

from setuptools import setup, find_packages
import os
import sys

# Read version from package without importing
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "ai_safety_guardrails", "__init__.py")
    with open(version_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    raise RuntimeError("Cannot find version string")

def get_package_info():
    init_file = os.path.join(os.path.dirname(__file__), "ai_safety_guardrails", "__init__.py")
    info = {}
    with open(init_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                info["version"] = line.split("=")[1].strip().strip('"').strip("'")
            elif line.startswith("__author__"):
                info["author"] = line.split("=")[1].strip().strip('"').strip("'")
            elif line.startswith("__email__"):
                info["email"] = line.split("=")[1].strip().strip('"').strip("'")
            elif line.startswith("__description__"):
                info["description"] = line.split("=")[1].strip().strip('"').strip("'")
    return info

package_info = get_package_info()

# Read long description from README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return package_info["description"]

# Core dependencies required for basic functionality
CORE_REQUIREMENTS = [
    # Core Python packages
    "pyyaml>=6.0.1",
    "requests>=2.31.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.5.0",
    
    # AI/ML core dependencies
    "torch>=2.1.0",
    "transformers>=4.35.0",
    "sentence-transformers>=2.2.0",
    "tokenizers>=0.15.0",
    "huggingface-hub>=0.19.0",
    "safetensors>=0.4.0",
    
    # Data processing
    "numpy>=1.26.0",
    "pandas>=2.1.0",
    "scikit-learn>=1.3.0",
    
    # NLP
    "spacy>=3.7.0",
    
    # Logging
    "loguru>=0.7.0",
    
    # Template engine
    "jinja2>=3.1.0",
    
    # CLI
    "click>=8.0.0",
    "rich>=13.0.0",
    "typer>=0.9.0",
]

# Optional dependencies for different use cases
EXTRAS_REQUIRE = {
    # Web framework templates
    "templates": [
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "streamlit>=1.25.0",
        "websockets>=12.0",
        "python-multipart>=0.0.6",
    ],
    
    # GPU acceleration
    "gpu": [
        "torch[cuda]>=2.1.0",
    ],
    
    # Development tools
    "dev": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "isort>=5.12.0",
    ],
    
    # Documentation
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
    ],
    
    # Advanced monitoring
    "monitoring": [
        "prometheus-client>=0.19.0",
        "psutil>=5.9.0",
    ],
    
    # Full installation
    "full": [
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0", 
        "streamlit>=1.25.0",
        "websockets>=12.0",
        "python-multipart>=0.0.6",
        "prometheus-client>=0.19.0",
        "psutil>=5.9.0",
        "gradio>=4.0.0",
        "jupyter>=1.0.0",
    ],
}

# Console scripts - CLI commands available after installation
CONSOLE_SCRIPTS = [
    "ai-safety=ai_safety_guardrails.cli.main:main",
    "ai-safety-create=ai_safety_guardrails.cli.create:create_app_cli",
    "ai-safety-models=ai_safety_guardrails.cli.models:models_cli",
]

setup(
    name="ai-safety-guardrails",
    version=package_info["version"],
    author="Udaya Vijay Anand",
    author_email="udayatejas2004@gmail.com",
    description=package_info["description"],
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/udsy19/AI-Safety-Guardrails",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content :: Content Management System",
    ],
    python_requires=">=3.9",
    install_requires=CORE_REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    
    # Include non-Python files (templates, configs, etc.)
    include_package_data=True,
    
    # Define what files to include
    package_data={
        "ai_safety_guardrails": [
            "templates/**/*",
            "configs/**/*",
            "*.yml",
            "*.yaml",
            "*.json",
        ],
    },
    
    # Command line scripts
    entry_points={
        "console_scripts": CONSOLE_SCRIPTS,
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/udsy19/AI-Safety-Guardrails/issues",
        "Source": "https://github.com/udsy19/AI-Safety-Guardrails",
        "Documentation": "https://github.com/udsy19/AI-Safety-Guardrails/blob/main/README.md",
    },
    
    # Keywords for PyPI search
    keywords=[
        "ai", "safety", "guardrails", "llm", "machine-learning", 
        "security", "content-filtering", "toxicity", "pii", "openai", 
        "ollama", "transformers", "nemo-guardrails"
    ],
    
    # Zip safe
    zip_safe=False,
)