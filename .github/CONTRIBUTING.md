# Contributing to AI Safety Guardrails

Thank you for your interest in contributing to AI Safety Guardrails! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

This project follows a standard code of conduct. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic knowledge of AI safety concepts

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/NemoGaurdrails-Package.git
   cd NemoGaurdrails-Package
   ```

## Development Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install development dependencies
pip install -e .[dev,full]

# Install spaCy model
python -m spacy download en_core_web_sm

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Installation

```bash
# Run tests
python test_package_minimal.py

# Test CLI
ai-safety --version
ai-safety test --detectors spam --text "Hello world"
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-detector`
- `fix/memory-leak-in-toxicity`
- `docs/update-installation-guide`
- `refactor/improve-error-handling`

### Code Style

We use the following tools for code quality:

```bash
# Format code
black ai_safety_guardrails/
isort ai_safety_guardrails/

# Lint code
flake8 ai_safety_guardrails/

# Type checking
mypy ai_safety_guardrails/
```

### Code Guidelines

1. **Follow PEP 8** for Python code style
2. **Use type hints** for all function parameters and return values
3. **Write docstrings** for all public functions and classes
4. **Add tests** for new functionality
5. **Update documentation** when adding features
6. **Keep functions focused** - one responsibility per function
7. **Use async/await** for I/O operations

### Adding New Detectors

To add a new detector:

1. Create a new file in `ai_safety_guardrails/detectors/`
2. Inherit from `BaseDetector`
3. Implement required methods:
   ```python
   from .base_detector import BaseDetector, DetectionResult
   
   class MyDetector(BaseDetector):
       def __init__(self, **kwargs):
           super().__init__(name="my_detector", **kwargs)
       
       async def load_model(self):
           # Load your model here
           pass
       
       async def detect(self, text: str, context: dict = None) -> DetectionResult:
           # Implement detection logic
           return DetectionResult(
               blocked=False,
               confidence=0.0,
               reason="Safe content"
           )
   ```
4. Add to `__init__.py` and detector registry
5. Add configuration to default config
6. Write tests

## Testing

### Running Tests

```bash
# Quick tests (no ML models)
python test_package_minimal.py

# Full tests (with ML models)
python examples/basic_usage.py

# Run pytest if available
pytest tests/ -v

# Run with coverage
pytest --cov=ai_safety_guardrails --cov-report=html
```

### Writing Tests

1. **Unit tests** for individual components
2. **Integration tests** for end-to-end functionality
3. **Performance tests** for critical paths
4. **Mock external dependencies** when possible

Example test:

```python
import pytest
from ai_safety_guardrails import SafetyGuard, DetectorConfig

@pytest.mark.asyncio
async def test_spam_detection():
    guard = SafetyGuard(detectors=[DetectorConfig("spam")])
    
    # Test spam content
    result = await guard.analyze_text("BUY NOW!!! LIMITED TIME!!!")
    assert result["spam"].blocked
    
    # Test normal content
    result = await guard.analyze_text("Hello, how are you?")
    assert not result["spam"].blocked
    
    await guard.cleanup()
```

## Submitting Changes

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** following the guidelines above

3. **Test your changes**:
   ```bash
   python test_package_minimal.py
   black ai_safety_guardrails/
   flake8 ai_safety_guardrails/
   ```

4. **Commit with clear messages**:
   ```bash
   git add .
   git commit -m "Add new spam detection patterns
   
   - Add support for cryptocurrency spam
   - Improve detection accuracy by 15%
   - Add tests for new patterns"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/my-new-feature
   ```

6. **Create a Pull Request** on GitHub

### Pull Request Guidelines

**Title**: Clear, descriptive title

**Description**: Include:
- What changes were made
- Why the changes were necessary
- How to test the changes
- Any breaking changes
- Related issues

**Checklist**:
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changelog entry added (if applicable)
- [ ] No breaking changes (or clearly documented)

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Creating a Release

1. **Update version** in `ai_safety_guardrails/__init__.py`
2. **Update CHANGELOG.md** with new features and fixes
3. **Create a tag**:
   ```bash
   git tag -a v1.1.0 -m "Release version 1.1.0"
   git push origin v1.1.0
   ```
4. **GitHub Actions** will automatically:
   - Run all tests
   - Build the package
   - Create a GitHub release
   - Publish to PyPI (if configured)

## Contributing Areas

We welcome contributions in these areas:

### 🔍 New Detectors
- Bias detection
- Emotion analysis
- Code injection detection
- Image safety analysis
- Audio content analysis

### 🚀 Performance Improvements
- Model optimization
- Caching strategies
- Batch processing
- Memory usage reduction

### 🔗 Integrations
- New LLM providers
- Vector databases
- Monitoring systems
- CI/CD platforms

### 📚 Documentation
- Tutorial improvements
- API documentation
- Example applications
- Best practices guides

### 🧪 Testing
- Test coverage improvement
- Performance benchmarks
- Integration tests
- Edge case testing

### 🎨 Templates
- New application templates
- Industry-specific examples
- Framework integrations

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community discussion
- **Email**: udayatejas2004@gmail.com for direct contact

## Recognition

Contributors will be recognized in:
- GitHub contributor list
- Release notes
- Documentation acknowledgments

Thank you for contributing to AI Safety Guardrails! Your efforts help make AI applications safer for everyone.