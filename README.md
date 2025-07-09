# AI Safety Guardrails

[![Version](https://img.shields.io/badge/version-1.0.1-blue)](https://pypi.org/project/ai-safety-guardrails/)
[![Python](https://img.shields.io/badge/python-3.9+-green)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![PyPI Downloads](https://img.shields.io/pypi/dm/ai-safety-guardrails)](https://pypi.org/project/ai-safety-guardrails/)

🛡️ **Professional AI Safety Package** for protecting LLM applications with multiple detection capabilities, flexible APIs, and enterprise-grade features.

> 📋 **New to this package?** Start with the [**Quick Start Guide**](QUICK_START.md) - it shows exactly how to use the package in simple steps!

## 🚀 Quick Start

### Installation

```bash
pip install ai-safety-guardrails
```

### Basic Usage

```python
from ai_safety_guardrails import SafetyGuard

# Create safety guard with default detectors
guard = SafetyGuard()

# Protect your LLM interactions
result = await guard.protect(
    input_text="User input here",
    llm_function=your_llm_function  # Your LLM call
)

if result.blocked:
    print(f"Content blocked: {result.reason}")
else:
    print(f"Safe response: {result.response}")
```

### Decorator API (One-Line Protection)

```python
from ai_safety_guardrails import safe_ai

@safe_ai(detectors=["toxicity", "pii", "prompt_injection"])
async def my_chatbot(user_input: str) -> str:
    # Your LLM logic here
    return await call_openai(user_input)

# Automatically protected!
response = await my_chatbot("Hello, what's my credit card number?")
```

## 🔥 Core Features

### 🛡️ Multiple Safety Detectors
- **Toxicity Detection** - Blocks harmful, offensive, or inappropriate content
- **PII Protection** - Detects and blocks personal information (emails, phone numbers, SSNs)
- **Prompt Injection** - Prevents attempts to manipulate AI behavior
- **Spam Detection** - Filters promotional and unwanted content
- **Topic Filtering** - Blocks content from restricted categories
- **Fact Checking** - Identifies potential misinformation

### 🔧 Two Powerful APIs

#### 1. **Library API** (Explicit Control)
```python
from ai_safety_guardrails import SafetyGuard

guard = SafetyGuard(
    detectors=["toxicity", "pii"],
    thresholds={"toxicity": 0.8, "pii": 0.9}
)

result = await guard.analyze_text("Check this text")
if result["toxicity"].blocked:
    print("Toxic content detected!")
```

#### 2. **Decorator API** (Transparent Protection)
```python
@safe_ai(detectors=["pii", "prompt_injection"])
def protected_function(text):
    return process_with_llm(text)
```

## 🎯 Quick Examples

### OpenAI Integration

```python
from ai_safety_guardrails import SafetyGuard
import openai

client = openai.OpenAI(api_key="your-key")
guard = SafetyGuard(detectors=["toxicity", "pii"])

async def safe_chat(user_message: str) -> str:
    # Check input safety
    input_check = await guard.analyze_text(user_message)
    if any(result.blocked for result in input_check.values()):
        return "I can't process that request."
    
    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_message}]
    )
    
    # Check output safety
    ai_response = response.choices[0].message.content
    output_check = await guard.analyze_text(ai_response)
    
    if any(result.blocked for result in output_check.values()):
        return "I need to revise my response for safety."
    
    return ai_response
```

### Batch Processing

```python
from ai_safety_guardrails import SafetyGuard

guard = SafetyGuard()

texts = ["Hello world", "Buy now!!!", "Send me your SSN"]
results = await guard.batch_analyze(texts)

for i, text_results in enumerate(results):
    if any(r.blocked for r in text_results.values()):
        print(f"Text {i} blocked: {texts[i]}")
```

## 🔧 CLI Tools

### Test Detectors
```bash
# Test specific detectors
ai-safety test --detectors toxicity --text "This is a test"

# Test all detectors
ai-safety test --text "Check this content"
```

### Create Applications
```bash
# Generate a complete chat app
ai-safety create app my-chat --template chat --llm openai

# Generate Streamlit dashboard
ai-safety create app my-dashboard --template streamlit --llm ollama

# List available templates
ai-safety create list-templates
```

### Model Management
```bash
# Download models
ai-safety models download toxicity

# Check model status
ai-safety models status

# Clear model cache
ai-safety models clear
```

### Health Monitoring
```bash
# Check system health
ai-safety health

# Validate configuration
ai-safety validate-config config.yml
```

## ⚙️ Configuration

### YAML Configuration

```yaml
# ai_safety_config.yml
detectors:
  toxicity:
    enabled: true
    threshold: 0.8
    model: "martin-ha/toxic-comment-model"
  
  pii:
    enabled: true
    threshold: 0.9
    entities: ["EMAIL", "PHONE", "SSN", "CREDIT_CARD"]
  
  prompt_injection:
    enabled: true
    threshold: 0.7

performance:
  timeout: 30
  max_concurrent: 10
  cache_enabled: true

logging:
  level: INFO
  file: "safety.log"
```

### Programmatic Configuration

```python
from ai_safety_guardrails import SafetyGuard, DetectorConfig

# Configure individual detectors
toxicity_config = DetectorConfig(
    name="toxicity",
    threshold=0.85,
    enabled=True
)

guard = SafetyGuard(
    detectors=[toxicity_config, "pii", "spam"],
    config_file="my_config.yml"
)
```

## 🌟 Advanced Features

### Custom Actions

```python
from ai_safety_guardrails import SafetyGuard

def custom_action(result, text):
    if result.confidence > 0.9:
        log_security_incident(text, result)
        send_alert(f"High confidence detection: {result.reason}")

guard = SafetyGuard(
    detectors=["toxicity"],
    on_blocked=custom_action
)
```

### Performance Monitoring

```python
# Get performance metrics
metrics = guard.get_metrics()
print(f"Total requests: {metrics['total_requests']}")
print(f"Blocked: {metrics['blocked_requests']}")
print(f"Average processing time: {metrics['avg_processing_time']:.3f}s")
```

### Circuit Breaker Pattern

```python
guard = SafetyGuard(
    detectors=["toxicity", "pii"],
    circuit_breaker=True,
    max_failures=5,
    recovery_timeout=60
)
```

## 📊 LLM Integrations

### OpenAI

```python
from ai_safety_guardrails.integrations import OpenAIClient

client = OpenAIClient(api_key="your-key")
guard = SafetyGuard()

result = await guard.protect(
    input_text="User message",
    llm_function=client.generate
)
```

### Ollama (Local Models)

```python
from ai_safety_guardrails.integrations import OllamaClient

client = OllamaClient(host="localhost:11434")
guard = SafetyGuard()

result = await guard.protect(
    input_text="User message",
    llm_function=lambda text: client.generate("llama2", text)
)
```

### Generic Integration

```python
async def my_custom_llm(prompt: str) -> str:
    # Your custom LLM logic
    return "LLM response"

result = await guard.protect(
    input_text="User input",
    llm_function=my_custom_llm
)
```

## 🎨 Template System

Generate complete applications with one command:

```bash
# Available templates
ai-safety create list-templates
```

Templates include:
- **chat** - Interactive chat application with safety
- **api** - FastAPI server with safety endpoints
- **streamlit** - Web dashboard with real-time monitoring
- **notebook** - Jupyter notebook with examples

Each template includes:
- Complete working code
- Configuration files
- Requirements and setup instructions
- Documentation and examples

## 🏥 Production Features

### Health Monitoring
```python
health = await guard.health_check()
print(f"System healthy: {health['overall_healthy']}")
print(f"Detectors loaded: {health['detectors_loaded']}")
```

### Logging and Metrics
```python
# Automatic structured logging
guard = SafetyGuard(log_level="INFO")

# Custom metrics collection
guard.enable_metrics(export_interval=60)
```

### Graceful Degradation
```python
# Continue working even if some detectors fail
guard = SafetyGuard(
    detectors=["toxicity", "pii", "spam"],
    fail_open=True  # Don't block if detectors fail
)
```

## 🔧 Development

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python test_package_minimal.py

# Run with ML dependencies
python test_package.py
```

### Custom Detectors
```python
from ai_safety_guardrails.detectors import BaseDetector

class MyCustomDetector(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__("custom", **kwargs)
    
    async def analyze(self, text: str) -> DetectionResult:
        # Your custom logic
        return DetectionResult(
            blocked=False,
            confidence=0.5,
            reason="Custom analysis"
        )

# Use your custom detector
guard = SafetyGuard(detectors=[MyCustomDetector()])
```

## 📚 Documentation

- [Technical Documentation](TECHNICAL_README.md) - Internal architecture and ML models
- [API Reference](https://github.com/udsy19/AI-Safety-Guardrails/blob/main/README.md)
- [Examples](examples/) - Complete usage examples
- [Contributing](CONTRIBUTING.md) - Development guidelines

## 🤝 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/udsy19/AI-Safety-Guardrails/issues)
- **Email**: udayatejas2004@gmail.com
- **PyPI**: [Package homepage](https://pypi.org/project/ai-safety-guardrails/)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🌟 Why AI Safety Guardrails?

- **Production Ready**: Battle-tested with enterprise-grade features
- **Flexible**: Two APIs to fit any use case
- **Comprehensive**: Multiple detection types in one package
- **Performance**: Async/await throughout with intelligent caching
- **Easy Integration**: Works with any LLM provider
- **Extensible**: Custom detectors and actions
- **Observable**: Built-in monitoring and metrics

---

**Created by**: [Udaya Vijay Anand](mailto:udayatejas2004@gmail.com)
**Repository**: [GitHub](https://github.com/udsy19/AI-Safety-Guardrails)
**Package**: [PyPI](https://pypi.org/project/ai-safety-guardrails/)