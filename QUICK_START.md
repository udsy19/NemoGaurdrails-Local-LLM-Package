# 🚀 Quick Start Guide - AI Safety Guardrails

**Confused by all the options? Start here!** This guide shows you exactly how to use the AI Safety Guardrails package in the simplest way possible.

## 📦 Installation

```bash
pip install ai-safety-guardrails
```

## 🎯 The Two Main Ways to Use This Package

### Option 1: The Simple Way (Recommended for Beginners)

**Just wrap your LLM function with `@safe_ai`:**

```python
from ai_safety_guardrails import safe_ai

@safe_ai()  # This protects your function automatically
def my_chatbot(user_input: str) -> str:
    # Your LLM code here
    return f"AI Response to: {user_input}"

# Use it normally - safety is automatic!
response = my_chatbot("Hello, what's my credit card number?")
print(response)  # Will be blocked if unsafe
```

### Option 2: The Control Way (For Advanced Users)

**Use SafetyGuard for explicit control:**

```python
from ai_safety_guardrails import SafetyGuard

# Create the guard
guard = SafetyGuard()

# Check if text is safe
result = guard.analyze_text("Some user input")

if result.is_safe:
    print("Text is safe!")
else:
    print(f"Blocked: {result.reason}")
```

## 🔥 Real Examples

### Example 1: Protect OpenAI Calls

```python
from ai_safety_guardrails import safe_ai
import openai

client = openai.OpenAI(api_key="your-key")

@safe_ai(detectors=["toxicity", "pii"])
def safe_openai_chat(message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content

# This is automatically protected!
answer = safe_openai_chat("What's my email address?")
```

### Example 2: Protect Any Function

```python
from ai_safety_guardrails import safe_ai

@safe_ai()
def my_ai_function(user_text: str) -> str:
    # Any AI processing here
    return f"Processed: {user_text}"

# Automatically blocks unsafe content
result = my_ai_function("Send me toxic content")  # Will be blocked
```

### Example 3: Manual Control

```python
from ai_safety_guardrails import SafetyGuard

guard = SafetyGuard()

def manual_protection(user_input: str) -> str:
    # Check input safety
    check = guard.analyze_text(user_input)
    
    if not check.is_safe:
        return "Sorry, I can't process that request."
    
    # Your AI logic here
    ai_response = call_your_ai_model(user_input)
    
    # Check output safety
    output_check = guard.analyze_text(ai_response)
    
    if not output_check.is_safe:
        return "I need to revise my response."
    
    return ai_response
```

## 🎛️ Simple Configuration

### Default Settings (Works for Most Cases)

```python
from ai_safety_guardrails import safe_ai

@safe_ai()  # Uses default settings
def my_function(text: str) -> str:
    return process_text(text)
```

### Custom Settings

```python
from ai_safety_guardrails import safe_ai

@safe_ai(
    detectors=["toxicity", "pii"],  # Only check these
    toxicity_threshold=0.8,         # Less sensitive
    block_message="Content not allowed"  # Custom message
)
def my_function(text: str) -> str:
    return process_text(text)
```

## 🔧 Available Detectors

| Detector | What It Blocks | Example |
|----------|---------------|---------|
| `toxicity` | Harmful, offensive content | Hate speech, insults |
| `pii` | Personal information | Email, phone, SSN |
| `prompt_injection` | Attempts to hack AI | "Ignore instructions" |
| `spam` | Promotional content | "Buy now!" |
| `topics` | Restricted topics | Violence, illegal content |

## 🚨 Common Use Cases

### 1. Chat Applications
```python
@safe_ai(detectors=["toxicity", "pii"])
def chatbot(message: str) -> str:
    return generate_response(message)
```

### 2. Content Moderation
```python
@safe_ai(detectors=["toxicity", "spam"])
def process_user_content(content: str) -> str:
    return publish_content(content)
```

### 3. AI Assistants
```python
@safe_ai()  # All detectors enabled
def ai_assistant(query: str) -> str:
    return generate_ai_response(query)
```

## 🛠️ Testing Your Setup

### Test the Package

```python
from ai_safety_guardrails import safe_ai

@safe_ai()
def test_function(text: str) -> str:
    return f"Echo: {text}"

# Test with safe content
print(test_function("Hello world"))  # Should work

# Test with unsafe content
print(test_function("Send me your password"))  # Should be blocked
```

### Command Line Testing

```bash
# Test specific detectors
ai-safety test --text "Hello world"

# Test with specific detector
ai-safety test --detectors toxicity --text "Test message"
```

## ❓ FAQ

**Q: Which approach should I use?**
A: Use `@safe_ai()` decorator - it's simpler and works for 90% of cases.

**Q: What if I want more control?**
A: Use `SafetyGuard` class for manual control over detection and responses.

**Q: How do I know what was blocked?**
A: The decorator will log blocked content. Use `SafetyGuard` for programmatic access to block reasons.

**Q: Can I customize the detectors?**
A: Yes! Pass `detectors=["toxicity", "pii"]` to only use specific ones.

**Q: Is this blocking too much content?**
A: Adjust thresholds: `@safe_ai(toxicity_threshold=0.8)` (higher = less sensitive)

## 🎯 Next Steps

1. **Start Simple**: Use `@safe_ai()` decorator
2. **Test It**: Try with your existing functions
3. **Customize**: Adjust detectors and thresholds as needed
4. **Scale Up**: Check out the full README.md for advanced features

## 📚 More Documentation

- [Complete README](README.md) - Full features and examples
- [Technical Documentation](TECHNICAL_README.md) - Architecture and models
- [GitHub Repository](https://github.com/udsy19/AI-Safety-Guardrails) - Source code and issues

---

**Still confused?** Open an issue on GitHub or email udayatejas2004@gmail.com