#!/usr/bin/env python3
"""
Quick test to verify the AI Safety Guardrails package is working.
"""

import asyncio
import sys
from pathlib import Path

# Add the package to path for testing
sys.path.insert(0, str(Path(__file__).parent))

async def test_imports():
    """Test that all main components can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test main imports
        from ai_safety_guardrails import SafetyGuard, DetectorConfig, safe_ai
        print("✅ Main imports successful")
        
        # Test detector imports
        from ai_safety_guardrails.detectors import (
            ToxicityDetector, PIIDetector, PromptInjectionDetector,
            TopicsDetector, FactCheckDetector, SpamDetector
        )
        print("✅ Detector imports successful")
        
        # Test CLI imports
        from ai_safety_guardrails.cli import main
        print("✅ CLI imports successful")
        
        # Test template imports
        from ai_safety_guardrails.templates import create_app
        print("✅ Template imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_basic_functionality():
    """Test basic functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from ai_safety_guardrails import SafetyGuard, DetectorConfig
        
        # Create a simple guard
        guard = SafetyGuard(detectors=[
            DetectorConfig("prompt_injection"),  # Pattern-based, no model needed
            DetectorConfig("spam")  # Pattern-based, no model needed
        ])
        
        print("✅ SafetyGuard created successfully")
        
        # Test text analysis
        results = await guard.analyze_text("Hello world, this is a test message")
        print(f"✅ Text analysis successful: {len(results)} detectors ran")
        
        # Test protection (with mock LLM)
        async def mock_llm(text: str) -> str:
            return f"Mock response to: {text}"
        
        result = await guard.protect(
            input_text="Hello world",
            llm_function=mock_llm
        )
        
        print(f"✅ Protection test successful: blocked={result.blocked}")
        
        # Test health check
        health = await guard.health_check()
        print(f"✅ Health check successful: healthy={health['overall_healthy']}")
        
        # Cleanup
        await guard.cleanup()
        print("✅ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_decorator():
    """Test the decorator API."""
    print("\n🧪 Testing decorator API...")
    
    try:
        from ai_safety_guardrails import safe_ai
        
        @safe_ai(detectors=["spam"], shared_instance=False)
        def test_function(message: str) -> str:
            return f"Processed: {message}"
        
        result = test_function("Hello world")
        print(f"✅ Decorator test successful: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Decorator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_configuration():
    """Test configuration system."""
    print("\n🧪 Testing configuration...")
    
    try:
        from ai_safety_guardrails.core.detector_config import SafetyConfig, DetectorConfig
        
        # Test default config
        config = SafetyConfig()
        print("✅ Default config created")
        
        # Test config validation
        issues = config.validate()
        print(f"✅ Config validation: {len(issues)} issues found")
        
        # Test detector config
        detector_config = DetectorConfig("toxicity", threshold=0.8)
        print(f"✅ Detector config created: {detector_config}")
        
        # Test string parsing
        parsed_config = DetectorConfig.from_string("pii:high")
        print(f"✅ String parsing successful: {parsed_config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def run_tests():
    """Run all tests."""
    print("🛡️  AI Safety Guardrails - Package Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Decorator API", test_decorator),
        ("Configuration", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            success = await test_func()
            if success:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Package is working correctly.")
        print("\nNext steps:")
        print("1. Install the package: pip install -e .")
        print("2. Try the examples: python examples/basic_usage.py")
        print("3. Create an app: ai-safety create my-app")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)