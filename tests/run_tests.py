#!/usr/bin/env python3
"""
Test runner for AI Safety Guardrails with different test configurations.
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result."""
    if description:
        print(f"\n🔄 {description}")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Success: {description}")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ Failed: {description}")
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.stdout:
            print("STDOUT:", result.stdout)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="AI Safety Guardrails Test Runner")
    parser.add_argument(
        "--suite", 
        choices=["unit", "integration", "e2e", "performance", "all"],
        default="unit",
        help="Test suite to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--fast", 
        action="store_true",
        help="Skip slow tests"
    )
    parser.add_argument(
        "--models", 
        action="store_true",
        help="Include tests that require model downloads"
    )
    parser.add_argument(
        "--api-keys", 
        action="store_true",
        help="Include tests that require API keys"
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true",
        help="Run benchmark tests and save results"
    )
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=ai_safety_guardrails", "--cov-report=html"])
    
    # Add parallel execution if requested
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    # Configure test selection based on suite
    if args.suite == "unit":
        cmd.extend(["-m", "unit"])
        if args.fast:
            cmd.extend(["-m", "unit and not slow"])
    elif args.suite == "integration":
        cmd.extend(["-m", "integration"])
        if not args.api_keys:
            cmd.extend(["-m", "integration and not requires_api_keys"])
    elif args.suite == "e2e":
        cmd.extend(["-m", "e2e"])
    elif args.suite == "performance":
        cmd.extend(["-m", "slow"])
    elif args.suite == "all":
        # Run all tests with appropriate filtering
        if args.fast:
            cmd.extend(["-m", "not slow"])
        if not args.api_keys:
            cmd.extend(["-m", "not requires_api_keys"])
        if not args.models:
            cmd.extend(["-m", "not requires_models"])
    
    # Add benchmark options
    if args.benchmark:
        cmd.extend(["--benchmark-only", "--benchmark-save=benchmark_results"])
    
    # Add test directory
    cmd.append("tests/")
    
    # Set environment variables
    env = os.environ.copy()
    env["AI_SAFETY_TEST_MODE"] = "true"
    env["PYTHONPATH"] = str(project_root)
    
    if args.api_keys:
        print("🔑 Including tests that require API keys")
        print("   Make sure to set OPENAI_API_KEY_TEST, ANTHROPIC_API_KEY_TEST")
    
    if args.models:
        print("📦 Including tests that require model downloads")
        print("   This may take longer on first run")
    
    # Run the tests
    print(f"\n🧪 Running {args.suite} tests...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1


def run_specific_tests():
    """Run specific test configurations."""
    
    # Quick smoke test
    print("🚀 Running smoke tests...")
    smoke_cmd = [
        "python", "-m", "pytest", 
        "tests/test_detectors.py::TestBaseDetector::test_detector_initialization",
        "tests/test_safety_guard.py::TestSafetyGuardInitialization::test_safety_guard_basic_init",
        "-v"
    ]
    if not run_command(smoke_cmd, "Smoke tests"):
        return False
    
    # Core functionality tests
    print("\n🔧 Running core functionality tests...")
    core_cmd = [
        "python", "-m", "pytest",
        "tests/test_detectors.py",
        "tests/test_safety_guard.py",
        "-m", "unit and not slow",
        "-v"
    ]
    if not run_command(core_cmd, "Core functionality tests"):
        return False
    
    # CLI tests
    print("\n💻 Running CLI tests...")
    cli_cmd = [
        "python", "-m", "pytest",
        "tests/test_cli.py",
        "-v"
    ]
    if not run_command(cli_cmd, "CLI tests"):
        return False
    
    return True


def run_ci_tests():
    """Run tests suitable for CI environment."""
    print("🏗️ Running CI test suite...")
    
    ci_cmd = [
        "python", "-m", "pytest",
        "tests/",
        "-m", "unit or (integration and not requires_api_keys)",
        "--cov=ai_safety_guardrails",
        "--cov-report=xml",
        "--cov-fail-under=75",
        "-v"
    ]
    
    return run_command(ci_cmd, "CI test suite")


if __name__ == "__main__":
    # Check if running in CI mode
    if "--ci" in sys.argv:
        sys.exit(0 if run_ci_tests() else 1)
    elif "--smoke" in sys.argv:
        sys.exit(0 if run_specific_tests() else 1)
    else:
        sys.exit(main())