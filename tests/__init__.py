"""
AI Safety Guardrails Test Suite

Comprehensive testing framework for all package components.
"""

import os
import sys

# Add package root to path for testing
package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_root not in sys.path:
    sys.path.insert(0, package_root)