"""
Pytest configuration: set a dummy OPENAI_API_KEY so that
config/settings.py can load without a real key during unit tests.
"""

import os
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-unit-tests")
