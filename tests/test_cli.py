"""
Tests for AI Safety Guardrails CLI functionality.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from click.testing import CliRunner

from ai_safety_guardrails.cli.main import main, test_detectors, health_check
from ai_safety_guardrails.cli.create import create_app_cli
from ai_safety_guardrails.cli.models import models_cli
from ai_safety_guardrails.cli.config import config_cli


class TestMainCLI:
    """Test main CLI commands."""

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert "AI Safety Guardrails CLI" in result.output

    def test_cli_version(self):
        """Test CLI version command."""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        assert "1.0.1" in result.output  # Should show current version

    def test_test_detectors_basic(self):
        """Test basic detector testing."""
        with patch('ai_safety_guardrails.cli.main.SafetyGuard') as mock_guard:
            # Mock SafetyGuard
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.blocked = False
            mock_result.confidence = 0.1
            mock_result.reason = None
            
            mock_instance.analyze_text = AsyncMock(return_value={
                "toxicity": mock_result
            })
            mock_guard.return_value = mock_instance
            
            runner = CliRunner()
            result = runner.invoke(main, [
                'test',
                '--text', 'Hello world',
                '--detectors', 'toxicity'
            ])
            
            assert result.exit_code == 0
            assert "Testing detectors" in result.output

    def test_test_detectors_blocked_content(self):
        """Test detector testing with blocked content."""
        with patch('ai_safety_guardrails.cli.main.SafetyGuard') as mock_guard:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.blocked = True
            mock_result.confidence = 0.9
            mock_result.reason = "Test blocked content"
            
            mock_instance.analyze_text = AsyncMock(return_value={
                "toxicity": mock_result
            })
            mock_guard.return_value = mock_instance
            
            runner = CliRunner()
            result = runner.invoke(main, [
                'test',
                '--text', 'This should be blocked',
                '--detectors', 'toxicity'
            ])
            
            assert result.exit_code == 0
            assert "BLOCKED" in result.output or "blocked" in result.output

    def test_test_detectors_all(self):
        """Test detector testing with all detectors."""
        with patch('ai_safety_guardrails.cli.main.SafetyGuard') as mock_guard:
            mock_instance = MagicMock()
            mock_results = {}
            
            detectors = ["toxicity", "pii", "prompt_injection", "spam", "topics", "fact_check"]
            for detector in detectors:
                mock_result = MagicMock()
                mock_result.blocked = False
                mock_result.confidence = 0.1
                mock_result.reason = None
                mock_results[detector] = mock_result
            
            mock_instance.analyze_text = AsyncMock(return_value=mock_results)
            mock_guard.return_value = mock_instance
            
            runner = CliRunner()
            result = runner.invoke(main, [
                'test',
                '--text', 'Test all detectors'
            ])
            
            assert result.exit_code == 0

    def test_health_check(self):
        """Test health check command."""
        with patch('ai_safety_guardrails.cli.main.SafetyGuard') as mock_guard:
            mock_instance = MagicMock()
            mock_health = MagicMock()
            mock_health.status = "healthy"
            mock_health.uptime = 3600.0
            mock_health.memory_usage = 512.0
            mock_health.detectors = {"toxicity": MagicMock(status="healthy")}
            
            mock_instance.health_check = AsyncMock(return_value=mock_health)
            mock_guard.return_value = mock_instance
            
            runner = CliRunner()
            result = runner.invoke(main, ['health'])
            
            assert result.exit_code == 0
            assert "healthy" in result.output.lower()

    def test_init_config(self):
        """Test configuration initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.yml")
            
            runner = CliRunner()
            result = runner.invoke(main, [
                'init-config',
                '--output', config_path
            ])
            
            assert result.exit_code == 0
            assert os.path.exists(config_path)

    def test_validate_config(self, temp_config_file):
        """Test configuration validation."""
        runner = CliRunner()
        result = runner.invoke(main, [
            'validate-config',
            temp_config_file
        ])
        
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_config_invalid(self):
        """Test validation of invalid configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("invalid: yaml: content:")
            invalid_config = f.name
        
        try:
            runner = CliRunner()
            result = runner.invoke(main, [
                'validate-config',
                invalid_config
            ])
            
            # Should handle invalid config gracefully
            assert result.exit_code != 0 or "error" in result.output.lower()
        finally:
            os.unlink(invalid_config)


class TestCreateCLI:
    """Test create CLI commands."""

    def test_create_help(self):
        """Test create command help."""
        runner = CliRunner()
        result = runner.invoke(create_app_cli, ['--help'])
        assert result.exit_code == 0
        assert "Create AI Safety Guardrails applications" in result.output

    def test_list_templates(self):
        """Test listing available templates."""
        runner = CliRunner()
        result = runner.invoke(create_app_cli, ['list-templates'])
        
        assert result.exit_code == 0
        assert "chat" in result.output
        assert "api" in result.output
        assert "streamlit" in result.output
        assert "notebook" in result.output

    def test_list_llms(self):
        """Test listing available LLM providers."""
        runner = CliRunner()
        result = runner.invoke(create_app_cli, ['list-llms'])
        
        assert result.exit_code == 0
        assert "openai" in result.output
        assert "ollama" in result.output

    def test_create_chat_app(self):
        """Test creating a chat application."""
        with tempfile.TemporaryDirectory() as temp_dir:
            app_path = os.path.join(temp_dir, "test_chat_app")
            
            with patch('ai_safety_guardrails.cli.create.create_app') as mock_create:
                mock_create.return_value = True
                
                runner = CliRunner()
                result = runner.invoke(create_app_cli, [
                    'app',
                    'test_chat_app',
                    '--template', 'chat',
                    '--llm', 'openai',
                    '--output', temp_dir,
                    '--detectors', 'toxicity,pii'
                ])
                
                assert result.exit_code == 0
                mock_create.assert_called_once()

    def test_create_api_app(self):
        """Test creating an API application."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('ai_safety_guardrails.cli.create.create_app') as mock_create:
                mock_create.return_value = True
                
                runner = CliRunner()
                result = runner.invoke(create_app_cli, [
                    'app',
                    'test_api_app',
                    '--template', 'api',
                    '--llm', 'ollama',
                    '--output', temp_dir
                ])
                
                assert result.exit_code == 0
                mock_create.assert_called_once()

    def test_create_streamlit_app(self):
        """Test creating a Streamlit application."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('ai_safety_guardrails.cli.create.create_app') as mock_create:
                mock_create.return_value = True
                
                runner = CliRunner()
                result = runner.invoke(create_app_cli, [
                    'app',
                    'test_streamlit_app',
                    '--template', 'streamlit',
                    '--output', temp_dir
                ])
                
                assert result.exit_code == 0
                mock_create.assert_called_once()

    def test_create_notebook(self):
        """Test creating a Jupyter notebook."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('ai_safety_guardrails.cli.create.create_app') as mock_create:
                mock_create.return_value = True
                
                runner = CliRunner()
                result = runner.invoke(create_app_cli, [
                    'app',
                    'test_notebook',
                    '--template', 'notebook',
                    '--llm', 'openai',
                    '--output', temp_dir
                ])
                
                assert result.exit_code == 0
                mock_create.assert_called_once()

    def test_create_invalid_template(self):
        """Test creating app with invalid template."""
        runner = CliRunner()
        result = runner.invoke(create_app_cli, [
            'app',
            'test_app',
            '--template', 'invalid_template',
            '--llm', 'openai'
        ])
        
        assert result.exit_code != 0

    def test_create_invalid_llm(self):
        """Test creating app with invalid LLM provider."""
        runner = CliRunner()
        result = runner.invoke(create_app_cli, [
            'app',
            'test_app',
            '--template', 'chat',
            '--llm', 'invalid_llm'
        ])
        
        assert result.exit_code != 0


class TestModelsCLI:
    """Test models CLI commands."""

    def test_models_help(self):
        """Test models command help."""
        runner = CliRunner()
        result = runner.invoke(models_cli, ['--help'])
        assert result.exit_code == 0

    def test_models_list(self):
        """Test listing models."""
        with patch('ai_safety_guardrails.cli.models.ModelManager') as mock_manager:
            mock_instance = MagicMock()
            mock_instance.list_models.return_value = {
                "toxicity": {"status": "loaded", "path": "/path/to/model"},
                "pii": {"status": "not_loaded", "path": None}
            }
            mock_manager.return_value = mock_instance
            
            runner = CliRunner()
            result = runner.invoke(models_cli, ['list'])
            
            assert result.exit_code == 0
            assert "toxicity" in result.output
            assert "pii" in result.output

    def test_models_download(self):
        """Test downloading models."""
        with patch('ai_safety_guardrails.cli.models.ModelManager') as mock_manager:
            mock_instance = MagicMock()
            mock_instance.download_model = AsyncMock(return_value=True)
            mock_manager.return_value = mock_instance
            
            runner = CliRunner()
            result = runner.invoke(models_cli, [
                'download',
                'toxicity'
            ])
            
            assert result.exit_code == 0

    def test_models_clear_cache(self):
        """Test clearing model cache."""
        with patch('ai_safety_guardrails.cli.models.ModelManager') as mock_manager:
            mock_instance = MagicMock()
            mock_instance.clear_cache.return_value = True
            mock_manager.return_value = mock_instance
            
            runner = CliRunner()
            result = runner.invoke(models_cli, ['clear-cache'])
            
            assert result.exit_code == 0

    def test_models_info(self):
        """Test getting model information."""
        with patch('ai_safety_guardrails.cli.models.ModelManager') as mock_manager:
            mock_instance = MagicMock()
            mock_instance.get_model_info.return_value = {
                "name": "toxicity",
                "status": "loaded",
                "path": "/path/to/model",
                "size": "100MB",
                "last_used": "2025-01-01"
            }
            mock_manager.return_value = mock_instance
            
            runner = CliRunner()
            result = runner.invoke(models_cli, [
                'info',
                'toxicity'
            ])
            
            assert result.exit_code == 0
            assert "toxicity" in result.output


class TestConfigCLI:
    """Test config CLI commands."""

    def test_config_help(self):
        """Test config command help."""
        runner = CliRunner()
        result = runner.invoke(config_cli, ['--help'])
        assert result.exit_code == 0

    def test_config_show(self, temp_config_file):
        """Test showing configuration."""
        runner = CliRunner()
        result = runner.invoke(config_cli, [
            'show',
            temp_config_file
        ])
        
        assert result.exit_code == 0
        assert "detectors" in result.output

    def test_config_validate(self, temp_config_file):
        """Test validating configuration."""
        runner = CliRunner()
        result = runner.invoke(config_cli, [
            'validate',
            temp_config_file
        ])
        
        assert result.exit_code == 0

    def test_config_create_template(self):
        """Test creating configuration template."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "new_config.yml")
            
            runner = CliRunner()
            result = runner.invoke(config_cli, [
                'create-template',
                '--output', config_path,
                '--detectors', 'toxicity,pii'
            ])
            
            assert result.exit_code == 0
            assert os.path.exists(config_path)

    def test_config_list_detectors(self):
        """Test listing available detectors."""
        runner = CliRunner()
        result = runner.invoke(config_cli, ['list-detectors'])
        
        assert result.exit_code == 0
        assert "toxicity" in result.output
        assert "pii" in result.output

    def test_config_update(self, temp_config_file):
        """Test updating configuration."""
        runner = CliRunner()
        result = runner.invoke(config_cli, [
            'update',
            temp_config_file,
            '--detector', 'toxicity',
            '--threshold', '0.9'
        ])
        
        assert result.exit_code == 0

    def test_config_merge(self, temp_config_file):
        """Test merging configurations."""
        # Create second config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
detectors:
  spam:
    enabled: true
    threshold: 0.6
""")
            second_config = f.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_config = os.path.join(temp_dir, "merged_config.yml")
            
            try:
                runner = CliRunner()
                result = runner.invoke(config_cli, [
                    'merge',
                    temp_config_file,
                    second_config,
                    '--output', output_config
                ])
                
                assert result.exit_code == 0
                assert os.path.exists(output_config)
            finally:
                os.unlink(second_config)


class TestCLIIntegration:
    """Test CLI integration scenarios."""

    def test_full_workflow(self):
        """Test a complete CLI workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Create configuration
            config_path = os.path.join(temp_dir, "config.yml")
            
            runner = CliRunner()
            result = runner.invoke(main, [
                'init-config',
                '--output', config_path
            ])
            assert result.exit_code == 0
            
            # 2. Validate configuration
            result = runner.invoke(main, [
                'validate-config',
                config_path
            ])
            assert result.exit_code == 0
            
            # 3. Test detectors
            with patch('ai_safety_guardrails.cli.main.SafetyGuard') as mock_guard:
                mock_instance = MagicMock()
                mock_result = MagicMock()
                mock_result.blocked = False
                mock_result.confidence = 0.1
                
                mock_instance.analyze_text = AsyncMock(return_value={
                    "toxicity": mock_result
                })
                mock_guard.return_value = mock_instance
                
                result = runner.invoke(main, [
                    'test',
                    '--text', 'Hello world',
                    '--config', config_path
                ])
                assert result.exit_code == 0

    def test_error_handling(self):
        """Test CLI error handling."""
        runner = CliRunner()
        
        # Test with non-existent config file
        result = runner.invoke(main, [
            'validate-config',
            '/non/existent/config.yml'
        ])
        assert result.exit_code != 0

    def test_cli_with_environment_variables(self):
        """Test CLI with environment variables."""
        with patch.dict(os.environ, {
            'AI_SAFETY_LOG_LEVEL': 'DEBUG',
            'AI_SAFETY_CACHE_DIR': '/tmp/test_cache'
        }):
            runner = CliRunner()
            result = runner.invoke(main, ['--help'])
            assert result.exit_code == 0

    def test_cli_async_operations(self):
        """Test CLI handling of async operations."""
        with patch('ai_safety_guardrails.cli.main.SafetyGuard') as mock_guard:
            mock_instance = MagicMock()
            
            # Simulate async operation that takes time
            async def slow_analyze(text, detectors=None):
                import asyncio
                await asyncio.sleep(0.1)  # Simulate processing time
                return {"toxicity": MagicMock(blocked=False, confidence=0.1)}
            
            mock_instance.analyze_text = slow_analyze
            mock_guard.return_value = mock_instance
            
            runner = CliRunner()
            result = runner.invoke(main, [
                'test',
                '--text', 'Test async operation'
            ])
            
            assert result.exit_code == 0

    def test_cli_output_formats(self):
        """Test different CLI output formats."""
        with patch('ai_safety_guardrails.cli.main.SafetyGuard') as mock_guard:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.blocked = False
            mock_result.confidence = 0.1
            mock_result.reason = None
            
            mock_instance.analyze_text = AsyncMock(return_value={
                "toxicity": mock_result
            })
            mock_guard.return_value = mock_instance
            
            runner = CliRunner()
            
            # Test JSON output
            result = runner.invoke(main, [
                'test',
                '--text', 'Hello world',
                '--output-format', 'json'
            ])
            assert result.exit_code == 0
            
            # Test table output (default)
            result = runner.invoke(main, [
                'test',
                '--text', 'Hello world',
                '--output-format', 'table'
            ])
            assert result.exit_code == 0