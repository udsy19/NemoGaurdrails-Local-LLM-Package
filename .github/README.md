# GitHub Configuration

This directory contains GitHub-specific configuration files for the AI Safety Guardrails project.

## Workflows

### 🔄 CI/CD Pipelines

#### `safety_config.yml` - PyPI Publishing
- **Triggers**: Release published, manual workflow dispatch
- **Purpose**: Build and publish package to PyPI using trusted publisher OIDC
- **Environment**: `ai_safety_guardrails` (configured on PyPI)
- **Features**:
  - Automated testing across Python versions
  - Security scanning
  - Package verification
  - Trusted publisher authentication

#### `ci.yml` - Continuous Integration
- **Triggers**: Push to main/develop, pull requests
- **Purpose**: Code quality, testing, and validation
- **Features**:
  - Multi-OS testing (Ubuntu, Windows, macOS)
  - Python 3.9-3.12 compatibility
  - Code formatting (Black, isort)
  - Linting (flake8)
  - Type checking (mypy)
  - Security scanning (safety, bandit)
  - Documentation validation
  - Integration tests

#### `release.yml` - Release Management
- **Triggers**: Version tags (`v*.*.*`), manual dispatch
- **Purpose**: Automated release creation
- **Features**:
  - Changelog generation
  - GitHub release creation
  - Asset uploads (wheel, source distribution)
  - Version management

### 🏗️ Workflow Features

#### Security
- OIDC trusted publisher for PyPI
- Security scanning with multiple tools
- Dependency vulnerability checks
- No API tokens required

#### Testing
- Multi-platform compatibility
- Multiple Python versions
- Integration with actual ML models
- CLI command testing
- Template generation testing

#### Performance
- Parallel job execution
- Artifact caching
- Selective testing strategies
- Matrix exclusions for efficiency

## Issue Templates

### 🐛 Bug Report (`bug_report.yml`)
Structured template for reporting bugs with:
- Problem description
- Reproduction steps
- Environment details
- Component categorization
- Version information

### ✨ Feature Request (`feature_request.yml`)
Template for suggesting new features:
- Feature description
- Problem statement
- Proposed solution
- Use case examples
- Priority assessment

## Pull Request Template

### 📝 `PULL_REQUEST_TEMPLATE.md`
Comprehensive template covering:
- Change description
- Type classification
- Testing requirements
- Code quality checklist
- Documentation updates
- Security considerations
- Breaking changes

## Contributing Guide

### 🤝 `CONTRIBUTING.md`
Detailed contribution guidelines:
- Development setup
- Code standards
- Testing procedures
- Submission process
- Recognition system

## Dependency Management

### 🔄 `dependabot.yml`
Automated dependency updates:
- Weekly Python package updates
- GitHub Actions updates
- Grouped updates by category
- Security-focused configurations
- Automatic assignment and labeling

## Configuration Summary

| File | Purpose | Trigger |
|------|---------|----------|
| `safety_config.yml` | PyPI Publishing | Release, Manual |
| `ci.yml` | Quality Checks | Push, PR |
| `release.yml` | Release Creation | Tags, Manual |
| `bug_report.yml` | Bug Reports | Issue Creation |
| `feature_request.yml` | Feature Requests | Issue Creation |
| `PULL_REQUEST_TEMPLATE.md` | PR Guidelines | PR Creation |
| `CONTRIBUTING.md` | Contribution Guide | Documentation |
| `dependabot.yml` | Dependency Updates | Scheduled |

## Environment Requirements

### PyPI Trusted Publisher
To use the PyPI publishing workflow:

1. **Configure on PyPI**:
   - Project name: `ai_safety_guardrails`
   - Repository: `udsy19/NemoGaurdrails-Package`
   - Workflow: `safety_config.yml`
   - Environment: `ai_safety_guardrails`

2. **GitHub Environment**:
   - Create environment: `ai_safety_guardrails`
   - Configure protection rules (optional)
   - No secrets required (OIDC handles authentication)

### GitHub Secrets
No secrets are required for the main workflows due to OIDC authentication.

## Usage

### Triggering Workflows

```bash
# Create a release (triggers publishing)
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin v1.1.0

# Manual workflow dispatch
# Go to Actions tab → Select workflow → Run workflow
```

### Monitoring
- **Actions tab**: View workflow runs
- **Security tab**: View security alerts
- **Insights tab**: View dependency graph

## Maintenance

### Regular Tasks
- [ ] Review Dependabot PRs weekly
- [ ] Monitor workflow performance
- [ ] Update workflow versions quarterly
- [ ] Review security scan results
- [ ] Update issue templates as needed

### Troubleshooting

#### Common Issues
1. **OIDC Authentication Failures**
   - Verify PyPI trusted publisher configuration
   - Check environment name matches
   - Ensure workflow file name is correct

2. **Test Failures**
   - Check Python version compatibility
   - Verify all dependencies are installable
   - Review test matrix configuration

3. **Security Scan Alerts**
   - Review Dependabot PRs
   - Check security advisory database
   - Update vulnerable dependencies

For more information, see the [GitHub Actions documentation](https://docs.github.com/en/actions).