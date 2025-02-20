# omaster

A Python package release tool that automates the entire release process with built-in quality checks and AI-powered features.

## Features

- Automated release pipeline with quality checks
- AI-powered commit message generation
- Semantic version management
- Code quality analysis
- Build and publish automation
- Git integration

## Installation

There are two main ways to use omaster:

### 1. Production Use (for users)

```bash
# Install from PyPI
uv pip install omaster

# Run the tool
uvx omaster [project_path]
```

### 2. Development Use (for contributors)

```bash
# Clone the repository
git clone https://github.com/yourusername/omaster.git
cd omaster

# Install dependencies
uv pip install -e .

# Run in development mode (recommended during development)
uv run -m omaster.main [project_path]
```

Note: `project_path` is optional and defaults to the current directory.

## Usage

### Why use `uv run` during development?

When working on omaster itself, always use `uv run -m omaster.main` because:
- Changes to code are immediately reflected
- No need to reinstall after each change
- Easier debugging and testing
- More reliable development workflow

### Release Pipeline

The tool runs a complete release pipeline that includes:

1. **Project Validation**
   - Required fields in pyproject.toml
   - Build system configuration
   - Entry points and dependencies
   - README.md content and structure

2. **Code Quality Checks**
   - Complexity analysis (cyclomatic, cognitive)
   - Dead code detection
   - Code similarity analysis
   - Style checks

3. **Change Analysis**
   - Uses OpenAI to analyze git changes
   - Generates meaningful commit messages
   - Suggests version bump type (major, minor, patch)

4. **Version Management**
   - Automatically bumps version based on changes
   - Follows semantic versioning

5. **Build & Publish**
   - Cleans old build files
   - Builds source and wheel distributions
   - Publishes to PyPI

6. **Git Integration**
   - Commits changes with AI-generated message
   - Pushes to remote repository

## Development Guide

### Common Development Tasks

1. **Running the Tool**
   ```bash
   # Always use this during development
   uv run -m omaster.main [project_path]
   ```

2. **Building and Testing**
   ```bash
   # Clean and build
   rm -rf dist/*
   uv build
   
   # Install and test the built package
   uv pip install dist/omaster-*.whl
   uvx omaster  # Test installed version
   ```

3. **Running Tests**
   ```bash
   uv run pytest
   uv run pytest --cov=omaster  # With coverage
   ```

### Environment Variables

- `OPENAI_API_KEY`: Required for AI features
  ```bash
  export OPENAI_API_KEY='your-api-key'
  ```

### Configuration

The tool can be configured using a `.omaster.yaml` file in your project root:

```yaml
ai:
  model: gpt-4o-mini  # or gpt-4o

quality:
  complexity:
    max_cyclomatic: 15
    max_cognitive: 20
    min_maintainability: 65
  similarity:
    min_lines: 6
    exact_match_threshold: 1.0
    ast_similarity_threshold: 0.7
```

Available configuration options:

- `ai.model`: AI model to use for analysis
  - `gpt-4o-mini`: Faster, more efficient model (default)
  - `gpt-4o`: More powerful model for complex analysis
- `quality`: Code quality thresholds and settings
  - `complexity`: Complexity analysis settings
  - `similarity`: Code similarity detection settings

### Error Handling

The tool uses a standardized error system with helpful messages. Each error includes:
- Error code and title
- Description of what went wrong
- Instructions on how to fix it
- Example solution when available

For example:
```
🚨 Error 🚨
Code: 600 - OpenAI API Key Missing

Description:
OPENAI_API_KEY environment variable not set

How to fix:
Set OPENAI_API_KEY environment variable

Example:
export OPENAI_API_KEY='your-api-key'