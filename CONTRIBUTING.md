# Contributing to Lifeboat

Thank you for your interest in contributing to Lifeboat! This document outlines the process for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/bniladridas/lifeboat.git
   cd lifeboat
   ```

3. Create a branch for your changes:
   ```bash
   git checkout -b your-branch-name
   ```

## Making Changes

1. Make your changes to the code or documentation
2. Follow the existing code style and conventions
3. Run tests to ensure your changes don't break anything:
   ```bash
   pip install -r requirements.txt
   pytest tests/
   ```

4. Update CHANGELOG.md with your changes

## Submitting a PR

1. Push your branch to your fork:
   ```bash
   git push origin your-branch-name
   ```

2. Open a Pull Request using our [PR template](.github/PULL_REQUEST_TEMPLATE/pull_request_template.md)

3. Ensure all CI checks pass

## Code Style

- Follow PEP 8 for Python code
- Use conventional commits: `[feat]`, `[fix]`, `[docs]`, `[chore]`, `[refactor]`
- Write docstrings for new functions

## Questions?

Open an issue for questions or discussions.
