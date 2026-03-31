# Contributing to MicroPyzzotMet

Thank you for considering contributing to MicroPyzzotMet! We welcome contributions in many forms, including code, bug reports, feature requests, documentation, and examples.

## Reporting Issues

If you encounter a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/bare92/micropyzzotmet/issues). Include:

- A clear description of the problem or desired feature
- Steps to reproduce (for bugs)
- Relevant output, error messages, or logs
- Your environment details (Python version, OS, installed packages)

## Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/micropyzzotmet.git
   cd micropyzzotmet
   ```

3. Create a virtual environment and install the package in development mode:
   ```bash
   chmod +x setup_micropyzzotmet_env.sh
   ./setup_micropyzzotmet_env.sh
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

4. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. Make your changes in the `src/micropyzzotmet/` directory
2. Run tests to ensure your changes don't break existing functionality:
   ```bash
   pytest
   ```

3. Format your code using ruff:
   ```bash
   ruff check --fix src/
   ruff format src/
   ```

4. Commit your changes with clear, descriptive commit messages:
   ```bash
   git add .
   git commit -m "Describe your changes clearly"
   ```

5. Push to your fork and open a pull request on the main repository

## Pull Request Guidelines

- Include a clear description of the changes and why they are needed
- Reference any related issues using `#issue_number`
- Ensure all tests pass before submitting
- Add or update tests for new functionality
- Update documentation if your changes affect user-facing functionality
- Keep pull requests focused on a single issue or feature when possible

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes explaining their purpose, parameters, and return values
- Use type hints where appropriate

## Testing

All new code should include tests. Tests are located in the `tests/` directory and use pytest.

To run tests:
```bash
pytest
```

To run tests with coverage:
```bash
pytest --cov=micropyzzotmet tests/
```

## Documentation

When adding new features or changing existing behavior, please update the relevant documentation in `docs/source/`.

## Commit Message Guidelines

- Use the imperative mood ("Add feature" not "Added feature")
- Use clear, concise messages
- Reference related issues when applicable
- Keep the first line to 50 characters or less

## Questions?

Feel free to start a discussion or open an issue if you have questions about how to contribute, or if you're unsure about an appropriate way to implement something.

Thank you for contributing to MicroPyzzotMet!
