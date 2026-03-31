# Publishing MicroPyzzotMet to PyPI

This guide explains how to publish MicroPyzzotMet to PyPI (Python Package Index) so it can be installed via `pip install micropyzzotmet`.

## Prerequisites

1. **PyPI Account**: Create a free account at [https://pypi.org/account/register/](https://pypi.org/account/register/)

2. **Generate an API Token**: 
   - Log in to PyPI
   - Click your account avatar → "Account settings"
   - Scroll to "API tokens"
   - Click "Add API token" 
   - Name it (e.g., "micropyzzotmet-github")
   - Copy the token (starts with `pypi-`)

3. **Add GitHub Secret**:
   - Go to your GitHub repository Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Paste the API token
   - Click "Add secret"

## Initial Setup (One Time)

The GitHub Actions workflows are already configured in `.github/workflows/`:
- `publish.yml`: Automated publishing on release

## Publishing Process

### Step 1: Prepare for Release

Before publishing, ensure:
- All tests pass: `pytest`
- Code is formatted: `ruff format src/`
- Version is updated in `src/micropyzzotmet/__init__.py` (currently `0.1.0`)

### Step 2: Create a Git Tag and Release

```bash
# Update version in __init__.py if needed
# Commit the changes
git add .
git commit -m "Bump version to 0.1.0"

# Create a git tag
git tag v0.1.0

# Push to remote
git push origin main
git push origin v0.1.0
```

### Step 3: Create a GitHub Release

Option A: Via command line with GitHub CLI:
```bash
gh release create v0.1.0 --title "v0.1.0" --notes "Initial PyPI release"
```

Option B: Via GitHub web interface:
1. Go to your repository on GitHub
2. Click "Releases" on the right sidebar
3. Click "Create a new release"
4. Tag: `v0.1.0`
5. Release title: `v0.1.0`
6. Description: Add release notes
7. Click "Publish release"

This will automatically trigger the `publish.yml` workflow, which will:
- Build the distribution
- Upload to PyPI

### Step 4: Verify Publication

After a few minutes, check:
```bash
pip install --upgrade micropyzzotmet
```

Or visit: https://pypi.org/project/micropyzzotmet/

## Manual Publishing (Alternative)

If you prefer to publish manually instead of using GitHub Actions:

```bash
# Build the distribution
python -m build

# Check the build
twine check dist/*

# Upload to PyPI (you'll be prompted for credentials)
twine upload dist/*
```

## Conda-Forge Publication

Once published to PyPI, you can submit to conda-forge:

1. Visit https://github.com/conda-forge/staged-recipes
2. Fork the repository
3. Add a new recipe in `recipes/micropyzzotmet/meta.yaml`
4. Submit a pull request
5. conda-forge maintainers will review and publish

## Troubleshooting

- **Authentication fails**: Check that the API token is correctly added to GitHub Secrets
- **Build fails**: Run `python -m build` locally to identify issues
- **Package already exists**: Versions must be unique on PyPI

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- Patch: `0.1.1` - Bug fixes
- Minor: `0.2.0` - New features (backward compatible)
- Major: `1.0.0` - Breaking changes

## References

- [PyPI Documentation](https://pypi.org/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)
