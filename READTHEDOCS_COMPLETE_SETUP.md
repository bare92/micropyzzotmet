# ReadTheDocs Documentation - Complete Setup Guide

## ✅ What Has Been Done

Your MicroPyzzotMet documentation has been enhanced and is ready to be hosted on ReadTheDocs. Here's what was improved:

### 1. **Enhanced `.readthedocs.yml` Configuration**

Improvements:
- ✅ Explicit Python 3.11 environment specification
- ✅ Development dependencies installation (`[dev]` extras)
- ✅ Multiple documentation formats (HTML, PDF, EPUB, HTML Zip)
- ✅ Search configuration for better discoverability
- ✅ GitHub webhook comments on pull requests
- ✅ Better error handling

**File**: `.readthedocs.yml`

### 2. **Improved Sphinx Configuration**

Enhancements to `docs/source/conf.py`:

**Theme Customization**:
- ✅ ReadTheDocs RTD theme with professional appearance
- ✅ GitHub integration (users can edit docs on GitHub)
- ✅ Sticky sidebar navigation
- ✅ Professional blue header (#2980B9)

**Better Autodoc**:
- ✅ Type hints in documentation
- ✅ Cross-references to numpy, xarray, pandas
- ✅ Copy code button in examples
- ✅ Automatic intersphinx linking

**Module Mocking**:
- ✅ All heavy scientific dependencies properly mocked
- ✅ Prevents "module not found" errors during build
- ✅ Allows documentation to build without full GDAL/compilation

### 3. **Added Documentation Dependencies**

Updated `pyproject.toml` with:
```python
docs = [
  "sphinx>=7.0",
  "sphinx-rtd-theme>=1.3",
  "sphinx-autodoc-typehints>=1.24",
  "sphinx-copybutton>=0.5"
]
```

Now users can install doc dependencies with:
```bash
pip install -e ".[docs]"
```

### 4. **Complete Setup Guide**

Created `READTHEDOCS_SETUP.md` with:
- ✅ Step-by-step account creation
- ✅ Repository import instructions
- ✅ Webhook configuration
- ✅ Local testing guide
- ✅ Customization options (logo, favicon)
- ✅ Troubleshooting common issues
- ✅ Advanced features guide

## 🚀 Next Steps to Go Live

### Step 1: Commit Your Changes

```bash
cd /home/riccardo/Documents/Pubblications/micropyzzotmet/micropyzzotmet

# Add all new/modified files
git add .readthedocs.yml docs/source/conf.py pyproject.toml READTHEDOCS_SETUP.md

# Commit
git commit -m "Improve ReadTheDocs configuration and documentation setup"

# Push to GitHub
git push origin main  # or your current branch
```

### Step 2: Create ReadTheDocs Account & Import Project

Follow the detailed instructions in [`READTHEDOCS_SETUP.md`](./READTHEDOCS_SETUP.md):

1. Go to https://readthedocs.org
2. Sign up with GitHub
3. Import repository: search for `micropyzzotmet`
4. Complete import (takes ~2-5 minutes for first build)

### Step 3: Verify Build Success

Visit: **https://micropyzzotmet.readthedocs.io** (after import)

You should see:
- Professional blue header ✅
- Sidebar with navigation ✅
- All sections: Getting Started, Usage, Methods, API ✅
- Search box ✅
- PDF/EPUB download options ✅

## 📊 Current Documentation Structure

Your ReadTheDocs site will include:

```
📖 MicroPyzzotMet Documentation
├── 🏠 Home
├── 📚 Getting Started
│   ├── Installation
│   └── Quickstart
├── 💼 Usage
│   ├── Configuration Files
│   └── Running Downscaling
├── 🔬 Methods
│   ├── Temperature Downscaling
│   ├── Radiation Downscaling
│   ├── Humidity Downscaling
│   ├── Precipitation Downscaling
│   ├── Longwave Downscaling
│   └── Wind Downscaling
├── 📖 API Reference
│   ├── get_era5_land
│   ├── main_micromet
│   ├── downscaling_variables
│   └── utils
└── 📝 Changelog
```

## 🛠 Local Testing Before Going Live

To test the documentation locally (before committing):

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Navigate to docs directory
cd docs

# Build HTML documentation
make clean
make html

# View in browser
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
firefox build/html/index.html  # Firefox on any OS
```

## 🎨 Customization Options

### Add a Project Logo

1. Place logo at `docs/source/_static/logo.png` (200x50 px recommended)
2. Uncomment in `docs/source/conf.py`:
   ```python
   html_logo = "_static/logo.png"
   ```

### Add GitHub Edit Links

Already configured! Users can click "Edit on GitHub" to suggest changes.

### Enable PDF Generation

Already enabled! Users can download PDF from any page (button in bottom left).

## 📱 Features Available After Going Live

Once on ReadTheDocs:

### ✅ Automatic Updates
Every push to GitHub automatically rebuilds documentation

### ✅ Multiple Versions
Documentation for each git tag (v0.1.0, v0.2.0, etc.)

### ✅ Pull Request Previews
Each PR gets a live documentation preview

### ✅ Search
Full-text search across all documentation pages

### ✅ Analytics
Track which pages are most visited

### ✅ Mobile Friendly
Responsive design works on phones, tablets, desktops

## 🔗 Useful URLs After Setup

| URL | Purpose |
|-----|---------|
| `https://micropyzzotmet.readthedocs.io` | Latest version |
| `https://micropyzzotmet.readthedocs.io/en/v0.1.0/` | Specific version |
| `https://micropyzzotmet.readthedocs.io/en/latest/` | Development version |
| `https://readthedocs.org/projects/micropyzzotmet/` | Admin dashboard |

## 🐛 Build Troubleshooting

If documentation fails to build on ReadTheDocs:

1. **Check Build Log**: Admin → Builds → Select build → View logs
2. **Common Issue**: Missing imports are usually already mocked
3. **Local Test**: Run `make html` locally to catch issues early
4. **Check Syntax**: Verify reStructuredText syntax in `.rst` files

### If Build Still Fails

Contact ReadTheDocs support (excellent!) or check:
- Ensure all `.rst` files have correct syntax
- Verify `sys.path` in `conf.py` is correct
- Check that all mock modules are listed

## 📚 Documentation Best Practices

1. **Keep examples current**: Update as code changes
2. **Use consistent style**: Follow existing documentation patterns
3. **Add docstrings**: Both functions and classes should have docstrings
4. **Link between pages**: Use `:ref:` for cross-references
5. **Include diagrams**: ASCII art or images help explain concepts

## ✅ Verification Checklist

Before considering this complete, verify:

- [ ] Changes committed and pushed to GitHub
- [ ] ReadTheDocs account created
- [ ] Repository imported on ReadTheDocs
- [ ] First build completed (check Builds tab)
- [ ] Documentation visible at https://micropyzzotmet.readthedocs.io
- [ ] All sections render correctly
- [ ] Search works
- [ ] PDF download available
- [ ] GitHub webhook configured (Settings → Webhooks)

## 🎯 Full Setup Command Quick Reference

```bash
# 1. Go to project directory
cd /home/riccardo/Documents/Pubblications/micropyzzotmet/micropyzzotmet

# 2. Test locally first
pip install -e ".[docs]"
cd docs && make html

# 3. Commit and push
cd ..
git add .readthedocs.yml docs/ pyproject.toml READTHEDOCS_SETUP.md
git commit -m "Improve ReadTheDocs setup"
git push origin main

# 4. Go to https://readthedocs.org and import repository
# 5. Wait for first build to complete
# 6. Visit https://micropyzzotmet.readthedocs.io

# That's it! Documentation is now live and auto-updates on every push.
```

## 📞 Get Help

- **ReadTheDocs Help**: https://docs.readthedocs.io/
- **Sphinx Docs**: https://www.sphinx-doc.org/
- **Common Issues**: See READTHEDOCS_SETUP.md "Troubleshooting" section
- **Community**: ReadTheDocs Discord/community for quick questions

## 🎉 Result

You now have professional, searchable, auto-updating documentation hosted at no cost on ReadTheDocs!

**Your documentation will automatically rebuild every time you push to GitHub.** 

This significantly strengthens your JOSS submission by demonstrating professional software practices! 🚀

---

## 📋 Files Modified/Created

| File | Change | Status |
|------|--------|--------|
| `.readthedocs.yml` | Enhanced config | ✅ Updated |
| `docs/source/conf.py` | Added themes & features | ✅ Updated |
| `pyproject.toml` | Added `[docs]` extras | ✅ Updated |
| `READTHEDOCS_SETUP.md` | New setup guide | ✅ Created |

All files are ready to commit and push to GitHub!
