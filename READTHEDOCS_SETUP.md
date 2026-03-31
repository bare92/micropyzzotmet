# ReadTheDocs Setup Guide for MicroPyzzotMet

This guide will help you set up your MicroPyzzotMet documentation on ReadTheDocs so it's automatically built and hosted online.

## 📋 Prerequisites

- GitHub account (already have it ✅)
- ReadTheDocs account (will create)
- Repository with proper documentation structure (✅ already configured)

## 🚀 Step-by-Step Setup

### Step 1: Create a ReadTheDocs Account

1. Go to https://readthedocs.org
2. Click **"Sign up"** in the top right
3. Choose **"Sign up with GitHub"**
4. Authorize ReadTheDocs to access your GitHub account
5. Complete your profile setup

### Step 2: Import Your Repository on ReadTheDocs

1. After signing in, click your **username** → **"My projects"** (or dashboard)
2. Click **"Import a Project"** button
3. Choose **"Import automatically from GitHub"**
4. Search for **`micropyzzotmet`** in your repositories
5. Select it and click **"Next"**

#### Configuration on Import:
- **Name**: `micropyzzotmet`
- **Repository**: `https://github.com/bare92/micropyzzotmet` 
- **Repository type**: Git
- **Default branch**: `main` or `joss` (whichever you use)
- **Documentation type**: Sphinx HTML
- **Language**: English
- **Programming language**: Python
- Leave other options as default

6. Click **"Finish"** to import

### Step 3: Verify Webhook Configuration

ReadTheDocs usually sets up webhooks automatically, but verify:

1. Go to your GitHub repository → **Settings** → **Webhooks**
2. Look for a webhook from `readthedocs.org`
3. Click it and verify it shows recent deliveries (green checkmarks)

If no webhook exists:
1. On ReadTheDocs dashboard, go to **Admin** → **Integrations**
2. Click **"Add integration"** and choose **GitHub incoming webhook**
3. Copy the webhook URL
4. Add it manually to GitHub Settings → Webhooks

### Step 4: Build Your Documentation (First Time)

1. Go to your ReadTheDocs project dashboard
2. Click **"Builds"** tab
3. Click **"Build version"** button to trigger manual build

**Wait for the build to complete** (~2-5 minutes)

When complete, you'll see:
```
✓ Build completed successfully
View documentation at: https://micropyzzotmet.readthedocs.io
```

### Step 5: View Your Live Documentation

Visit: **https://micropyzzotmet.readthedocs.io**

Your documentation is now live! 🎉

## 📚 Documentation Structure

Your ReadTheDocs site includes:

- **Main page**: Installation, quickstart, overview
- **Getting Started**: Installation instructions
- **Usage**: Configuration files, how to run downscaling
- **Methods**: Detailed algorithm descriptions for each variable
- **API Reference**: Auto-generated Python API docs
- **Changelog**: Version history

## 🔧 Customizations Available

### Add a Logo

1. Create a `docs/source/_static/` directory (if not exists)
2. Add your logo as `logo.png` (~200x50 px recommended)
3. Update `docs/source/conf.py`:

```python
html_logo = "_static/logo.png"
```

### Add a Favicon

1. Add `favicon.ico` to `docs/source/_static/`
2. Update `docs/source/conf.py`:

```python
html_favicon = "_static/favicon.ico"
```

### Automatic Builds on Push

By default, ReadTheDocs automatically rebuilds when you push to GitHub:

```bash
git add .
git commit -m "Improve documentation"
git push origin main
```

Check **Admin** → **Versions** on ReadTheDocs to see build history.

### Add PDF Download

Users can download PDF from the documentation. PDF building is already enabled in `.readthedocs.yml`.

## 🐛 Troubleshooting

### Build Fails with "Module not found" Error

**Problem**: `ModuleNotFoundError: No module named 'numpy'` etc.

**Solution**: This is expected! The `.readthedocs.yml` automatically mocks these modules. Check:

1. Build log shows `autodoc_mock_imports` in conf.py ✅
2. `.readthedocs.yml` installs with `pip install -e .[dev]` ✅
3. All necessary module mocks are listed in `docs/source/conf.py` ✅

If builds still fail with import errors:
- Go to **Admin** → **Advanced settings**
- Set **Build command**: leave as default
- Try rebuilding

### Documentation Looks Broken/Empty

**Problem**: Pages aren't rendering properly

**Solutions**:

1. **Check reStructuredText syntax**:
   ```bash
   cd docs
   make clean
   make html
   ```
   Look for warnings in output.

2. **Check for broken references**:
   In `.readthedocs.yml`, set `fail_on_warning: true` temporarily to catch issues

3. **Clear cache and rebuild**:
   - Go to **Admin** → **Advanced settings**
   - Toggle **"Build pull request previews"** off then on
   - Trigger manual rebuild

### Autodoc Not Finding My Packages

**Problem**: API reference pages are empty

**Solutions**:

1. Ensure package is properly installed: `pip install -e .[dev]`
2. Check `docs/source/conf.py` - verify `sys.path` includes your package
3. Verify mock modules are listed for heavy dependencies

## 📖 Building Locally Before Pushing

To test your documentation locally before pushing:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs
make clean
make html

# View in browser
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
```

## 🔗 Documentation URLs

- **Main**: `https://micropyzzotmet.readthedocs.io`
- **Latest version**: `https://micropyzzotmet.readthedocs.io/en/latest/`
- **Specific version** (e.g., v0.1.0): `https://micropyzzotmet.readthedocs.io/en/v0.1.0/`
- **PDF**: Available as download from any page
- **GitHub edit links**: Automatically added to each page

## 🚀 Advanced Features Available

Once your documentation is live, you can enable:

### 1. **Documentation Versioning**
- Different versions for different branches/tags
- Users can switch between versions
- Auto-enable for `v*` tags

### 2. **Pull Request Previews**
- Automatic documentation builds for PRs
- Share preview link with reviewers
- Goes to `https://micropyzzotmet--<pr-number>.readthedocs.build/`

### 3. **Private Documentation** (Pro feature)
- Restrict access to documentation
- Only available with paid ReadTheDocs account

### 4. **Search Analytics**
- Track what users search for
- Available in **Admin** → **Analytics**

### 5. **Custom Domain** (Pro feature)
- Point your own domain to ReadTheDocs
- Example: `docs.yourcompany.com`

## 📊 Monitoring Your Documentation

### Check Build Status

1. **Dashboard**: View last build status at a glance
2. **Builds tab**: Detailed build history with logs
3. **Admin → Audit log**: Track all changes

### View Build Logs

1. Go to **Builds** tab
2. Click on any build
3. Click **"View raw"** to see full build output
4. Look for errors in red (often near the end)

## 🎓 Tips for Better Documentation

1. **Add examples**: Use code blocks in `.rst` files
2. **Keep it current**: Update docs when code changes
3. **Use cross-references**: Link between pages with `:ref:`
4. **Document parameters**: Use numpy/Google docstring style
5. **Include diagrams**: reStructuredText supports image embedding

## 🔗 Useful Links

- [ReadTheDocs Documentation](https://docs.readthedocs.io/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://docutils.sourceforge.io/rst.html)
- [napoleon docstring style](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)

## 📞 Getting Help

If builds fail or something looks wrong:

1. **Check Build Logs**: ReadTheDocs → Builds → Select build → View logs
2. **Common Issues**: Google the error message from logs
3. **Local Testing**: Try building locally first with `make html`
4. **Contact Support**: ReadTheDocs has excellent support for free tier

## ✅ Verification Checklist

- [ ] ReadTheDocs account created
- [ ] Repository imported on ReadTheDocs
- [ ] First build completed successfully
- [ ] Documentation visible at https://micropyzzotmet.readthedocs.io
- [ ] All sections (Getting Started, Usage, API, etc.) render correctly
- [ ] You can download PDF
- [ ] Logo/favicon display correctly (if added)
- [ ] GitHub webhook configured for auto-builds

Once all box above are checked, your documentation is fully set up! 🎉

## 🎯 Next Steps

1. **Push any pending changes to GitHub**
2. **Visit your ReadTheDocs dashboard**
3. **Share your documentation URL** with others
4. **Update your README** to link to the documentation:
   ```markdown
   [Read the Documentation](https://micropyzzotmet.readthedocs.io/)
   ```

---

**Your documentation is now automatically updated every time you push to GitHub!**
