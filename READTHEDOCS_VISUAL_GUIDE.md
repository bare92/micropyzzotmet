# ReadTheDocs Visual Preview & Launch Guide

## 🎨 What Your Documentation Will Look Like

### Home Page

```
╔════════════════════════════════════════════════════════════════╗
║  📘 MicroPyzzotMet                    🔍 Search...  🌙 Versions║
║═════════════════════════════════════════════════════════════════║
║                                                                 ║
║  Welcome to MicroPyzzotMet's documentation!                   ║
║  ────────────────────────────────────────────                 ║
║                                                                 ║
║  MicroPyzzotMet is a Python library for downscaling           ║
║  meteorological variables (e.g. temperature, radiation...)    ║
║  using DEMs and reanalysis data such as ERA5-Land.            ║
║                                                                 ║
│ ┌─ Getting Started ─────────────────────────────────────────┐ ║
│ │ • Installation - How to install MicroPyzzotMet           │ ║
│ │ • Quickstart - Your first downscaling workflow           │ ║
│ └────────────────────────────────────────────────────────────┘ ║
│                                                                 ║
│ ┌─ Usage Guide ─────────────────────────────────────────────┐ ║
│ │ • Configuration Files - How to configure                 │ ║
│ │ • Running Downscaling - Execute workflow                 │ ║
│ └────────────────────────────────────────────────────────────┘ ║
│                                                                 ║
│ ┌─ Scientific Details ──────────────────────────────────────┐ ║
│ │ • Temperature Downscaling                                │ ║
│ │ • Radiation Downscaling                                  │ ║
│ │ • Humidity & Precipitation                               │ ║
│ └────────────────────────────────────────────────────────────┘ ║
│                                                                 ║
│ ┌─ API Reference ───────────────────────────────────────────┐ ║
│ │ • Full Python API documentation with examples           │ ║
│ └────────────────────────────────────────────────────────────┘ ║
║                                                                 ║
╚════════════════════════════════════════════════════════════════╝
```

### Sidebar Navigation

```
Left Sidebar (Sticky Navigation):
┌─────────────────────────────┐
│ 📖 MicroPyzzotMet           │
│ 📊 0.1.0                    │ 
├─────────────────────────────┤
│ 📍 Getting Started          │
│   ├─ Installation          │
│   └─ Quickstart            │
├─────────────────────────────┤
│ 💼 Usage                    │
│   ├─ Configuration Files    │
│   └─ Running Downscaling    │
├─────────────────────────────┤
│ 🔬 Methods                  │
│   ├─ Temperature            │
│   ├─ Radiation              │
│   ├─ Humidity               │
│   ├─ Precipitation          │
│   └─ Wind                   │
├─────────────────────────────┤
│ 📖 API                      │
│   ├─ ERA5 Land              │
│   ├─ Main Micromet          │
│   ├─ Downscaling            │
│   └─ Utils                  │
├─────────────────────────────┤
│ 📝 Changelog                │
└─────────────────────────────┘
```

### Page Features

Each documentation page will have:

```
┌──────────────────────────────────────────────────┐
│ Edit on GitHub ↗️  PDF | EPUB | HTML Zip        │
├──────────────────────────────────────────────────┤
│                                                  │
│ # Installation                                   │
│                                                  │
│ To install MicroPyzzotMet...                    │
│                                                  │
│ ```bash                                          │
│ pip install micropyzzotmet                       │
│ ``` 📋 (Copy button on hover)                   │
│                                                  │
│ ## From Source                                   │
│                                                  │
│ If you want to contribute or use develop...     │
│                                                  │
└──────────────────────────────────────────────────┘
```

### API Documentation Example

```
📖 get_era5_land

Module for downloading ERA5-Land data from EarthDataHub.

Functions:

build_earthdatahub_url(dataset_path, pat=None, machine="earthdatahub.com")
  Build an authenticated EarthDataHub URL for xarray/fsspec access.
  
  Parameters:
    dataset_path : str
      Dataset path under data.earthdatahub.destine.eu
    pat : str, optional
      Explicit PAT token. If None, credentials from ~/.netrc
    
  Returns:
    str
      HTTPS URL embedding credentials

get_earthdatahub_credentials(machine="earthdatahub.com")
  Load EarthDataHub credentials from ~/.netrc.
  ...
```

## 🚀 Launch Steps (Take 5 minutes!)

### Step 1: View Documentation Locally (2 min)

```bash
cd /home/riccardo/Documents/Pubblications/micropyzzotmet/micropyzzotmet

# Documentation already builds successfully! View it:
open docs/build/html/index.html  # macOS
xdg-open docs/build/html/index.html  # Linux
```

### Step 2: Push to GitHub (1 min)

```bash
git add .readthedocs.yml docs/ pyproject.toml READTHEDOCS*.md
git commit -m "Enhance ReadTheDocs configuration and documentation"
git push origin main
```

### Step 3: Create ReadTheDocs Account & Import (2 min)

1. **Create account**: Go to https://readthedocs.org
2. **Click "Sign up"** → Select **"Sign up with GitHub"**
3. **Authorize** ReadTheDocs to access GitHub
4. **Click "Import a Project"** → **"Import automatically from GitHub"**
5. **Search** for `micropyzzotmet` and select it
6. **Click "Finish"** to import

### Step 4: Wait for Build (3 min)

- Go to **Builds** tab on ReadTheDocs dashboard
- Click **"Build version"** to start first build
- Wait for it to succeed (usually 2-5 minutes)
- Check logs if there are any issues

### Step 5: View Your Live Documentation! ✅

Visit: **https://micropyzzotmet.readthedocs.io**

## 📊 Live Site Features You'll Have

| Feature | What It Does |
|---------|-------------|
| **Search** 🔍 | Full-text search across all documentation |
| **Versions** 📚 | Switch between v0.1.0, v0.2.0, etc. |
| **PDF Download** 📄 | Readers can download full documentation as PDF |
| **Mobile Responsive** 📱 | Works perfectly on phones, tablets, desktops |
| **Edit Links** ✏️ | Readers can suggest documentation improvements |
| **Code Permalinks** 🔗 | Each API page links to source code on GitHub |
| **Auto-Updates** 🔄 | Rebuilds automatically when you push to GitHub |
| **Analytics** 📈 | See which pages are most visited |

## 🎯 After Going Live

### Every time you push to GitHub:
```bash
git add .
git commit -m "Update documentation"
git push origin main
# ✅ ReadTheDocs automatically rebuilds!
```

### When you create a release:
```bash
git tag v0.2.0
git push origin v0.2.0
# ✅ New version appears in versions dropdown!
```

## 📱 Share Your Documentation

Once live, you have several URLs to share:

**Primary URL:**
```
https://micropyzzotmet.readthedocs.io
```

**In README.md:**
```markdown
[📖 Read the Documentation](https://micropyzzotmet.readthedocs.io)
```

**In Social Media:**
```
Check out our documentation: https://micropyzzotmet.readthedocs.io ✨
```

**In code repositories/papers:**
```
Documentation: https://micropyzzotmet.readthedocs.io
```

## 🎓 How This Helps Your JOSS Submission

✅ **Professional Appearance**: Polished, searchable documentation
✅ **Automatic Updates**: Shows commitment to maintenance
✅ **Open Source Signal**: Professional open-source hosting
✅ **Easy Community Access**: Everyone can view docs
✅ **GitHub Integration**: Encourages community contributions
✅ **Industry Standard**: Using the same platform as numpy, scipy, etc.

## 🔗 Quick Reference

| Task | URL/Command |
|------|------------|
| View live docs | https://micropyzzotmet.readthedocs.io |
| Access dashboard | https://readthedocs.org/projects/micropyzzotmet/ |
| Admin settings | https://readthedocs.org/projects/micropyzzotmet/admin/ |
| Build logs | https://readthedocs.org/projects/micropyzzotmet/builds/ |
| Documentation | https://docs.readthedocs.io |

## ✨ Success Indicators

You'll know it's working when:

- [ ] ✅ Build shows "Build succeeded" (green)
- [ ] ✅ Can access https://micropyzzotmet.readthedocs.io
- [ ] ✅ Home page loads with professional styling
- [ ] ✅ Can navigate between pages
- [ ] ✅ Search box works
- [ ] ✅ Can download PDF
- [ ] ✅ GitHub edit links appear on pages

## 📞 Troubleshooting

**Q: Documentation won't build on ReadTheDocs but works locally?**
A: Check build logs for import errors. Likely missing dependency or typo in .rst file.

**Q: Pages look unstyled/broken?**
A: Clear ReadTheDocs cache. Go to Admin → Advanced → click "Reset builds" button.

**Q: Changes not showing after push?**
A: ReadTheDocs shows latest 30 days of history. Check Builds tab to see if new build started.

## 🎉 You're All Set!

All the hard work is done. Just:

1. Push your changes to GitHub ✅
2. Go to https://readthedocs.org
3. Import your repository
4. Wait a few minutes for the build
5. Share your documentation URL

Your professional, auto-updating documentation is live! 🚀

---

## 📚 Complete File List

Files you've modified/created for ReadTheDocs:

```
✅ .readthedocs.yml                       - Primary configuration
✅ docs/source/conf.py                    - Sphinx configuration (enhanced)
✅ pyproject.toml                         - Added [docs] dependencies
✅ READTHEDOCS_SETUP.md                  - Step-by-step setup guide
✅ READTHEDOCS_COMPLETE_SETUP.md         - Comprehensive guide
```

All ready to commit and push! Your documentation infrastructure is now professional-grade. 🎓
