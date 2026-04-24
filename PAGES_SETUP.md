# Enable GitHub Pages

The documentation is configured to auto-deploy to GitHub Pages via a GitHub Actions workflow.

## Option 1: Automatic Deployment (Recommended)

The workflow `.github/workflows/deploy-pages.yml` will:

1. **Automatically trigger** on every push to `main` that modifies `docs/**` or `mkdocs.yml`
2. **Build** the MkDocs documentation using Python 3.10
3. **Deploy** to the `gh-pages` branch
4. **Publish** at: **https://shanksLSC.github.io/sensVeridian/**

No manual configuration needed—it works automatically!

### Monitor Deployment

- Go to: **https://github.com/shanksLSC/sensVeridian/actions**
- Look for "Deploy to GitHub Pages (gh-pages branch)" workflow
- You should see a ✅ **success** after the next push to `main`

## Option 2: Manual Configuration (If Needed)

If the automatic deployment doesn't work, manually enable Pages:

1. Go to: **https://github.com/shanksLSC/sensVeridian/settings/pages**

2. Under "Build and deployment":
   - **Source:** Select `Deploy from a branch`
   - **Branch:** Select `gh-pages` / `/ (root)`

3. Click **Save**

4. Pages will build from the `gh-pages` branch (created by the workflow)

## Workflow Files

- **`.github/workflows/deploy-pages.yml`** (primary)
  - Deploys to `gh-pages` branch using `peaceiris/actions-gh-pages`
  - Works even if Pages isn't manually enabled
  - Recommended approach

- **`.github/workflows/pages.yml`** (alternative)
  - Uses GitHub's native Pages deployment (requires manual Pages enablement)

- **`.github/workflows/docs.yml`** (alternative)
  - Legacy workflow, can be removed

## Troubleshooting

### "Error: Get Pages site failed"

**Solution:** Wait 1-2 minutes after the workflow succeeds, then:

```bash
# Visit the site after first successful workflow run:
https://shanksLSC.github.io/sensVeridian/

# Or monitor the workflow:
https://github.com/shanksLSC/sensVeridian/actions
```

### Workflow didn't run

**Check:**
- Is there a ✅ or ❌ in the **Actions** tab?
- Did you push to `main`?
- Did you modify `docs/**` or `mkdocs.yml`?

**Force a run:**
1. Go to **Actions** tab
2. Click **"Deploy to GitHub Pages (gh-pages branch)"**
3. Click **"Run workflow"** (top right, blue button)
4. Select `main` branch
5. Click **"Run workflow"**

## Once Enabled

The documentation site will be live at:

```
https://shanksLSC.github.io/sensVeridian/
```

All 20+ pages including:
- Installation & Quick Start
- Complete CLI Reference
- Data Model with Sample Queries
- Python API Documentation
- Troubleshooting Guide
- And more...
