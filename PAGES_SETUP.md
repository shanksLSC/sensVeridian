# Enable GitHub Pages

The documentation is configured to auto-deploy to GitHub Pages. To enable it:

1. Go to: **https://github.com/shanksLSC/sensVeridian/settings/pages**

2. Under "Build and deployment":
   - Set **Source** to: `GitHub Actions`
   - (Or if GitHub Actions isn't available, use: `Deploy from a branch` → `main` → `/root`)

3. Save settings

4. GitHub Actions will automatically:
   - Build the MkDocs documentation on every push to `main`
   - Deploy to **https://shanksLSC.github.io/sensVeridian/**

The workflow files are in:
- `.github/workflows/pages.yml` (primary deployment workflow)
- `.github/workflows/docs.yml` (alternative, also valid)

You can monitor the deployment in the **Actions** tab of the repository.
