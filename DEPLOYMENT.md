# FlashForge Deployment Guide - Render.com

Complete step-by-step guide to deploy FlashForge Flask app on Render.com free tier with OCR support and persistent database storage.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Pre-Deployment Setup](#pre-deployment-setup)
3. [GitHub Repository Setup](#github-repository-setup)
4. [Render.com Account Setup](#rendercom-account-setup)
5. [Deploy on Render](#deploy-on-render)
6. [Post-Deployment](#post-deployment)
7. [Troubleshooting](#troubleshooting)
8. [Updating Your App](#updating-your-app)

---

## Prerequisites

Before starting, ensure you have:

- ✅ **GitHub Account** - Free account works fine
- ✅ **Google Gemini API Key** - Get free key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- ✅ **Git installed** on your local machine
- ✅ **Python 3.12** installed locally (for testing)

### Getting Your Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the API key (starts with `AIza...`)
5. **Important**: Save this key securely - you'll need it for Render deployment

**Free Tier Limits:**
- 15 requests per minute
- 1,500 requests per day
- Sufficient for personal/educational use

---

## Pre-Deployment Setup

### Step 1: Verify Local Files

Ensure these files exist in your project root:

- ✅ `app.py` - Main Flask application
- ✅ `requirements-deploy.txt` - Python dependencies (without Windows packages)
- ✅ `build.sh` - Build script for Render
- ✅ `render.yaml` - Render configuration
- ✅ `.env.example` - Environment variables template
- ✅ `templates/` - HTML templates directory
- ✅ `static/` - CSS/JS static files directory
- ✅ `utils/` - Application utilities directory

### Step 2: Test Locally (Optional but Recommended)

1. Create a `.env` file from `.env.example`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your actual values:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   SECRET_KEY=generate_with_python_command_below
   DATABASE_PATH=database/flashforge.db
   TESS_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe  # Windows path
   ```

3. Generate a secure SECRET_KEY:
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```
   Copy the output and paste it as `SECRET_KEY` in `.env`

4. Test the app locally:
   ```bash
   python app.py
   ```
   Visit `http://localhost:5000` and verify everything works.

---

## GitHub Repository Setup

### Step 3: Initialize Git Repository (if not already done)

```bash
git init
git add .
git commit -m "Prepare for Render deployment"
```

### Step 4: Create GitHub Repository

1. Go to [GitHub](https://github.com/new)
2. Create a new repository (public or private)
3. Name it `flashforge` or your preferred name
4. **Don't** initialize with README (you already have files)

### Step 5: Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/flashforge.git
git branch -M main
git push -u origin main
```

**Important**: Ensure `.env` is in `.gitignore` and never commit API keys!

---

## Render.com Account Setup

### Step 6: Create Render Account

1. Visit [Render.com](https://render.com)
2. Click "Get Started for Free"
3. Sign up with your GitHub account (recommended) or email
4. Verify your email address
5. **No credit card required** for free tier

### Step 7: Connect GitHub (if using GitHub signup)

1. Render will automatically connect your GitHub account
2. Authorize Render to access your repositories
3. You can revoke access anytime in GitHub settings

---

## Deploy on Render

### Step 8: Create New Web Service

1. In Render dashboard, click **"New +"** → **"Web Service"**
2. Click **"Connect account"** if prompted (connect GitHub)
3. Find and select your `flashforge` repository
4. Click **"Connect"**

### Step 9: Configure Service Settings

#### Basic Settings

- **Name**: `flashforge` (or your preferred name)
- **Region**: Choose closest to you (e.g., `Oregon (US West)` for US)
- **Branch**: `main` (or your default branch)
- **Root Directory**: Leave blank
- **Runtime**: `Python 3`
- **Build Command**: `./build.sh`
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

#### Advanced Settings (Click "Advanced")

- **Auto-Deploy**: `Yes` (deploys automatically on git push)
- **Health Check Path**: `/`

### Step 10: Configure Environment Variables

In the **Environment** section, add these variables:

| Key | Value | Notes |
|-----|-------|-------|
| `GEMINI_API_KEY` | `your_actual_gemini_api_key` | Your Google Gemini API key |
| `SECRET_KEY` | `generate_random_string` | Use: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `DATABASE_PATH` | `/var/data/flashforge.db` | Persistent disk path |
| `TESS_PATH` | `/usr/bin/tesseract` | Tesseract OCR path (Linux) |
| `PYTHON_VERSION` | `3.12.0` | Python version |

**To generate SECRET_KEY:**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Step 11: Add Persistent Disk

**Important**: This ensures your database persists across deployments!

1. Scroll down to **"Disks"** section
2. Click **"Add Disk"**
3. Configure:
   - **Name**: `flashforge-data`
   - **Mount Path**: `/var/data`
   - **Size**: `1 GB` (free tier limit)
4. Click **"Save"**

### Step 12: Deploy

1. Review all settings
2. Click **"Create Web Service"**
3. Render will start building (takes 5-10 minutes first time)
4. Monitor build logs for any errors
5. Wait for **"Live"** status

**Build Process:**
- Installs system dependencies (Tesseract, poppler-utils)
- Installs Python packages from `requirements-deploy.txt`
- Creates database directory
- Starts Gunicorn server

---

## Post-Deployment

### Step 13: Verify Deployment

1. Click on your service URL (e.g., `https://flashforge.onrender.com`)
2. You should see the FlashForge home page
3. Test registration: Create a new user account
4. Test login: Log in with created account
5. Test deck creation: Create a test deck from dashboard

### Step 14: Test Core Features

1. **Upload PDF**: Upload a text-based PDF file
   - Should extract text successfully
   - Generate flashcards

2. **Test OCR**: Upload a scanned/image PDF
   - Should use OCR to extract text
   - Generate flashcards from OCR text

3. **Review Flashcards**: Start a review session
   - Test spaced repetition functionality
   - Verify cards are saved correctly

4. **Dashboard**: Check analytics and statistics
   - Verify charts load correctly
   - Check user stats

### Step 15: Monitor Logs

1. In Render dashboard, click **"Logs"** tab
2. Check for any errors or warnings
3. Common issues:
   - Missing environment variables
   - Database connection errors
   - OCR initialization failures

---

## Troubleshooting

### Build Fails

**Problem**: Build script fails or times out

**Solutions**:
1. Check `build.sh` has execute permissions:
   ```bash
   git update-index --chmod=+x build.sh
   git commit -m "Make build.sh executable"
   git push
   ```

2. Verify `requirements-deploy.txt` doesn't have Windows packages
3. Check Python version compatibility
4. Review build logs for specific error messages

### App Crashes on Startup

**Problem**: Service shows "Unhealthy" or crashes immediately

**Solutions**:
1. **Check Environment Variables**:
   - Verify all required variables are set
   - Ensure `GEMINI_API_KEY` is correct
   - Check `DATABASE_PATH` matches disk mount path

2. **Check Logs**:
   - Look for Python import errors
   - Verify database initialization succeeds
   - Check for missing dependencies

3. **Database Issues**:
   - Ensure persistent disk is mounted at `/var/data`
   - Verify `DATABASE_PATH` is `/var/data/flashforge.db`
   - Check disk has available space

### OCR Not Working

**Problem**: OCR fails or doesn't extract text from images

**Solutions**:
1. Verify Tesseract is installed (check build logs)
2. Check `TESS_PATH` is `/usr/bin/tesseract`
3. Test with simple text PDF first
4. Check OCR handler logs for errors

### Slow Performance

**Problem**: App is slow or times out

**Solutions**:
1. **Free Tier Limitations**:
   - 512 MB RAM (limited)
   - Shared CPU (can be slow)
   - Cold starts after 15 min inactivity (30-60s)

2. **Optimizations**:
   - Reduce flashcard generation batch size
   - Optimize database queries
   - Consider upgrading to paid tier ($7/month)

3. **Cold Starts**:
   - Free tier apps sleep after 15 min inactivity
   - First request wakes the app (takes 30-60 seconds)
   - Subsequent requests are faster

### Database Not Persisting

**Problem**: Data disappears after redeployment

**Solutions**:
1. Verify persistent disk is configured:
   - Check disk is mounted at `/var/data`
   - Verify `DATABASE_PATH` uses `/var/data/flashforge.db`
   - Ensure disk size is sufficient (1 GB free tier)

2. Check disk status in Render dashboard
3. Database file should persist across deployments

### Environment Variables Not Working

**Problem**: App can't read environment variables

**Solutions**:
1. Verify variables are set in Render dashboard
2. Check variable names match exactly (case-sensitive)
3. Restart service after adding variables
4. Use `os.getenv()` in code (already implemented)

---

## Updating Your App

### Automatic Deployment

Render automatically deploys when you push to your connected branch:

```bash
git add .
git commit -m "Update feature"
git push origin main
```

Render will detect the push and start a new deployment.

### Manual Deployment

1. In Render dashboard, click **"Manual Deploy"**
2. Select **"Deploy latest commit"**
3. Wait for deployment to complete

### Rollback

If a deployment breaks your app:

1. Go to **"Events"** tab in Render dashboard
2. Find the last working deployment
3. Click **"Rollback"**
4. Confirm rollback

---

## Free Tier Limitations

### Render Free Tier

- ✅ **750 hours/month** - Enough for one always-on service
- ✅ **512 MB RAM** - Sufficient for Flask app
- ✅ **1 GB persistent disk** - Enough for SQLite database
- ⚠️ **Sleeps after 15 min** - Cold start takes 30-60 seconds
- ⚠️ **Shared CPU** - Can be slow under load

### Gemini API Free Tier

- ✅ **15 requests/minute** - Rate limit
- ✅ **1,500 requests/day** - Daily limit
- ✅ **No credit card required**
- ⚠️ **Rate limiting** - May need to add delays for heavy usage

### Recommendations

- **For Development**: Free tier is perfect
- **For Production**: Consider upgrading to Render Starter ($7/month) for:
  - No sleep (always on)
  - More RAM (512 MB → 2 GB)
  - Better CPU performance
  - Faster response times

---

## Security Best Practices

1. **Never commit `.env` file** - Already in `.gitignore`
2. **Use strong SECRET_KEY** - Generate with `secrets.token_hex(32)`
3. **Keep API keys secure** - Don't share Render dashboard access
4. **Regular updates** - Keep dependencies updated
5. **Monitor logs** - Check for suspicious activity

---

## Additional Resources

- [Render Documentation](https://render.com/docs)
- [Flask Deployment Guide](https://flask.palletsprojects.com/en/latest/deploying/)
- [Gunicorn Configuration](https://docs.gunicorn.org/en/stable/settings.html)
- [Google Gemini API Docs](https://ai.google.dev/docs)

---

## Support

If you encounter issues not covered here:

1. Check Render build logs for errors
2. Review application logs for runtime errors
3. Verify all environment variables are set
4. Test locally first before deploying
5. Check Render status page for service outages

---

## Quick Reference

### Important Commands

```bash
# Generate SECRET_KEY
python -c "import secrets; print(secrets.token_hex(32))"

# Test locally
python app.py

# Push to GitHub
git add .
git commit -m "Update"
git push origin main
```

### Important Paths

- **Database (Local)**: `database/flashforge.db`
- **Database (Render)**: `/var/data/flashforge.db`
- **Tesseract (Local Windows)**: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- **Tesseract (Render)**: `/usr/bin/tesseract`

### Environment Variables

| Variable | Local Value | Render Value |
|----------|-------------|--------------|
| `DATABASE_PATH` | `database/flashforge.db` | `/var/data/flashforge.db` |
| `TESS_PATH` | Windows path | `/usr/bin/tesseract` |
| `GEMINI_API_KEY` | Your key | Your key |
| `SECRET_KEY` | Generated | Generated |

---

**Congratulations!** Your FlashForge app should now be live on Render.com! 🎉

