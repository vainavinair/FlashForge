# Render Deployment Checklist

Use this checklist to ensure a smooth deployment on Render.com.

## Pre-Deployment Checklist

- [ ] **GitHub Repository Ready**
  - [ ] Code pushed to GitHub
  - [ ] All files committed
  - [ ] `.env` file is NOT committed (check `.gitignore`)

- [ ] **Environment Variables Prepared**
  - [ ] Google Gemini API key obtained
  - [ ] SECRET_KEY generated: `python -c "import secrets; print(secrets.token_hex(32))"`
  - [ ] Values ready to paste into Render dashboard

- [ ] **Files Verified**
  - [ ] `build.sh` exists and has shebang (`#!/bin/bash`)
  - [ ] `render.yaml` exists with correct configuration
  - [ ] `requirements-deploy.txt` exists (no `pywin32`)
  - [ ] `app.py` has production settings (`debug=False`)
  - [ ] `utils/datamodels/db.py` uses `DATABASE_PATH` env var

- [ ] **Local Testing** (Optional but Recommended)
  - [ ] App runs locally: `python app.py`
  - [ ] Can register/login
  - [ ] Can upload files
  - [ ] Can generate flashcards

## Render Deployment Checklist

- [ ] **Account Setup**
  - [ ] Render account created
  - [ ] GitHub connected (if using GitHub signup)

- [ ] **Service Configuration**
  - [ ] Repository connected
  - [ ] Name set: `flashforge`
  - [ ] Region selected
  - [ ] Build Command: `./build.sh`
  - [ ] Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

- [ ] **Environment Variables Set**
  - [ ] `GEMINI_API_KEY` = your actual key
  - [ ] `SECRET_KEY` = generated random string
  - [ ] `DATABASE_PATH` = `/var/data/flashforge.db`
  - [ ] `TESS_PATH` = `/usr/bin/tesseract`
  - [ ] `PYTHON_VERSION` = `3.12.0`

- [ ] **Persistent Disk Added**
  - [ ] Disk name: `flashforge-data`
  - [ ] Mount path: `/var/data`
  - [ ] Size: `1 GB`

- [ ] **Deployment**
  - [ ] Service created
  - [ ] Build completed successfully
  - [ ] Status shows "Live"

## Post-Deployment Checklist

- [ ] **Basic Functionality**
  - [ ] Home page loads
  - [ ] Can register new user
  - [ ] Can login
  - [ ] Dashboard displays

- [ ] **Core Features**
  - [ ] Can create deck
  - [ ] Can upload PDF (text extraction works)
  - [ ] Can upload scanned PDF (OCR works)
  - [ ] Flashcards generate successfully
  - [ ] Can review flashcards
  - [ ] Spaced repetition works

- [ ] **Persistence**
  - [ ] Data persists after app restart
  - [ ] Database file exists in persistent disk
  - [ ] User data survives redeployment

- [ ] **Monitoring**
  - [ ] Check logs for errors
  - [ ] Monitor resource usage
  - [ ] Verify no crashes

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Build fails | Check build logs, verify `build.sh` is executable |
| App crashes | Check environment variables, review logs |
| OCR not working | Verify `TESS_PATH` is `/usr/bin/tesseract` |
| Database not persisting | Check disk mount path matches `DATABASE_PATH` |
| Slow performance | Normal for free tier, consider upgrade |

## Quick Commands

```bash
# Generate SECRET_KEY
python -c "import secrets; print(secrets.token_hex(32))"

# Test locally
python app.py

# Make build.sh executable (before first push)
git update-index --chmod=+x build.sh
git commit -m "Make build.sh executable"
git push
```

## Important Notes

- ⚠️ Free tier apps sleep after 15 minutes of inactivity
- ⚠️ First request after sleep takes 30-60 seconds (cold start)
- ⚠️ 512 MB RAM limit - may need optimization for heavy usage
- ✅ Database persists across deployments (thanks to persistent disk)
- ✅ Auto-deploy enabled - pushes to GitHub trigger new deployments

---

**Ready to deploy?** Follow the detailed steps in [DEPLOYMENT.md](DEPLOYMENT.md)

