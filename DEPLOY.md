# Deployment Guide for PubMed RAG Analyzer

This guide shows how to deploy your PubMed RAG Limitation Analyzer to the cloud.

## Quick Start (Recommended: Render.com)

### Prerequisites
1. GitHub account
2. OpenAI API key (from platform.openai.com)
3. Optional: NCBI API key (from ncbi.nlm.nih.gov/account/)

### Step 1: Prepare Repository

```bash
cd /Users/andreacacioppo/pubmed-rag

# Create requirements.txt with all dependencies
.venv/bin/pip freeze > requirements.txt

# Verify requirements.txt was created
ls -lh requirements.txt
```

### Step 2: Create WSGI Entry Point

Create `wsgi.py` in the project root:

```python
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import FastAPI app
from web.api import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### Step 3: Create Procfile

Create `Procfile` in the project root (for Render):

```
web: gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 4 --timeout 60
```

### Step 4: Push to GitHub

```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial deployment setup"

# Create GitHub repository at github.com/new
# Then push to your repo
git remote add origin https://github.com/YOUR_USERNAME/pubmed-rag.git
git branch -M main
git push -u origin main
```

### Step 5: Deploy on Render

1. Go to [render.com](https://render.com) and sign up
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account
4. Select your `pubmed-rag` repository
5. Configure the service:

| Setting | Value |
|---------|-------|
| **Name** | `pubmed-rag` |
| **Environment** | `Python` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 4 --timeout 60` |
| **Instance Type** | Starter (free tier) |

6. Add Environment Variables in Render dashboard:
   - `OPENAI_API_KEY` - Your OpenAI API key (required)
   - `NCBI_API_KEY` - NCBI API key (optional)
   - `NCBI_EMAIL` - Your email for NCBI

7. Click **"Deploy"**

Your app will be live at: `https://pubmed-rag.onrender.com`

---

## Alternative Platforms

### Railway (Easy alternative)

1. Go to [railway.app](https://railway.app)
2. Click **"New Project"** → **"Deploy from GitHub"**
3. Select your repository
4. Railway auto-detects Python and installs dependencies
5. Add environment variables in the dashboard
6. Deploy automatically on git push

### PythonAnywhere

1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload files via web interface or git
3. Create virtual environment:
   ```bash
   mkvirtualenv pubmed-rag --python=/usr/bin/python3.10
   pip install -r requirements.txt
   ```
4. Configure web app in dashboard
5. Add environment variables in WSGI file

### Docker (Self-hosted)

1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 8000
   CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:8000"]
   ```

2. Build and run:
   ```bash
   docker build -t pubmed-rag .
   docker run -p 8000:8000 -e OPENAI_API_KEY=your-key pubmed-rag
   ```

---

## Environment Variables Required

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `NCBI_API_KEY` | No | NCBI API key for higher rate limits |
| `NCBI_EMAIL` | No | Your email for NCBI |
| `PORT` | Auto | Set by platform (default 8000) |

---

## Troubleshooting

### Build Fails
- Check `requirements.txt` includes all dependencies
- Ensure Python 3.10+ is specified
- Verify WSGI entry point is correct

### App Won't Start
- Check logs in platform dashboard
- Verify environment variables are set
- Test locally first: `python serve.py`

### API Keys Not Working
- Verify keys in environment variables
- Check key permissions/scopes
- Test with a simple API call

### Memory Issues
- Reduce workers in gunicorn (try 2 instead of 4)
- Upgrade instance type if needed

---

## Post-Deployment

1. **Test the app** at your deployed URL
2. **Check logs** for any errors
3. **Test basic functionality**:
   - Enter a topic and search
   - Ask a question
   - Generate a report
4. **Set custom domain** (optional)
5. **Monitor usage** to avoid unexpected costs

---

## Cost Estimates

| Platform | Free Tier | Notes |
|----------|-----------|-------|
| Render | 750 hours/month | Good for small projects |
| Railway | $5 credit/month | Easy to use |
| PythonAnywhere | 1 web worker | Simple Python hosting |
| Docker/VPS | Varies | More control, more setup |

---

## Security Notes

- Never commit `.env` file to git
- Use environment variables for all secrets
- Restrict API keys to your deployment URL when possible
- Monitor API usage regularly

---

## Support

If you encounter issues:
1. Check platform logs
2. Test locally first
3. Review environment variables
4. Check platform documentation
