# 🌐 Deploy Frontend to Vercel

## Quick Steps

### Option 1: Deploy with Vercel CLI (Fastest)

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy:**
   ```bash
   cd C:\Users\Sanju\OneDrive\Desktop\project
   vercel
   ```

3. **Follow prompts:**
   - Link to Vercel account? Yes
   - Set up and deploy? Yes
   - Which scope? (your account)
   - Project name: aemer
   - Directory: ./
   - Override settings? No

4. **Done!** You'll get a URL like: `https://aemer.vercel.app`

---

### Option 2: Deploy via GitHub (Recommended)

1. **Push to GitHub:**
   ```bash
   cd C:\Users\Sanju\OneDrive\Desktop\project
   git init
   git add .
   git commit -m "AEMER - Emotion Recognition"
   git remote add origin https://github.com/YOUR_USERNAME/aemer.git
   git push -u origin main
   ```

2. **Go to [vercel.com](https://vercel.com)**

3. **Click "Import Project"**

4. **Select your GitHub repo**

5. **Deploy settings:**
   - Framework: Vite
   - Build: `npm run build`
   - Output: `dist`

6. **Click Deploy**

---

## Your URLs Will Be:

| Component | URL |
|-----------|-----|
| **Frontend** | https://aemer-YOUR_USERNAME.vercel.app |
| **Backend API** | https://sanjulasunath-aemer.hf.space |

## Share This Link:
After deployment, share your Vercel URL with anyone:
**https://aemer.vercel.app** ← Example
