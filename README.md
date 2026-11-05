<<<<<<< HEAD
# Streamlit_dl_showcase
showcasing the DL models 
=======
# 🎮 NeuralPlay Standalone (Streamlit Community Cloud & Self-Contained)

A standalone Streamlit web application with built-in video upload, local file persistence, and smooth streaming playback — zero external backend required!

---

## 📁 What's Inside This Folder

- **`app.py`**: The complete application code (UI, cyberpunk CSS, filtering, video upload handling, admin portal).
- **`videos/`**: Bundled with all demo videos (`apex_drift_ppo.mp4`, `cyber_biped_sac.mp4`, `neural_fps_transformer.mp4`, `voxel_world_diffusion.mp4`). Any video uploaded via the Admin Console is automatically saved here.
- **`data/projects.json`**: JSON database storing project metadata, metrics, Hugging Face URLs, and video locations.
- **`.streamlit/config.toml`**: Custom theme configuration and 500 MB video upload limit.
- **`requirements.txt`**: Only requires `streamlit>=1.30.0`.

---

## 🚀 Running Locally

From inside this folder:
```bash
streamlit run app.py
```
Or from the project root:
```bash
.venv/bin/streamlit run standalone_app/app.py
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push your repository to GitHub.
2. In [share.streamlit.io](https://share.streamlit.io):
   - **Repository**: Your GitHub repo
   - **Main file path**: `standalone_app/app.py`
3. In **Advanced settings ➔ Secrets**, set:
   ```toml
   ADMIN_PASSKEY = "your-custom-passkey"
   ```
4. Click **Deploy**!

---

## 🎥 Video Upload & Streaming

When logged in as Administrator in the sidebar:
- Select **➕ Host New ML Project**.
- Under **Video Source Option**, choose **Upload Video File (MP4/WebM)**.
- Browse and upload any video up to 500 MB.
- The video is saved directly to `videos/` and cached in session memory for instant streaming playback in both the **🎮 Project Gallery** and **🎬 Cinema Spotlight** tabs!
>>>>>>> c49426f (Initial commit)
