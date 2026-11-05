"""
NeuralPlay // ML Game Observatory
Standalone Streamlit Application for Streamlit Community Cloud or local hosting.

Features:
- Video uploading (persists to local disk under `videos/` AND memory cache for seamless streaming)
- Local streaming via native st.video player
- Cyberpunk brutalist UI with custom inlined CSS
- Projects stored in `data/projects.json` with fallback defaults
- Admin authentication via st.secrets["ADMIN_PASSKEY"] (or fallback "admin123")
"""

import os
import json
import uuid
import base64
import textwrap
from datetime import datetime
from pathlib import Path

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 1. Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralPlay // ML Showcase",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Paths & Directories
# ─────────────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = APP_DIR / "videos"
DATA_DIR = APP_DIR / "data"
PROJECTS_FILE = DATA_DIR / "projects.json"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Security & Admin Passkey
# ─────────────────────────────────────────────────────────────────────────────
try:
    ADMIN_PASSKEY = st.secrets["ADMIN_PASSKEY"]
except Exception:
    ADMIN_PASSKEY = "admin123"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Built-in Seed Data
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_PROJECTS = [
    {
        "id": "apex-drift-ppo",
        "title": "ApexDrift AI: Deep PPO Autonomous Racing",
        "category": "Reinforcement Learning",
        "framework": "PyTorch + Gymnasium",
        "description": "High-speed autonomous racing agent trained with Proximal Policy Optimization inside a custom OpenAI Gymnasium environment. The agent masters drift control, LIDAR-based obstacle avoidance, and adaptive speed regulation across 12 procedurally generated tracks.",
        "video_path": "videos/apex_drift_ppo.mp4",
        "video_type": "local",
        "huggingface_url": "https://huggingface.co/models/rl-zoo3/ppo-LunarLander-v2",
        "hf_repo_type": "Model",
        "metrics": [
            {"label": "Peak Reward", "value": "+1,850"},
            {"label": "Training Steps", "value": "10M"},
            {"label": "Inference Latency", "value": "12 ms"},
            {"label": "Algorithm", "value": "PPO"}
        ],
        "tags": ["PPO", "Autonomous Racing", "LIDAR", "Gymnasium", "PyTorch"],
        "github_url": "https://github.com/DLR-RM/rl-baselines3-zoo",
        "created_at": "2026-11-01"
    },
    {
        "id": "cyber-biped-sac",
        "title": "CyberBiped: SAC Continuous Locomotion & Parkour",
        "category": "Physics & Locomotion",
        "framework": "MuJoCo + Stable-Baselines3",
        "description": "3D humanoid ragdoll agent mastering dynamic bipedal gait, terrain obstacle vaulting, and automatic fall-recovery using Soft Actor-Critic with heavy domain randomization on mass and joint friction.",
        "video_path": "videos/cyber_biped_sac.mp4",
        "video_type": "local",
        "huggingface_url": "https://huggingface.co/models/sb3/cyberbiped-sac-humanoid",
        "hf_repo_type": "Model",
        "metrics": [
            {"label": "Forward Velocity", "value": "4.82 m/s"},
            {"label": "Fall Recovery", "value": "99.1%"},
            {"label": "Torque Efficiency", "value": "0.84"},
            {"label": "Algorithm", "value": "SAC"}
        ],
        "tags": ["MuJoCo", "SAC", "Locomotion", "Humanoid", "Physics Simulation"],
        "github_url": "https://github.com/DLR-RM/stable-baselines3",
        "created_at": "2026-10-18"
    },
    {
        "id": "shadow-tactics-transformer",
        "title": "ShadowTactics Bot: Multi-Agent Spatial Transformer",
        "category": "Multi-Agent & RTS",
        "framework": "Hugging Face Transformers + PyTorch",
        "description": "Squad tactical combat AI utilizing spatial-temporal attention mechanisms to command units in real-time. Capable of synchronized flanking maneuvers, target priority evaluation, and dynamic cover utilization.",
        "video_path": "videos/neural_fps_transformer.mp4",
        "video_type": "local",
        "huggingface_url": "https://huggingface.co/models/ml-games/shadowtactics-transformer-8b",
        "hf_repo_type": "Model",
        "metrics": [
            {"label": "Winrate vs Hard", "value": "94.6%"},
            {"label": "Decision Latency", "value": "8.4 ms"},
            {"label": "Attention Heads", "value": "12 Heads"},
            {"label": "Parameters", "value": "35M Parameters"}
        ],
        "tags": ["Multi-Agent", "Transformer", "RTS", "Decision AI", "PyTorch"],
        "github_url": "https://github.com/huggingface/transformers",
        "created_at": "2026-10-22"
    },
    {
        "id": "voxel-diffusion-3d",
        "title": "VoxelWorld Diffusion: 3D Latent World Engine",
        "category": "Generative 3D",
        "framework": "Diffusers + Custom C++ Engine",
        "description": "Generative diffusion model producing seamless, infinite 3D voxel biomes in real time. Responds dynamically to player exploration direction, biome prompts ('Bioluminescent Canyon'), and seed variations.",
        "video_path": "videos/voxel_world_diffusion.mp4",
        "video_type": "local",
        "huggingface_url": "https://huggingface.co/spaces/ml-games/voxelworld-3d-diffusion",
        "hf_repo_type": "Space",
        "metrics": [
            {"label": "Inference Time", "value": "16.2 ms"},
            {"label": "Voxel Grid", "value": "128x128x64"},
            {"label": "VRAM Usage", "value": "2.8 GB"},
            {"label": "Architecture", "value": "3D U-Net Latent"}
        ],
        "tags": ["Generative AI", "Diffusion", "Voxel", "Procedural Generation", "Diffusers"],
        "github_url": "https://github.com/huggingface/diffusers",
        "created_at": "2026-10-26"
    }
]

# ─────────────────────────────────────────────────────────────────────────────
# 5. Data Persistence Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_projects_data() -> list[dict]:
    if PROJECTS_FILE.exists():
        try:
            with open(PROJECTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception:
            pass
    # Persist defaults if missing
    save_projects_data(FALLBACK_PROJECTS)
    return [dict(p) for p in FALLBACK_PROJECTS]


def save_projects_data(projects: list[dict]) -> None:
    try:
        with open(PROJECTS_FILE, "w", encoding="utf-8") as f:
            json.dump(projects, f, indent=2)
    except Exception as e:
        st.warning(f"Note: Local file write failed ({e}); changes will persist in session.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Styling & Design System
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700;800;900&family=JetBrains+Mono:wght@500;700;800&display=swap');

:root {
  --bg-deep: #060911;
  --bg-card: rgba(11, 16, 28, 0.85);
  --neon-cyan: #00f0ff;
  --neon-green: #00ffaa;
  --neon-purple: #d946ef;
  --hf-yellow: #ffd21e;
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --font-display: 'Space Grotesk', -apple-system, sans-serif;
  --font-body: 'Outfit', -apple-system, sans-serif;
  --font-code: 'JetBrains Mono', monospace;
  --shadow-brutal: 6px 6px 0px #000000;
}

html, body, [class*="css"] {
  font-family: var(--font-body);
  color: var(--text-main);
}

.stApp {
  background-color: var(--bg-deep) !important;
}

.main, .block-container {
  background: transparent !important;
  padding-top: 1.8rem !important;
  padding-bottom: 3.5rem !important;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { background: transparent !important; }

/* ── Header ── */
.center-header {
  text-align: center;
  padding: 2.2rem 1rem 1.6rem 1rem;
}
.brutalist-tag {
  display: inline-block;
  padding: 4px 14px;
  background: #000000;
  color: var(--neon-cyan);
  border: 2px solid var(--neon-cyan);
  font-family: var(--font-code);
  font-size: 0.76rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  border-radius: 3px;
  box-shadow: 3px 3px 0px #000000;
  margin-bottom: 0.9rem;
}
.main-heading {
  font-family: var(--font-display);
  font-size: 3.4rem;
  font-weight: 900;
  line-height: 1.05;
  margin: 0 0 0.8rem 0;
  letter-spacing: -0.035em;
  color: #ffffff;
  text-transform: uppercase;
  text-shadow: 4px 4px 0px #000000, 0 0 25px rgba(0, 240, 255, 0.4);
}

/* ── Project Card ── */
.project-card {
  background: var(--bg-card) !important;
  backdrop-filter: blur(16px) !important;
  border: 2px solid rgba(0, 240, 255, 0.5) !important;
  border-radius: 4px !important;
  padding: 1.6rem !important;
  margin-bottom: 2rem !important;
  box-shadow: var(--shadow-brutal) !important;
  transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
  position: relative;
}
.project-card:hover {
  transform: translate(-3px, -3px) !important;
  border-color: var(--neon-cyan) !important;
  box-shadow: 9px 9px 0px #000000, 9px 9px 0px 2px var(--neon-cyan) !important;
}
.project-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--neon-cyan);
  opacity: 0.9;
}
.card-header-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.9rem;
  gap: 10px;
  flex-wrap: wrap;
}
.category-tag {
  font-size: 0.74rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 4px 12px;
  border-radius: 3px;
  font-family: var(--font-code);
  border: 2px solid #000000;
  box-shadow: 2px 2px 0px #000000;
}
.category-tag.rl { background: var(--neon-cyan); color: #000000; }
.category-tag.physics { background: var(--neon-green); color: #000000; }
.category-tag.multiagent { background: var(--hf-yellow); color: #000000; }
.category-tag.generative { background: var(--neon-purple); color: #ffffff; }
.category-tag.default { background: #94a3b8; color: #000000; }

.framework-pill {
  font-family: var(--font-code);
  font-size: 0.76rem;
  font-weight: 700;
  color: #cbd5e1;
  background: #080c16;
  padding: 4px 10px;
  border-radius: 3px;
  border: 2px solid #1e293b;
  box-shadow: 2px 2px 0px #000000;
}
.card-title {
  font-family: var(--font-display);
  font-size: 1.45rem;
  font-weight: 800;
  color: #ffffff;
  margin: 0.3rem 0 0.6rem 0;
  line-height: 1.25;
  letter-spacing: -0.015em;
}
.card-desc {
  font-size: 0.94rem;
  color: #cbd5e1;
  line-height: 1.55;
  margin-bottom: 1.2rem;
}

/* ── Video Player Frame ── */
.stVideo {
  border: 2px solid rgba(0, 240, 255, 0.45) !important;
  border-radius: 4px !important;
  overflow: hidden !important;
  box-shadow: 5px 5px 0px #000000 !important;
  margin-bottom: 1rem !important;
}

/* ── Metrics Grid ── */
.metrics-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
  gap: 8px;
  margin: 1rem 0 1.2rem 0;
}
.metric-box {
  background: #070a12;
  border: 2px solid #1e293b;
  border-radius: 4px;
  padding: 9px 8px;
  text-align: center;
  box-shadow: 3px 3px 0px #000000;
}
.metric-val {
  font-family: var(--font-code);
  font-size: 1rem;
  font-weight: 800;
  color: var(--neon-cyan);
}
.metric-lbl {
  font-family: var(--font-code);
  font-size: 0.68rem;
  font-weight: 700;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 3px;
}

/* ── Hugging Face Link Button ── */
.hf-link-container {
  margin-top: 1.2rem;
  padding-top: 1.1rem;
  border-top: 2px solid rgba(255, 255, 255, 0.1);
}
.hf-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  width: 100%;
  padding: 13px 20px;
  border-radius: 4px;
  background: var(--hf-yellow);
  color: #000000 !important;
  font-family: var(--font-display);
  font-size: 0.98rem;
  font-weight: 900;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  text-decoration: none !important;
  border: 2px solid #000000;
  box-shadow: 5px 5px 0px #000000;
  transition: all 0.15s ease;
}
.hf-button:hover {
  transform: translate(2px, 2px);
  box-shadow: 3px 3px 0px #000000;
  background: #ffe04d;
  color: #000000 !important;
}
.hf-button .hf-emoji {
  font-size: 1.25rem;
  line-height: 1;
}

/* ── Tags List ── */
.tags-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 0.6rem;
}
.tag-item {
  font-family: var(--font-code);
  font-size: 0.72rem;
  font-weight: 700;
  color: #94a3b8;
  background: #080c16;
  padding: 3px 8px;
  border-radius: 3px;
  border: 1px solid #334155;
  box-shadow: 1px 1px 0px #000000;
}

/* ── Input Overrides ── */
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stTextArea > div > div > textarea {
  background-color: #070a12 !important;
  border: 2px solid #334155 !important;
  color: #ffffff !important;
  border-radius: 4px !important;
  font-family: var(--font-code) !important;
  font-size: 0.92rem !important;
  box-shadow: 3px 3px 0px #000000 !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--neon-cyan) !important;
  box-shadow: 4px 4px 0px var(--neon-cyan) !important;
}

.stButton > button {
  background: var(--neon-cyan) !important;
  color: #000000 !important;
  font-family: var(--font-display) !important;
  font-weight: 900 !important;
  font-size: 0.94rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
  border-radius: 4px !important;
  border: 2px solid #000000 !important;
  box-shadow: 4px 4px 0px #000000 !important;
  padding: 10px 20px !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover {
  transform: translate(2px, 2px) !important;
  box-shadow: 2px 2px 0px #000000 !important;
  background: #33f3ff !important;
}

/* ── Navigation Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 10px;
  background-color: rgba(9, 13, 22, 0.9);
  padding: 8px;
  border-radius: 6px;
  border: 2px solid #1e293b;
  box-shadow: 5px 5px 0px #000000;
  margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--font-display) !important;
  font-size: 0.95rem !important;
  font-weight: 800 !important;
  color: #94a3b8 !important;
  border-radius: 4px !important;
  padding: 10px 22px !important;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.stTabs [aria-selected="true"] {
  background-color: var(--neon-cyan) !important;
  color: #000000 !important;
  border: 2px solid #000000 !important;
  box-shadow: 3px 3px 0px #000000 !important;
  font-weight: 900 !important;
}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
  background-color: #070a14 !important;
  border-right: 2px solid var(--neon-cyan) !important;
  box-shadow: 6px 0px 0px #000000 !important;
}
.sidebar-admin-header h2 {
  font-family: var(--font-display) !important;
  font-size: 1.45rem !important;
  font-weight: 900 !important;
  color: var(--neon-cyan) !important;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  text-shadow: 2px 2px 0px #000000;
  margin-top: 0 !important;
  margin-bottom: 0.2rem !important;
}
"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────
if "projects" not in st.session_state:
    st.session_state.projects = load_projects_data()

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if "uploaded_video_cache" not in st.session_state:
    st.session_state.uploaded_video_cache = {}


# ─────────────────────────────────────────────────────────────────────────────
# 8. Category & Video Resolution Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_category_class(category: str) -> str:
    cat = (category or "").lower()
    if "reinforcement" in cat or "rl" in cat:
        return "rl"
    elif "physics" in cat or "locomotion" in cat:
        return "physics"
    elif "multi-agent" in cat or "rts" in cat:
        return "multiagent"
    elif "generative" in cat or "3d" in cat or "diffusion" in cat:
        return "generative"
    return "default"


def resolve_video_source(project: dict):
    """
    Returns a playable object for st.video():
    1. Memory cache (for newly uploaded videos during active session)
    2. Local file on disk (videos/filename.mp4)
    3. External URL (http / https)
    """
    vid_key = project.get("video_cache_key")
    if vid_key and vid_key in st.session_state.uploaded_video_cache:
        return st.session_state.uploaded_video_cache[vid_key]

    raw_path = project.get("video_path", "")
    if not raw_path:
        return None

    if raw_path.startswith("http://") or raw_path.startswith("https://"):
        return raw_path

    # Check relative to APP_DIR
    local_p = APP_DIR / raw_path
    if local_p.exists() and local_p.is_file():
        return str(local_p)

    # Check directly inside VIDEOS_DIR by filename
    fname = Path(raw_path).name
    direct_vid = VIDEOS_DIR / fname
    if direct_vid.exists() and direct_vid.is_file():
        return str(direct_vid)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 9. Main Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="center-header">
  <div class="brutalist-tag">⚡ OBSERVATORY ARCHIVE // REPO</div>
  <h1 class="main-heading">ML PROJECT SHOWCASE</h1>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

projects_list = st.session_state.projects

# ─────────────────────────────────────────────────────────────────────────────
# 10. Navigation Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_gallery, tab_cinema = st.tabs([
    "🎮 Project Gallery",
    "🎬 Cinema Spotlight"
])

# =============================================================================
# TAB 1: PROJECT GALLERY
# =============================================================================
with tab_gallery:
    f_col1, f_col2, f_col3 = st.columns([3, 2, 1.5])

    with f_col1:
        search_query = st.text_input(
            "Search Projects",
            placeholder="Search by title, tag, algorithm (e.g. PPO, SAC, Diffusion)...",
            label_visibility="collapsed"
        )

    with f_col2:
        all_categories = ["All Categories"] + sorted(list(set(p.get("category", "General") for p in projects_list)))
        selected_category = st.selectbox(
            "Category Filter",
            options=all_categories,
            label_visibility="collapsed"
        )

    with f_col3:
        sort_by = st.selectbox(
            "Sort By",
            options=["Newest First", "Title (A-Z)"],
            label_visibility="collapsed"
        )

    # Filter & Sort
    filtered = list(projects_list)
    if selected_category != "All Categories":
        filtered = [p for p in filtered if p.get("category") == selected_category]

    if search_query:
        q = search_query.lower()
        filtered = [
            p for p in filtered
            if q in p.get("title", "").lower()
            or q in p.get("description", "").lower()
            or q in p.get("framework", "").lower()
            or any(q in t.lower() for t in p.get("tags", []))
        ]

    if sort_by == "Newest First":
        filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    elif sort_by == "Title (A-Z)":
        filtered.sort(key=lambda x: x.get("title", "").lower())

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if not filtered:
        st.info("No projects match your search or filter. Try a different keyword or category!")
    else:
        col_left, col_right = st.columns(2, gap="large")
        columns = [col_left, col_right]

        for idx, project in enumerate(filtered):
            target_col = columns[idx % 2]
            with target_col:
                cat_class = get_category_class(project.get("category", "General"))

                # Metrics HTML
                metrics_html = ""
                metrics = project.get("metrics", [])
                if metrics:
                    metrics_items = "".join([
                        f"""<div class="metric-box">
                              <div class="metric-val">{m.get('value', '-')}</div>
                              <div class="metric-lbl">{m.get('label', '')}</div>
                           </div>""" for m in metrics[:4]
                    ])
                    metrics_html = f'<div class="metrics-row">{metrics_items}</div>'

                # Tags HTML
                tags_html = ""
                tags = project.get("tags", [])
                if tags:
                    tags_items = "".join([f'<span class="tag-item">#{t}</span>' for t in tags[:5]])
                    tags_html = f'<div class="tags-list">{tags_items}</div>'

                hf_url = project.get("huggingface_url", "https://huggingface.co")
                hf_type = project.get("hf_repo_type", "Model")

                # Top info card HTML
                st.markdown(textwrap.dedent(f"""
                <div class="project-card">
                  <div class="card-header-bar">
                    <span class="category-tag {cat_class}">{project.get('category', 'ML Game')}</span>
                    <span class="framework-pill">{project.get('framework', 'PyTorch')}</span>
                  </div>
                  <div class="card-title">{project.get('title', 'Untitled Project')}</div>
                  <div class="card-desc">{project.get('description', '')}</div>
                """), unsafe_allow_html=True)

                # Stream Video (Uploaded file, local file, or direct stream)
                video_obj = resolve_video_source(project)
                if video_obj is not None:
                    try:
                        st.video(video_obj)
                    except Exception as e:
                        st.warning(f"Could not load video: {e}")
                else:
                    st.markdown(
                        '<div style="height: 60px; display: flex; align-items: center; justify-content: center; '
                        'border: 2px dashed #334155; border-radius: 4px; color: #64748b; font-size: 0.85rem; margin-bottom: 1rem;">'
                        'No video available for this project.</div>',
                        unsafe_allow_html=True
                    )

                # Metrics & Links
                st.markdown(textwrap.dedent(f"""
                  {metrics_html}
                  {tags_html}

                  <div class="hf-link-container">
                    <a href="{hf_url}" target="_blank" class="hf-button">
                      <span class="hf-emoji">🤗</span>
                      <span>Explore on Hugging Face ({hf_type})</span>
                    </a>
                  </div>
                </div>
                """), unsafe_allow_html=True)

                # Quick Actions
                with st.expander("🛠 Project Quick Actions / Details"):
                    st.code(f"git clone {hf_url}", language="bash")
                    if st.session_state.admin_authenticated:
                        if st.button("🗑 Remove Project (Admin)", key=f"del_{project.get('id', idx)}"):
                            st.session_state.projects = [p for p in st.session_state.projects if p.get("id") != project.get("id")]
                            save_projects_data(st.session_state.projects)
                            st.success(f"Project removed.")
                            st.rerun()

# =============================================================================
# TAB 2: CINEMA SPOTLIGHT
# =============================================================================
with tab_cinema:
    if not projects_list:
        st.info("No projects available to spotlight.")
    else:
        proj_titles = [f"{p.get('title')} ({p.get('category')})" for p in projects_list]
        selected_cinema_idx = st.selectbox(
            "Select Spotlight Project",
            options=range(len(projects_list)),
            format_func=lambda i: proj_titles[i]
        )

        cin_proj = projects_list[selected_cinema_idx]
        cin_video_obj = resolve_video_source(cin_proj)
        c_cat_class = get_category_class(cin_proj.get("category", "General"))

        # Cinema Header
        st.markdown(textwrap.dedent(f"""
        <div style="background: rgba(11, 16, 28, 0.85); border-radius: 4px; border: 2px solid #00f0ff; box-shadow: 6px 6px 0px #000000; padding: 1.8rem; margin-bottom: 1.5rem;">
          <div style="display: flex; gap: 10px; margin-bottom: 0.9rem;">
            <span class="category-tag {c_cat_class}">{cin_proj.get('category')}</span>
            <span class="framework-pill">{cin_proj.get('framework')}</span>
          </div>
          <h2 style="font-family: var(--font-display); font-size: 2.3rem; font-weight: 900; color: #fff; margin: 0 0 0.8rem 0; text-transform: uppercase;">{cin_proj.get('title')}</h2>
          <p style="font-size: 1.05rem; color: #cbd5e1; max-width: 900px; line-height: 1.6;">{cin_proj.get('description')}</p>
        </div>
        """), unsafe_allow_html=True)

        # Video Player
        if cin_video_obj is not None:
            try:
                st.video(cin_video_obj)
            except Exception as e:
                st.warning(f"Video unavailable: {e}")
        else:
            st.warning("Video file not found for this spotlight project.")

        # Deep Dive Metrics and Hugging Face Link
        cin_c1, cin_c2 = st.columns([1.8, 1])
        with cin_c1:
            st.markdown("### 📊 Performance & Telemetry Metrics")
            cin_metrics = cin_proj.get("metrics", [])
            if cin_metrics:
                m_cols = st.columns(len(cin_metrics))
                for idx_m, m in enumerate(cin_metrics):
                    with m_cols[idx_m]:
                        st.metric(label=m.get("label", ""), value=m.get("value", ""))

            st.markdown("### 🏷 Model Architecture Tags")
            st.markdown(" ".join([f"`#{t}`" for t in cin_proj.get("tags", [])]))

        with cin_c2:
            st.markdown("### 🤗 Hugging Face Repository")
            st.markdown(textwrap.dedent(f"""
            <div style="background: #070a12; border-radius: 4px; padding: 1.4rem; border: 2px solid #ffd21e; box-shadow: 5px 5px 0px #000000;">
              <div style="font-size: 0.82rem; font-family: var(--font-code); font-weight: 700; color: #ffd21e; text-transform: uppercase; margin-bottom: 12px; letter-spacing: 0.05em;">// DIRECT HUB REPOSITORY ACCESS</div>
              <a href="{cin_proj.get('huggingface_url')}" target="_blank" class="hf-button" style="text-align: center;">
                <span class="hf-emoji">🤗</span>
                <span>Open on Hugging Face</span>
              </a>
            </div>
            """), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 11. Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; margin-top: 4rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.06); color: #64748b; font-size: 0.85rem;">
  NeuralPlay ML Game Observatory // Built with Streamlit &amp; Hugging Face Hub Integration
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 12. SIDEBAR: ADMIN CONSOLE
# =============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-admin-header"><h2>🔐 Admin Console</h2></div>', unsafe_allow_html=True)

    if not st.session_state.admin_authenticated:
        st.markdown('<p style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;">Sign in to publish new ML game projects or manage the registry.</p>', unsafe_allow_html=True)
        admin_key = st.text_input("Admin Passkey", type="password", placeholder="Passkey", key="sb_admin_pass")
        if st.button("🔓 Sign In", use_container_width=True, key="btn_admin_login"):
            if admin_key == ADMIN_PASSKEY:
                st.session_state.admin_authenticated = True
                st.success("Authenticated as Administrator!")
                st.rerun()
            else:
                st.error("Invalid passkey.")
    else:
        st.markdown('<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;"><span style="color:#00ffaa; font-size:0.85rem; font-weight:700;">● Admin Session Active</span></div>', unsafe_allow_html=True)
        if st.button("🔒 Sign Out", use_container_width=True, key="btn_admin_logout"):
            st.session_state.admin_authenticated = False
            st.rerun()

        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 14px 0;'>", unsafe_allow_html=True)

        admin_op = st.radio("Admin Operations", ["➕ Host New ML Project", "📋 Manage Registry"], label_visibility="collapsed", key="sb_admin_op")

        if admin_op == "➕ Host New ML Project":
            st.markdown("#### 🚀 Host New ML Project")
            with st.form("admin_add_project_form", clear_on_submit=True):
                title = st.text_input("Project Title *", placeholder="e.g. ApexDrift AI: Deep PPO")
                category = st.selectbox(
                    "Category *",
                    options=[
                        "Reinforcement Learning",
                        "Physics & Locomotion",
                        "Multi-Agent & RTS",
                        "Generative 3D",
                        "Game LLM / NPCs",
                        "Computer Vision & Tracking",
                        "Other"
                    ]
                )
                framework = st.text_input("Framework & Tools", placeholder="e.g. PyTorch + Gymnasium")
                hf_url = st.text_input("Hugging Face Hub URL *", placeholder="https://huggingface.co/...")
                hf_type = st.selectbox("HF Repo Type", options=["Model", "Space", "Dataset"])

                st.markdown("##### 🎥 Video Source")
                video_input_type = st.radio(
                    "Video Source Option",
                    ["Upload Video File (MP4/WebM)", "Direct Video URL"],
                    horizontal=True,
                    label_visibility="collapsed"
                )

                uploaded_video = None
                video_url_input = ""
                if "Upload" in video_input_type:
                    uploaded_video = st.file_uploader(
                        "Upload Gameplay Video File",
                        type=["mp4", "webm", "mov", "avi"],
                        help="The video is saved locally to videos/ and streamed smoothly."
                    )
                else:
                    video_url_input = st.text_input("Direct Video Stream URL", placeholder="https://.../video.mp4")

                github_link = st.text_input("GitHub URL (Optional)", placeholder="https://github.com/...")
                tags_input = st.text_input("Tags (comma separated)", placeholder="PPO, Continuous Control")
                desc = st.text_area("Description *", placeholder="Environment details, algorithms, reward setup...", height=100)

                st.markdown("##### 📈 Telemetry Metrics")
                m1_lbl = st.text_input("Metric 1 Name", value="Peak Reward")
                m1_val = st.text_input("Metric 1 Value", placeholder="+1,850")
                m2_lbl = st.text_input("Metric 2 Name", value="Training Steps")
                m2_val = st.text_input("Metric 2 Value", placeholder="10M Steps")
                m3_lbl = st.text_input("Metric 3 Name", value="Inference Latency")
                m3_val = st.text_input("Metric 3 Value", placeholder="12 ms")
                m4_lbl = st.text_input("Metric 4 Name", value="Algorithm")
                m4_val = st.text_input("Metric 4 Value", placeholder="PPO / SAC")

                submitted = st.form_submit_button("✨ Publish Project", use_container_width=True)

                if submitted:
                    if not title.strip():
                        st.error("Please provide a Project Title.")
                    elif not hf_url.strip():
                        st.error("Please provide a Hugging Face URL.")
                    elif not desc.strip():
                        st.error("Please provide a Project Description.")
                    else:
                        final_video_path = ""
                        final_video_type = "url"
                        video_cache_key = None

                        if "Upload" in video_input_type and uploaded_video is not None:
                            try:
                                video_bytes = uploaded_video.getvalue()
                                ext = Path(uploaded_video.name).suffix or ".mp4"
                                safe_fname = f"{uuid.uuid4().hex[:10]}_{Path(uploaded_video.name).stem}{ext}"
                                save_dest = VIDEOS_DIR / safe_fname

                                # Write to disk
                                with open(save_dest, "wb") as f_out:
                                    f_out.write(video_bytes)

                                # Also cache in session state for instant streaming playback
                                video_cache_key = f"cache_{safe_fname}"
                                st.session_state.uploaded_video_cache[video_cache_key] = video_bytes

                                final_video_path = f"videos/{safe_fname}"
                                final_video_type = "local"
                            except Exception as e:
                                st.error(f"Error saving video: {e}")
                                st.stop()

                        elif video_url_input.strip():
                            final_video_path = video_url_input.strip()
                            final_video_type = "url"
                        else:
                            final_video_path = "videos/apex_drift_ppo.mp4"
                            final_video_type = "local"

                        new_metrics = []
                        for lbl, val in [(m1_lbl, m1_val), (m2_lbl, m2_val), (m3_lbl, m3_val), (m4_lbl, m4_val)]:
                            if lbl.strip() and val.strip():
                                new_metrics.append({"label": lbl.strip(), "value": val.strip()})

                        new_tags = [t.strip() for t in tags_input.split(",") if t.strip()]
                        if not new_tags:
                            new_tags = [category, framework] if framework else [category]

                        new_project = {
                            "id": str(uuid.uuid4()),
                            "title": title.strip(),
                            "category": category,
                            "framework": framework.strip() or "PyTorch",
                            "description": desc.strip(),
                            "video_path": final_video_path,
                            "video_type": final_video_type,
                            "video_cache_key": video_cache_key,
                            "huggingface_url": hf_url.strip(),
                            "hf_repo_type": hf_type,
                            "metrics": new_metrics,
                            "tags": new_tags,
                            "github_url": github_link.strip(),
                            "created_at": datetime.now().strftime("%Y-%m-%d")
                        }

                        st.session_state.projects.insert(0, new_project)
                        save_projects_data(st.session_state.projects)
                        st.success(f"🎉 '{title}' successfully published and video ready for streaming!")
                        st.balloons()
                        st.rerun()

        elif admin_op == "📋 Manage Registry":
            st.markdown(f"#### 📋 Hosted Projects ({len(st.session_state.projects)})")
            for p in st.session_state.projects:
                with st.container():
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px; margin-bottom: 6px;">
                      <div style="font-weight: 700; font-size: 0.88rem; color: #fff;">{p.get('title')}</div>
                      <div style="font-size: 0.72rem; color: #94a3b8;">{p.get('category')} • {p.get('framework')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("🗑 Remove Project", key=f"sb_del_{p.get('id')}", use_container_width=True):
                        # Clean up cache if any
                        k = p.get("video_cache_key")
                        if k and k in st.session_state.uploaded_video_cache:
                            del st.session_state.uploaded_video_cache[k]

                        st.session_state.projects = [x for x in st.session_state.projects if x.get("id") != p.get("id")]
                        save_projects_data(st.session_state.projects)
                        st.success(f"Removed '{p.get('title')}'.")
                        st.rerun()
