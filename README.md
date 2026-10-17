<div align="center">

# 🛡️ Privacy-Preserving Vision with Generative Inpainting

**Detects faces and licence plates in images and videos, then replaces them with realistic AI-generated content instead of black boxes or blur.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion%202-Inpainting-8A2BE2?style=flat-square)
![Hugging Face](https://img.shields.io/badge/Diffusers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=flat-square&logo=yolo&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

</div>

---

## 📌 Why

Faces and licence plates in photos and videos are personal data under **GDPR** and **CCPA**. Classic anonymization (blur, black boxes) protects privacy but ruins the image. This project **detects** sensitive regions and **regenerates** them with a diffusion model, so the result stays private *and* still looks natural.

<!-- Add a before/after image here, using photos you own or that are licensed for reuse:
![Before and after](docs/before_after.png) -->

## ⚙️ How it works

```
Image / video frame
   │
   ├─► Face detection ── InsightFace SCRFD (buffalo_l)
   ├─► Plate detection ─ YOLO (custom weights, optional) → OpenCV Haar cascade fallback
   │
   ▼
Binary mask from bounding boxes (+ adjustable dilation)
   │
   ▼
Stable Diffusion 2 inpainting (diffusers) ── prompt + negative prompt
   │
   ▼
Optional synthetic plate text ──► anonymized image / video
```

## ✨ Features

- **Face detection** with InsightFace SCRFD, running on GPU when available and falling back to CPU
- **Licence plate detection** with YOLO (bring your own `.pt` weights), falling back to an OpenCV Haar cascade
- **Generative inpainting** with `stabilityai/stable-diffusion-2-inpainting`, with tunable steps and guidance
- **Synthetic plate text:** draws a random fake plate number over detected plates
- **Mask controls:** separate dilation for faces and plates, plus a live red mask-overlay preview
- **Video anonymization:** processes MP4s frame by frame with a frame limit for demos
- **Quality evaluation:** PSNR and SSIM computed on the *non-masked* regions, to check that the rest of the image is preserved
- **Streamlit UI** with Image, Video, Evaluation and Datasets tabs, and a download button for the result

## 🚀 Getting started

### Option A: Docker

```bash
git clone https://github.com/Anupbhagat77/<repo-name>.git
cd <repo-name>
docker compose up --build
```

Open **http://localhost:8501**. The first inpainting run downloads the Stable Diffusion model (~1–2 GB), which is cached in a Docker volume for later runs.

### Option B: Local Python

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

> 💡 A CUDA GPU makes inpainting much faster. On CPU each image takes noticeably longer.

## 📁 Project structure

```
app/
├── streamlit_app.py   # UI: image, video, evaluation and dataset tabs
├── detectors.py       # InsightFace face detection, YOLO + Haar plate detection
├── inpaint.py         # Stable Diffusion 2 inpainting pipeline
└── utils.py           # masks, overlays, fake plate text
data/                  # inputs (git-ignored)
outputs/               # results (git-ignored)
Dockerfile
docker-compose.yml
requirements.txt
```

## 🔮 Future work

- Fine-tune a YOLO model on CCPD for more robust plate detection
- Speed up video by detecting on keyframes and tracking between them
- Identity-leakage checks, such as face-recognition similarity before vs. after

## 👥 Credits

University project at **HTW Berlin**. Technical implementation (detection pipeline, generative inpainting, evaluation and UI) by **Anup Bhagat**.

---

<div align="center">
<sub>More projects on my <a href="https://anupbhagat77.github.io">portfolio</a> · <a href="https://www.linkedin.com/in/-anup-bhagat-">LinkedIn</a></sub>
</div>
