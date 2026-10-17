# app/streamlit_app.py
import sys, os, io, cv2, numpy as np, streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from app.detectors import detect_faces, detect_plates_yolo, detect_plates_haar
from app.inpaint import inpaint
from app.utils import boxes_to_mask, overlay_mask, save_boxes_json, draw_fake_plate_text

st.set_page_config(page_title="Privacy-Preserving Inpainting", layout="wide")
st.title("Privacy-Preserving Vision (Detection + Generative Inpainting)")

# --- Sidebar options
st.sidebar.header("Options")
yolo_plate_weights = st.sidebar.text_input("YOLO plate weights (.pt) (optional)", value="")
face_dilate = st.sidebar.slider("Face dilation (px)", 0, 32, 12)
plate_dilate = st.sidebar.slider("Plate dilation (px)", 0, 32, 8)
inference_steps = st.sidebar.slider("Inpainting steps", 10, 60, 30)
guidance = st.sidebar.slider("Guidance scale", 1.0, 12.0, 7.0)
draw_fake = st.sidebar.checkbox("Draw random fake plate text", value=True)

tabs = st.tabs(["Image", "Video", "Evaluation", "Datasets"])

# -------- IMAGE TAB --------
with tabs[0]:
    st.subheader("Image anonymization")

    # persistent state holders
    if "last_out_img" not in st.session_state:
        st.session_state.last_out_img = None
    if "last_out_path" not in st.session_state:
        st.session_state.last_out_path = None

    up = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"], key="image_uploader")
    preview_col1, preview_col2 = st.columns(2)
    run_area = st.container()
    result_area = st.container()

    if up:
        img = Image.open(up).convert("RGB")
        rgb = np.array(img)

        # detect
        faces = detect_faces(rgb)
        plates = detect_plates_yolo(rgb, yolo_plate_weights) or detect_plates_haar(rgb)

        # build mask
        face_mask = boxes_to_mask(rgb, faces, face_dilate)
        plate_mask = boxes_to_mask(rgb, plates, plate_dilate)
        mask = np.maximum(face_mask, plate_mask)
        over = overlay_mask(rgb, mask)

        with preview_col1:
            st.image(img, caption="Original", use_container_width=True)
        with preview_col2:
            st.image(over, caption="Mask overlay", use_container_width=True)

        # quick sanity: empty mask?
        if (mask > 0).sum() == 0:
            st.warning("No faces/plates detected — mask is empty. Try a clearer image or increase dilation.")
        
        # action button + spinner
        if run_area.button("Run generative inpainting", type="primary", use_container_width=False):
            with st.spinner("Inpainting… first run may download the model (~1–2 GB)…"):
                out_img = inpaint(Image.fromarray(rgb), Image.fromarray(mask),
                                  steps=inference_steps, guidance=guidance)
                if draw_fake and len(plates) > 0:
                    out_img = draw_fake_plate_text(out_img, plates)

                # persist + save
                os.makedirs("outputs", exist_ok=True)
                out_path = os.path.join("outputs", "anonymized_image.jpg")
                out_img.save(out_path)
                st.session_state.last_out_img = out_img
                st.session_state.last_out_path = out_path

                # save inputs for Evaluation tab
                os.makedirs("data", exist_ok=True)
                Image.fromarray(mask).save("data/mask.png")
                img.save("data/input.jpg")

    # Always render the result section so it's visible after rerun
    if st.session_state.last_out_img is not None:
        st.markdown("### Anonymized result")
        st.image(st.session_state.last_out_img, use_container_width=True)
        # download button
        buf = io.BytesIO()
        st.session_state.last_out_img.save(buf, format="JPEG", quality=95)
        st.download_button("Download anonymized image",
                           data=buf.getvalue(),
                           file_name="anonymized_image.jpg",
                           mime="image/jpeg")
        if st.session_state.last_out_path:
            st.success(f"Saved: {st.session_state.last_out_path}")

# -------- VIDEO TAB --------
with tabs[1]:
    st.subheader("Video anonymization")
    vid = st.file_uploader("Upload a short MP4", type=["mp4"])
    max_frames = st.number_input("Max frames (for demo)", min_value=0, value=180)
    if vid and st.button("Process video"):
        import tempfile
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(tmp_in,"wb") as f: f.write(vid.read())
        cap = cv2.VideoCapture(tmp_in)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join("outputs","anonymized_video.mp4")
        os.makedirs("outputs", exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok: break
            if max_frames and idx>=max_frames: break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            faces = detect_faces(rgb)
            plates = detect_plates_yolo(rgb, yolo_plate_weights) or detect_plates_haar(rgb)
            mask = boxes_to_mask(rgb, faces, face_dilate)
            mask = np.maximum(mask, boxes_to_mask(rgb, plates, plate_dilate))
            out_img = inpaint(Image.fromarray(rgb), Image.fromarray(mask), steps=inference_steps, guidance=guidance)
            if draw_fake and len(plates)>0:
                out_img = draw_fake_plate_text(out_img, plates)
            writer.write(cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR))
            idx += 1
        cap.release(); writer.release()
        st.success(f"Saved {out_path}")
        st.video(out_path)

# -------- EVAL TAB --------
with tabs[2]:
    st.subheader("Evaluation (PSNR/SSIM on non-masked regions)")
    o = st.file_uploader("Original image", type=["jpg","jpeg","png"], key="eval_o")
    a = st.file_uploader("Anonymized image", type=["jpg","jpeg","png"], key="eval_a")
    m = st.file_uploader("Mask image (white=changed)", type=["png"], key="eval_m")
    if o and a and m and st.button("Compute metrics"):
        from skimage.color import rgb2gray
        oimg = np.array(Image.open(o).convert("RGB"))
        aimg = np.array(Image.open(a).convert("RGB"))
        mask = np.array(Image.open(m).convert("L"))
        if oimg.shape != aimg.shape:
            aimg = cv2.resize(aimg, (oimg.shape[1], oimg.shape[0]))
        keep = mask==0
        if keep.sum()==0:
            st.warning("No unmasked pixels."); 
        else:
            p=[]
            for c in range(3):
                p.append(psnr(oimg[:,:,c][keep], aimg[:,:,c][keep], data_range=255))
            P = float(np.mean(p))
            S = ssim((rgb2gray(oimg)*255).astype(np.uint8), (rgb2gray(aimg)*255).astype(np.uint8), data_range=255)
            st.write(f"**PSNR (non-masked):** {P:.2f} dB")
            st.write(f"**SSIM:** {S:.4f}")

# -------- DATASETS TAB --------
with tabs[3]:
    st.subheader("Datasets (quick samples)")
    st.write("Download a small sample from popular datasets for testing.")
    col1,col2 = st.columns(2)
    with col1:
        if st.button("Download small WIDER FACE sample"):
            st.info("Open the WIDER FACE page and fetch a few images manually, or script a downloader. For now, drag-drop a few images into data/.")
    with col2:
        if st.button("Download small CCPD sample"):
            st.info("Same idea for CCPD (plates). For a demo, add a handful of plate images into data/ and try the Image tab.")
