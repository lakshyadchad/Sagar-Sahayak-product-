import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import time
import pandas as pd
import os
import datetime
from PIL import Image
from torchvision import transforms

# --- IMPORTS FROM YOUR MODULES ---
from model_architecture import UNetGenerator # The U-Net Class (Phase 2)
from ultralytics import YOLO                 # The YOLO Lib (Phase 3)  # type: ignore
from metrics import ImageMetrics             # The Math Class (Step 3)
from auth_db import verify_user, add_user    # Authentication

# --- CONFIGURATION ---
# UPDATE THESE PATHS TO MATCH YOUR FILE LOCATIONS EXACTLY
ENHANCER_PATH = r''
DETECTOR_PATH = r''
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. PAGE SETUP & CSS INJECTION ---
st.set_page_config(
    page_title="SAGAR SAHAYAK - Command Center",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS LOADER ---
def load_css(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, 'css', file_name)
    try:
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass # Fail silently if CSS is missing to avoid corruption

# Call the function
load_css("style.css")

# --- 2. MODEL LOADER (CACHED) ---
@st.cache_resource
def load_system():
    print("--- Loading System Models ---")
    
    # A. Load Enhancer (U-Net)
    enhancer = UNetGenerator().to(DEVICE)
    # weights_only=True is a security fix for newer PyTorch versions
    enhancer.load_state_dict(torch.load(ENHANCER_PATH, map_location=DEVICE)) 
    enhancer.eval()
    
    # B. Load Detector (YOLO)
    detector = YOLO(DETECTOR_PATH)
    
    # C. Load Metrics Tool
    metrics = ImageMetrics()
    
    print(f"--- Models Loaded on {DEVICE.upper()} ---")
    return enhancer, detector, metrics

# --- 3. HELPER: CONTRAST BOOSTER ---
def apply_clahe(image):
    """Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# --- 4. AUTHENTICATION SYSTEM ---
def login_page():
    """Display login page"""
    st.markdown("<h1 style='text-align: center;'>⚓ SAGAR SAHAYAK</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Underwater Surveillance System</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if username and password:
                        if verify_user(username, password):
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
                    else:
                        st.warning("Please enter both username and password")
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Username", placeholder="Choose username", key="reg_user")
                new_password = st.text_input("Password", type="password", placeholder="Choose password", key="reg_pass")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                register = st.form_submit_button("Register", use_container_width=True)
                
                if register:
                    if new_username and new_password and confirm_password:
                        if new_password == confirm_password:
                            if add_user(new_username, new_password):
                                st.success("✅ Registration successful! Please login.")
                            else:
                                st.error("❌ Username already exists")
                        else:
                            st.error("❌ Passwords do not match")
                    else:
                        st.warning("Please fill all fields")
        
        st.markdown("---")
        st.info("**Default Credentials:**\nUsername: `admin`\nPassword: `admin123`")

def check_authentication():
    """Check if user is authenticated"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        login_page()
        st.stop()

# --- 5. MAIN APP LOGIC ---
def main():
    # Check authentication first
    check_authentication()
    
    # Logout button in sidebar
    with st.sidebar:
        st.markdown(f"👤 **User:** {st.session_state.username}")
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        st.markdown("---")
    
    # Header
    st.markdown("# ⚓ SAGAR SAHAYAK")
    st.markdown("### AI-Driven Underwater Surveillance System")
    st.divider()

    # Load Models with a spinner
    with st.spinner(f"Initializing Neural Networks on {DEVICE.upper()}..."):
        try:
            enhancer, detector, metrics = load_system()
        except Exception as e:
            st.error(f"System Failure: Could not load models. {e}")
            st.stop()

    # --- SIDEBAR CONTROLS ---
    st.sidebar.markdown("### ⚙️ MISSION PARAMETERS")
    
    input_mode = st.sidebar.radio("Feed Source", ["Upload Video File", "Live Camera Feed", "🔬 Validation Mode (PSNR/SSIM)"])
    conf_thresh = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.70, 0.05)
    iou_thresh = st.sidebar.slider("Overlap Threshold (IOU)", 0.0, 1.0, 0.45, 0.05)
    enable_clahe = st.sidebar.toggle("Enable CLAHE Booster", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Engine: {DEVICE} | Backend: PyTorch {torch.__version__}")

    # --- DEFINE AI TRANSFORM (Used by both validation and video modes) ---
    ai_transform = transforms.Compose([
        transforms.Resize((384, 384)),  # Increased to 384x384 for better quality while maintaining FPS
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # --- VIDEO SOURCE HANDLING ---
    cap = None
    if input_mode == "🔬 Validation Mode (PSNR/SSIM)":
        st.markdown("## 🔬 Validation Mode: Calculate Real PSNR & SSIM")
        st.info("Upload a **murky/degraded image** and its corresponding **ground truth/clean reference** to calculate accurate PSNR and SSIM metrics.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📤 Upload Murky Image")
            murky_file = st.file_uploader("Degraded/Underwater Image", type=['jpg', 'jpeg', 'png'], key="murky")
        
        with col2:
            st.markdown("### 📤 Upload Ground Truth")
            ground_truth_file = st.file_uploader("Clean/Reference Image", type=['jpg', 'jpeg', 'png'], key="gt")
        
        if murky_file and ground_truth_file:
            # Load images
            murky_img = Image.open(murky_file).convert('RGB')
            gt_img = Image.open(ground_truth_file).convert('RGB')
            
            # Convert to numpy arrays
            murky_np = np.array(murky_img)
            gt_np = np.array(gt_img)
            
            # Convert RGB to BGR for OpenCV
            murky_bgr = cv2.cvtColor(murky_np, cv2.COLOR_RGB2BGR)
            gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
            
            # Resize ground truth to match murky image size if different
            if murky_bgr.shape[:2] != gt_bgr.shape[:2]:
                gt_bgr = cv2.resize(gt_bgr, (murky_bgr.shape[1], murky_bgr.shape[0]))
            
            # Display original images
            st.markdown("### 📸 Input Images")
            display_col1, display_col2, display_col3 = st.columns(3)
            
            with display_col1:
                st.image(murky_img, caption="Murky/Degraded Image", use_container_width=True)
            
            with display_col2:
                st.image(gt_img, caption="Ground Truth/Reference", use_container_width=True)
              # --- ENHANCEMENT PIPELINE ---
            with st.spinner("Enhancing murky image with AI..."):
                # Prepare image for U-Net
                murky_pil = Image.fromarray(cv2.cvtColor(murky_bgr, cv2.COLOR_BGR2RGB))
                murky_tensor = ai_transform(murky_pil).unsqueeze(0).to(DEVICE)  # type: ignore[attr-defined]
                
                # Enhance
                with torch.no_grad():
                    enhanced_tensor = enhancer(murky_tensor)
                
                # Convert back to image
                enhanced_np = enhanced_tensor.squeeze(0).cpu().numpy()
                enhanced_np = ((enhanced_np * 0.5 + 0.5) * 255).astype(np.uint8)
                enhanced_bgr = cv2.cvtColor(np.transpose(enhanced_np, (1, 2, 0)), cv2.COLOR_RGB2BGR)
                enhanced_bgr = cv2.resize(enhanced_bgr, (murky_bgr.shape[1], murky_bgr.shape[0]))
                
                # Apply CLAHE if enabled
                if enable_clahe:
                    enhanced_bgr = apply_clahe(enhanced_bgr)
            
            with display_col3:
                st.image(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB), caption="AI Enhanced Result", use_container_width=True)
            
            st.divider()
            
            # --- CALCULATE METRICS ---
            st.markdown("### 📊 Quality Metrics Analysis")
            
            # Calculate UIQM for all three
            uiqm_murky = metrics.calculate_uiqm(murky_bgr)
            uiqm_enhanced = metrics.calculate_uiqm(enhanced_bgr)
            uiqm_gt = metrics.calculate_uiqm(gt_bgr)
            
            # Calculate PSNR & SSIM: Enhanced vs Ground Truth
            psnr_score = metrics.calculate_psnr(enhanced_bgr, gt_bgr)
            ssim_score = metrics.calculate_ssim(enhanced_bgr, gt_bgr)
            
            # Calculate PSNR & SSIM: Murky vs Ground Truth (baseline)
            psnr_baseline = metrics.calculate_psnr(murky_bgr, gt_bgr)
            ssim_baseline = metrics.calculate_ssim(murky_bgr, gt_bgr)
            
            # Display metrics
            metric_cols = st.columns(6)
            
            with metric_cols[0]:
                improvement = ((psnr_score - psnr_baseline) / psnr_baseline * 100) if psnr_baseline > 0 else 0
                st.metric("PSNR (Enhanced)", f"{psnr_score:.2f} dB", f"+{improvement:.1f}%")
            
            with metric_cols[1]:
                improvement = ((ssim_score - ssim_baseline) / ssim_baseline * 100) if ssim_baseline > 0 else 0
                st.metric("SSIM (Enhanced)", f"{ssim_score:.4f}", f"+{improvement:.1f}%")
            
            with metric_cols[2]:
                st.metric("PSNR (Baseline)", f"{psnr_baseline:.2f} dB", "Original")
            
            with metric_cols[3]:
                st.metric("SSIM (Baseline)", f"{ssim_baseline:.4f}", "Original")
            
            with metric_cols[4]:
                improvement = ((uiqm_enhanced - uiqm_murky) / uiqm_murky * 100) if uiqm_murky > 0 else 0
                st.metric("UIQM (Enhanced)", f"{uiqm_enhanced:.3f}", f"+{improvement:.1f}%")
            
            with metric_cols[5]:
                st.metric("UIQM (Murky)", f"{uiqm_murky:.3f}", "Original")
            
            st.divider()
            
            # Interpretation Guide
            st.markdown("### 📖 Interpretation Guide")
            
            col_guide1, col_guide2, col_guide3 = st.columns(3)
            
            with col_guide1:
                st.markdown("**PSNR (Peak Signal-to-Noise Ratio)**")
                st.markdown(f"- Enhanced: `{psnr_score:.2f} dB`")
                if psnr_score > 30:
                    st.success("✅ Excellent quality (>30 dB)")
                elif psnr_score > 20:
                    st.info("✓ Good quality (20-30 dB)")
                else:
                    st.warning("⚠ Poor quality (<20 dB)")
                st.caption(f"Improvement: {psnr_score - psnr_baseline:.2f} dB over baseline")
            
            with col_guide2:
                st.markdown("**SSIM (Structural Similarity)**")
                st.markdown(f"- Enhanced: `{ssim_score:.4f}`")
                if ssim_score > 0.9:
                    st.success("✅ Excellent similarity (>0.9)")
                elif ssim_score > 0.7:
                    st.info("✓ Good similarity (0.7-0.9)")
                else:
                    st.warning("⚠ Poor similarity (<0.7)")
                st.caption(f"Improvement: {ssim_score - ssim_baseline:.4f} over baseline")
            
            with col_guide3:
                st.markdown("**UIQM (No Reference Needed)**")
                st.markdown(f"- Enhanced: `{uiqm_enhanced:.3f}`")
                if uiqm_enhanced > uiqm_murky:
                    st.success(f"✅ Enhanced is better (+{((uiqm_enhanced-uiqm_murky)/uiqm_murky*100):.1f}%)")
                else:
                    st.warning("⚠ Enhancement did not improve UIQM")
                st.caption("Higher UIQM = Better underwater image quality")
            
            # Summary
            st.markdown("---")
            st.markdown("### 🎯 Summary")
            
            if psnr_score > psnr_baseline and ssim_score > ssim_baseline:
                st.success(f"✅ **Enhancement Successful!** The AI-enhanced image is closer to ground truth with PSNR improved by {psnr_score - psnr_baseline:.2f} dB and SSIM improved by {ssim_score - ssim_baseline:.4f}.")
            elif psnr_score > psnr_baseline or ssim_score > ssim_baseline:
                st.info(f"✓ **Partial Improvement**: Some metrics improved over the baseline.")
            else:
                st.warning(f"⚠ **Enhancement Issue**: Metrics did not improve over baseline. Check algorithm parameters.")
        
        else:
            st.warning("Please upload both murky image and ground truth reference to calculate metrics.")
        
        return  # Exit early, don't run video processing
        
    elif input_mode == "Upload Video File":
        uploaded = st.sidebar.file_uploader("Select Mission Footage", type=['mp4', 'avi', 'mov'])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)
    else:
        # Live Feed logic
        cam_id = st.sidebar.number_input("Camera Index", 0, 5, 0)
        if st.sidebar.button("Start Live Feed"):
            cap = cv2.VideoCapture(cam_id)

    # --- THE DASHBOARD LOOP ---
    if cap:
        # 1. Create Layout Placeholders (The "Screen")
        col_raw, col_proc = st.columns(2)
        with col_raw: 
            st.markdown("**📡 RAW OPTICAL FEED**")
            raw_ph = st.empty()
        with col_proc: 
            st.markdown("**🎯 ENHANCED TARGET FEED**")
            proc_ph = st.empty()

        st.markdown("### 📊 REAL-TIME TELEMETRY")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        met_fps = m1.empty()
        met_uiqm = m2.empty()
        met_psnr = m3.empty()
        met_ssim = m4.empty()
        met_obj = m5.empty()
        met_log = m6.empty()

        prev_time = 0
        
        # NEW: Mission Log Data Structure
        mission_log = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # --- A. PREPROCESSING ---
                # Resize to 720p (1280x720) for HD processing
                frame_720p = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                raw_display = frame_720p.copy()
                  # --- B. ENHANCEMENT PIPELINE (Optimized for FPS) ---
                # Convert BGR to RGB once
                frame_rgb = cv2.cvtColor(frame_720p, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_tensor = ai_transform(img_pil).unsqueeze(0).to(DEVICE)  # type: ignore[attr-defined]
                
                with torch.no_grad():
                    clean_tensor = enhancer(img_tensor)
                
                # Efficient tensor to numpy conversion
                clean_img = clean_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
                clean_img = (clean_img * 0.5 + 0.5)  # Denormalize [-1,1] to [0,1]
                clean_img = np.clip(clean_img, 0, 1)
                clean_uint8 = (clean_img * 255).astype(np.uint8)
                
                # Upscale back to 720p with LINEAR (faster than LANCZOS)
                clean_resized = cv2.resize(clean_uint8, (1280, 720), interpolation=cv2.INTER_LINEAR)
                clean_bgr = cv2.cvtColor(clean_resized, cv2.COLOR_RGB2BGR)

                if enable_clahe:
                    clean_bgr = apply_clahe(clean_bgr)

                # --- C. DETECTION PIPELINE ---
                det_input = cv2.resize(clean_bgr, (640, 640))
                results = detector(det_input, conf=conf_thresh, iou=iou_thresh, verbose=False)
                
                annotated_frame = results[0].plot()

                # --- D. METRICS & LOGGING ---
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
                prev_time = curr_time
                
                uiqm_score = metrics.calculate_uiqm(clean_bgr)
                
                
                num_threats = len(results[0].boxes)

                # NEW: Record Data
                mission_log.append({
                    "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "Threats": num_threats,
                    "UIQM": round(uiqm_score, 2),
                    "FPS": round(fps, 1)
                })

                # --- E. UI UPDATES ---
                # Display at 720p HD resolution (no additional resize needed)
                raw_rgb = cv2.cvtColor(raw_display, cv2.COLOR_BGR2RGB)
                annotated_resized = cv2.resize(annotated_frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                annotated_rgb = cv2.cvtColor(annotated_resized, cv2.COLOR_BGR2RGB)
                
                raw_ph.image(raw_rgb, channels="RGB", use_container_width=True)
                proc_ph.image(annotated_rgb, channels="RGB", use_container_width=True)

                met_fps.metric("System Latency", f"{int(fps)} FPS")
                
                delta_val = "Normal"
                if uiqm_score < 3.0: delta_val = "- Low Visibility"
                met_uiqm.metric("Image Quality (UIQM)", f"{uiqm_score:.2f}", delta_val)
                
                if num_threats > 0:
                    met_obj.metric("Active Threats", f"{num_threats}", "ENGAGEMENT REQUIRED", delta_color="inverse")
                    met_log.error(f"⚠️ THREAT DETECTED")
                else:
                    met_obj.metric("Active Threats", "0", "Sector Clear")
                    met_log.info("System Scanning...")

        finally:
            cap.release()
            
        # --- F. POST-MISSION REPORTING (NEW) ---
        if len(mission_log) > 0:
            st.divider()
            st.subheader("📑 Mission Debrief Report")
            
            df = pd.DataFrame(mission_log)
            
            # 1. Metric Charts
            c1, c2 = st.columns(2)
            with c1:
                st.line_chart(df, x="Timestamp", y="UIQM", color="#00FF00")
                st.caption("Visibility Score Over Time")
            with c2:
                st.area_chart(df, x="Timestamp", y="Threats", color="#FF0000")
                st.caption("Threat Detection Timeline")
            
            # FPS chart
            st.line_chart(df, x="Timestamp", y="FPS", color="#0088FF")
            st.caption("System Performance (FPS)")
                
            # 2. Data Download with user info
            # Add user info to dataframe
            df['User'] = st.session_state.username
            df['Mission_ID'] = f"MISSION_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Mission Log (CSV)",
                data=csv,
                file_name=f"mission_log_{st.session_state.username}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
            )

if __name__ == "__main__":
    main()