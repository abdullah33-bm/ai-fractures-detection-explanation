# --- ADD THESE LINES AT THE TOP ---
from dotenv import load_dotenv
load_dotenv() # This line loads the variables from .env into the environment
# --- END OF ADDED LINES ---

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO
import google.generativeai as genai
import os
# --- ADD THESE IMPORTS ---
from fpdf import FPDF # Correct import for fpdf2 library
from datetime import datetime
# At the top of app.py
import pyperclip
import hashlib
# --- END OF ADDED IMPORTS ---

# --- GOOGLE GEMINI API CONFIGURATION ---
try:
    # Use the environment variable loaded from .env
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ GOOGLE_API_KEY not found. Make sure it's set in your .env file.")
        st.stop()
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring Google Gemini API: {e}. Check your .env file and API key.")
    st.stop()


# Page config
st.set_page_config(
    page_title="Fracture Detection AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fantasy theme
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Raleway:wght@300;400;600&display=swap');

    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0e0;
    }

    /* Headers */
    h1 {
        font-family: 'Cinzel', serif;
        background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5rem !important;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(168, 237, 234, 0.3);
    }

    h2, h3 {
        font-family: 'Cinzel', serif;
        color: #a8edea;
    }

    /* Card-like containers */
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(168, 237, 234, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin: 1rem 0;
        font-family: 'Raleway', sans-serif; /* Apply Raleway to explanation card */
    }
     .main-card p { /* Ensure paragraphs inside the card also use the font */
        font-family: 'Raleway', sans-serif;
    }


    /* File uploader styling */
    .stFileUploader > div > div > button { /* Target button inside uploader */
       color: #a8edea; /* Make text match theme */
    }
    .stFileUploader > div > div > div { /* Target text area inside uploader */
        color: #e0e0e0; /* Make text match theme */
    }
    .stFileUploader > label { /* Target the label ("Choose an X-Ray image...") */
         color: #e0e0e0 !important; /* Make label text match theme */
         font-family: 'Raleway', sans-serif !important;
    }


    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-family: 'Raleway', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Info boxes */
    .stAlert {
        background: rgba(168, 237, 234, 0.1);
        border-left: 4px solid #a8edea;
        border-radius: 10px;
        font-family: 'Raleway', sans-serif; /* Apply Raleway */
    }
     .stAlert p { /* Ensure paragraphs inside alerts use the font */
        font-family: 'Raleway', sans-serif;
    }


    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSelectbox label { /* Style selectbox label */
        font-family: 'Raleway', sans-serif; /* Apply Raleway to sidebar text */
    }


    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #a8edea;
        font-family: 'Cinzel', serif;
    }
    [data-testid="stMetricLabel"] p { /* Target metric label */
        font-family: 'Raleway', sans-serif; /* Apply Raleway */
        color: #fed6e3; /* Match header secondary color */
    }


    /* Spinner */
    .stSpinner > div {
        border-top-color: #a8edea !important;
    }
    .stSpinner > div > p { /* Target spinner text */
       font-family: 'Raleway', sans-serif;
       color: #e0e0e0;
    }

     /* Expander header */
    .streamlit-expanderHeader {
        font-family: 'Raleway', sans-serif;
        font-size: 1.1rem;
        color: #fed6e3; /* Match header secondary color */
    }
    /* General text */
    p, .stMarkdown p, li, .stMarkdown li {
       font-family: 'Raleway', sans-serif;
       color: #e0e0e0; /* Ensure default text is light */
    }
    .stImage > img {
        border-radius: 10px; /* Slightly round image corners */
    }

</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO('model.pt') # Use model.pt
        # st.success("🦴 YOLOv8 Detection Model Loaded") # Optional: uncomment if needed
        return model
    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {str(e)}")
        return None

# --- GEMINI EXPLANATION FUNCTION ---
@st.cache_data(show_spinner=False)
def get_fracture_explanation(image_bytes, detection_summary, detected_fractures, conf_threshold, language="English", image_hash: str = None):
    """Generate an explanation for a given image using image bytes and a textual detection summary.

    This function is safe to cache because it only accepts hashable/serializable inputs (bytes and strings).
    Pass `image_hash` if you want caching to key on a precomputed image hash.
    """
    if not detected_fractures:
        return "No fractures were detected by the AI model above the set confidence threshold."

    prompt_header = (
        f"You are an AI assistant specialized in analyzing X-ray images for educational purposes. Respond ONLY in {language}. "
        "Analyze this X-ray image for bone fractures. "
        "Describe what you see in the X-ray, highlighting any visible breaks or anomalies. "
        "Based on the AI detection results provided below, explain *why* a fracture might be identified in the specified areas. "
        "Focus on visual characteristics like bone discontinuity, displacement, or abnormal lines. "
        f"Provide a concise, clear explanation in simple terms, strictly in {language}. Do not give medical advice or diagnosis."
    )

    # Ensure detection_summary already contains details about detections and threshold
    gemini_model = genai.GenerativeModel('models/gemini-2.5-pro') # Using 2.5 Pro

    try:
        response = gemini_model.generate_content(
            [prompt_header, {"mime_type": "image/jpeg", "data": image_bytes}, detection_summary]
        )
        if response.parts:
            return response.text
        else:
            block_reason = "Unknown reason"
            try:
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    block_reason = response.prompt_feedback.block_reason
            except Exception:
                pass
            st.warning(f"Gemini did not generate an explanation. Reason: {block_reason}")
            return "The AI could not provide an explanation for this image, possibly due to safety filters or content restrictions."
    except Exception as e:
        err_str = str(e)
        # Detect common leaked/revoked API key message patterns (403 or leaked)
        if '403' in err_str or 'leaked' in err_str.lower() or 'api key' in err_str.lower():
            st.error("❌ Gemini API returned 403 — the configured API key was rejected or reported leaked.")
            st.error("Immediate action: revoke the leaked key in Google Cloud, create a new API key, and update your `.env` `GOOGLE_API_KEY`.")
            st.info("After rotating the key, restart the app (stop and `streamlit run app.py`). Do not commit API keys to source control.")
            return ("The AI explanation service is unavailable because the configured API key was rejected. "
                    "Rotate your API key and set `GOOGLE_API_KEY` in the environment to resume.")
        else:
            st.error(f"❌ Error generating explanation from Gemini: {e}")
            return "An error occurred while communicating with the explanation AI."

# --- IMAGE PROCESSING FUNCTION ---
def process_image(image, model, conf_threshold):
    """Process image and detect fractures"""
    image_rgb = image.convert("RGB")
    img_array = np.array(image_rgb)
    results = model(img_array, conf=conf_threshold)
    annotated_img = results[0].plot() # BGR numpy array
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    return annotated_img_rgb, results[0] # Return RGB numpy array and the results object

# --- PDF GENERATION FUNCTION ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14) # Changed font to Arial
        self.cell(0, 10, 'Fracture Detection AI Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        page_width = self.w - 2 * self.l_margin
        self.set_xy(self.l_margin + page_width - 40, -15)
        self.set_font('Arial', 'I', 8)
        self.cell(40, 10, timestamp, 0, 0, 'R')

# --- PDF GENERATION FUNCTION ---
# ... (PDF Class definition remains the same) ...

def generate_report_pdf(annotated_image_pil, explanation_text, original_filename):
    """Generates a PDF report containing the annotated image and explanation."""
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Add Annotated Image ---
    img_byte_arr = io.BytesIO()
    # Ensure image is RGB before saving for PDF
    annotated_image_pil.convert("RGB").save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0) # <-- Ensure the stream position is at the beginning

    # ... (Calculate image dimensions - no changes here) ...
    page_width = pdf.w - 2 * pdf.l_margin
    img_width, img_height = annotated_image_pil.size
    aspect_ratio = img_height / img_width
    display_width = page_width * 0.8
    display_height = display_width * aspect_ratio
    max_img_height = pdf.h - pdf.t_margin - pdf.b_margin - 40
    if display_height > max_img_height:
        display_height = max_img_height
        display_width = display_height / aspect_ratio
    x_pos = (pdf.w - display_width) / 2
    y_pos = pdf.get_y()

    # --- Pass the BytesIO stream directly to pdf.image ---
    pdf.image(img_byte_arr, x=x_pos, y=y_pos, w=display_width, type='PNG') # Specify type='PNG'
    # --- End Change ---
    pdf.set_y(y_pos + display_height + 10)

    # ... (Rest of the function: Add Explanation, Disclaimer, Return Bytes) ...
    # --- Add Explanation Text ---
    pdf.set_font('Arial', 'B', 12)
    # ... (rest remains the same)

    # Return PDF content as bytes
    try:
        # Request bytes directly using dest='buffer'
        pdf_bytes = pdf.output(dest='buffer')
        return pdf_bytes
    except Exception as e:
        st.error(f"Error during PDF byte generation: {e}")
        return b""

# --- MAIN APP FUNCTION ---
def main():
    # --- Corrected: Initialize session state variables ---
    if 'last_explained_image_name' not in st.session_state:
        st.session_state.last_explained_image_name = None
    if 'last_explained_image_hash' not in st.session_state:
        st.session_state.last_explained_image_hash = None
    if 'explanation_generated_for_this_image' not in st.session_state:
        st.session_state.explanation_generated_for_this_image = False
    if 'current_explanation' not in st.session_state:
        st.session_state.current_explanation = "" # Store the explanation text
    if 'selected_language' not in st.session_state:
         st.session_state.selected_language = "English" # Default language
    if 'last_explained_language' not in st.session_state:
         st.session_state.last_explained_language = None
    # --- End Initialization ---

    # Header
    st.markdown("<h1>🔮 Fracture Detection AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-family: Raleway, sans-serif; font-size: 1.2rem; color: #fed6e3; margin-bottom: 2rem;'>Advanced Medical Imaging Analysis powered by Deep Learning</p>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---") # Added separator

        # --- Corrected: Uncommented Slider ---
        conf_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25, # Default value
            step=0.05,
            help="Minimum confidence for a detection to be shown. Lower values detect more, potentially including false positives."
        )
        # --- End Slider ---

        # --- Language Selector ---
        st.markdown("---") # Separator
        languages = ["English", "Arabic", "Spanish", "French"] # Add more as needed
        selected_language = st.selectbox(
            "Explanation Language",
            options=languages,
            index=languages.index(st.session_state.selected_language), # Set default based on session state
            help="Choose the language for the AI explanation."
        )
        # Update session state if language changes
        language_changed_in_sidebar = False
        if selected_language != st.session_state.selected_language:
            st.session_state.selected_language = selected_language
            st.session_state.explanation_generated_for_this_image = False # Force regen if language changes
            language_changed_in_sidebar = True # Flag that language changed
        # --- END Language Selector ---

        st.markdown("---") # Separator
        st.markdown("### 📊 Model Info")
        st.info("**Detection Model:** YOLOv8 (Fine-tuned)\n\n**Explanation Model:** Gemini 2.5 Pro\n\n**Task:** Fracture Detection & Explanation\n\n**Input:** X-Ray Images") # Corrected model name display

        st.markdown("---")
        st.markdown("### 🔬 How it works")
        st.markdown("""
        1.  **Upload** an X-Ray image 🖼️.
        2.  **Adjust** the detection confidence threshold ⚙️.
        3.  The **YOLOv8** model identifies potential fractures 🔍.
        4.  Detected areas are **highlighted** on the image 🟥.
        5.  If fractures are found **and** the image/language is new for this session, **Google Gemini** provides a text explanation ✨.
        6.  **Download** the annotated image or a full PDF report 📄.
        """) # Updated How it works

    # Main content layout
    col1, col2 = st.columns([1, 1], gap="large")

    # --- Corrected: Define all Placeholders ---
    original_image_placeholder = col1.empty()
    image_info_placeholder = col1.empty()
    results_placeholder = col2.empty()
    details_placeholder = col2.empty() # <<< ADD THIS LINE
    explanation_placeholder = col2.empty()
    download_buttons_placeholder = col2.empty()
    # --- End Placeholders ---


    with col1:
        st.markdown("### 📤 Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose an X-Ray image...",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            label_visibility="collapsed",
            help="Upload a clear X-Ray image for fracture detection"
        )

    # --- Process only if a file is uploaded ---
    if uploaded_file is not None:
        # Compute a content hash for the uploaded file so different images with the same filename are detected
        try:
            image_bytes = uploaded_file.getvalue()
            image_hash = hashlib.sha256(image_bytes).hexdigest()
        except Exception:
            image_bytes = None
            image_hash = None

        # Check if the uploaded file content has changed since last explanation
        file_changed = st.session_state.last_explained_image_hash != image_hash
        if file_changed:
            # Reset explanation flag and stored text if file content changed
            st.session_state.explanation_generated_for_this_image = False
            st.session_state.current_explanation = ""
            st.session_state.last_explained_image_name = uploaded_file.name # Track the new filename
            st.session_state.last_explained_image_hash = image_hash
            st.session_state.last_explained_language = None # Reset language tracker for new file

        # Load and display the original image
        image = Image.open(uploaded_file)
        original_image_placeholder.image(image, caption="Original X-Ray", use_container_width=True)
        image_info_placeholder.markdown(f"**Image Info:** `{uploaded_file.name}` ({image.size[0]}x{image.size[1]})")

        # Load YOLO model (uses cache)
        yolo_model = load_yolo_model()

        if yolo_model is not None:
            # Wrap processing in a try-except block
            try:
                # Perform detection with spinner
                with st.spinner("🔮 Analyzing X-Ray with YOLOv8..."):
                    # --- Corrected: Use conf_threshold from slider ---
                    annotated_img_rgb, results_obj = process_image(image, yolo_model, conf_threshold)
                    # --- End Correction ---

                # Display YOLO results
                results_placeholder.image(annotated_img_rgb, caption="Detected Fractures", use_container_width=True)
                num_detections = len(results_obj.boxes) if hasattr(results_obj, 'boxes') else 0

                # --- Handle Detections ---
                if num_detections > 0:
                    st.success(f"✅ {num_detections} potential fracture region(s) detected!")

                    # Show detection details in main area (col2)
                    st.markdown("### 📋 Detection Details")
                    # Safely access results properties
                    if hasattr(results_obj, 'boxes') and hasattr(results_obj, 'names') and results_obj.names:
                        for idx, box in enumerate(results_obj.boxes):
                            conf = float(box.conf[0]) if box.conf is not None and len(box.conf) > 0 else 0.0
                            cls = int(box.cls[0]) if box.cls is not None and len(box.cls) > 0 else -1
                            class_name = results_obj.names.get(cls, "Unknown")
                            with st.expander(f"Detection {idx + 1}: {class_name}", expanded=idx < 3):
                                exp_col_a, exp_col_b = st.columns(2)
                                with exp_col_a:
                                    st.metric("Confidence", f"{conf:.2%}")
                                with exp_col_b:
                                    st.metric("Class", class_name)
                    else:
                        st.warning("Could not display detection details.")


                    # --- Corrected: Explanation Logic with session state ---
                    language_changed = st.session_state.selected_language != st.session_state.last_explained_language
                    needs_explanation = num_detections > 0 and (not st.session_state.explanation_generated_for_this_image or language_changed)

                    if needs_explanation:
                        st.markdown(f"### 📝 AI Explanation ({st.session_state.selected_language})")
                        with st.spinner(f"✨ Asking Gemini for an explanation in {st.session_state.selected_language}..."):
                            # Build a textual detection summary to pass to the explanation function
                            detection_summary = f"\n\nAI Model Detections (confidence threshold > {conf_threshold:.2f}):\n"
                            if results_obj and hasattr(results_obj, 'boxes') and hasattr(results_obj, 'names') and results_obj.names:
                                for idx, box in enumerate(results_obj.boxes):
                                    conf = float(box.conf[0]) if box.conf is not None and len(box.conf) > 0 else 0.0
                                    cls = int(box.cls[0]) if box.cls is not None and len(box.cls) > 0 else -1
                                    class_name = results_obj.names.get(cls, "Unknown")
                                    detection_summary += f"- Detected '{class_name}' with {conf:.2f} confidence.\n"
                            else:
                                detection_summary += "- No detection details available.\n"

                            explanation = get_fracture_explanation(
                                image_bytes,
                                detection_summary,
                                True,
                                conf_threshold, # Pass correct threshold
                                language=st.session_state.selected_language,
                                image_hash=image_hash
                            )
                            st.session_state.current_explanation = explanation # Store explanation
                            st.session_state.explanation_generated_for_this_image = True # Mark as generated
                            st.session_state.last_explained_language = st.session_state.selected_language # Store language used
                            explanation_placeholder.markdown(f"<div class='main-card' style='padding: 1.5rem; margin-top: 0.5rem;'>{explanation}</div>", unsafe_allow_html=True)
                    elif num_detections > 0 and st.session_state.current_explanation:
                        # If already generated for this image/language, display stored one
                         st.markdown(f"### 📝 AI Explanation ({st.session_state.last_explained_language})")
                         explanation_placeholder.markdown(f"<div class='main-card' style='padding: 1.5rem; margin-top: 0.5rem;'>{st.session_state.current_explanation}</div>", unsafe_allow_html=True)
                    # --- End Explanation Logic ---

                    # --- NEW: Add Copy Button ---
                    if st.button("📋 Copy Explanation Text", key="copy_explanation_btn"):
                        try:
                            pyperclip.copy(st.session_state.current_explanation)
                            st.success("Explanation copied to clipboard!")
                        except Exception as clip_error:
                            st.error(f"Could not copy text: {clip_error}")
                    # --- END NEW ---


                    # --- Corrected: Download Buttons with PDF ---
                    with download_buttons_placeholder.container():
                        st.markdown("<hr style='border: 1px solid rgba(168, 237, 234, 0.2);'>", unsafe_allow_html=True)
                        col_btn1, col_btn2 = st.columns(2)

                        # 1. Download Annotated Image
                        buf_img = io.BytesIO()
                        Image.fromarray(annotated_img_rgb).save(buf_img, format="PNG")
                        with col_btn1:
                             st.download_button(
                                label="📥 Download Annotated Image",
                                data=buf_img.getvalue(),
                                file_name=f"fracture_detection_{uploaded_file.name}.png",
                                mime="image/png",
                                key="download_img_btn",
                                use_container_width=True
                            )

                        # 2. Download PDF Report (only if explanation exists)
                        if st.session_state.current_explanation:
                            try:
                                annotated_pil = Image.fromarray(annotated_img_rgb)
                                pdf_bytes = generate_report_pdf(annotated_pil, st.session_state.current_explanation, uploaded_file.name)
                                with col_btn2:
                                    st.download_button(
                                        label="📄 Download Full Report (PDF)",
                                        data=pdf_bytes,
                                        file_name=f"fracture_report_{uploaded_file.name}.pdf",
                                        mime="application/pdf",
                                        key="download_pdf_btn",
                                        use_container_width=True
                                    )
                            except Exception as pdf_error:
                                st.error(f"Failed to generate PDF report: {pdf_error}")
                        else:
                             # Optionally show a disabled state or placeholder if needed
                             with col_btn2:
                                 st.button("📄 Download Full Report (PDF)", disabled=True, use_container_width=True, help="Explanation must be generated first.")
                    # --- End Download Buttons ---

                else: # No detections found
                    st.info("ℹ️ No fractures detected above the current confidence threshold. Try lowering the threshold.")
                    # Corrected: Clear all relevant placeholders and state
                    results_placeholder.empty()
                    explanation_placeholder.empty()
                    download_buttons_placeholder.empty()
                    st.session_state.explanation_generated_for_this_image = False
                    st.session_state.current_explanation = ""
                    st.session_state.last_explained_language = None


            except Exception as e:
                st.error(f"❌ Error during image processing or analysis: {str(e)}")
                # import traceback # Uncomment for detailed debug trace
                # st.error(traceback.format_exc()) # Uncomment for detailed debug trace

        else: # YOLO model didn't load
            st.error("❌ YOLO Model failed to load. Please ensure 'model.pt' is in the same directory.") # Updated model name

    else: # No file uploaded
        # Corrected: Clear all placeholders and reset state
        original_image_placeholder.info("👆 Upload an X-Ray image to begin fracture detection")
        image_info_placeholder.empty()
        results_placeholder.empty()
        explanation_placeholder.empty()
        download_buttons_placeholder.empty()
        if st.session_state.last_explained_image_name is not None:
             st.session_state.last_explained_image_name = None
             st.session_state.explanation_generated_for_this_image = False
             st.session_state.current_explanation = ""
             st.session_state.last_explained_language = None
             st.session_state.last_explained_image_hash = None

    # Footer
    st.markdown("<hr style='border: 1px solid rgba(168, 237, 234, 0.2); margin-top: 2rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; font-size: 0.9rem;'>⚕️ For research and educational purposes only. Not intended for clinical diagnosis.</p>", unsafe_allow_html=True)

# --- RUN THE APP ---
# Corrected syntax for the main execution block
if __name__ == "__main__":
    main()
