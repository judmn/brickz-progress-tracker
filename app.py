import streamlit as st
import base64
from PIL import Image
import io
import openai
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Brickz-AI",
    page_icon="🧱",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #EFF6FF, #E0E7FF);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    .progress-box {
        background: linear-gradient(to right, #ECFDF5, #EFF6FF);
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #10B981;
        margin: 20px 0;
    }
    .reference-box {
        background: #F0FDF4;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #22C55E;
        margin-bottom: 20px;
    }
    .camera-box {
        background: #EFF6FF;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #3B82F6;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'progress_result' not in st.session_state:
    st.session_state.progress_result = None
if 'captured_frame' not in st.session_state:
    st.session_state.captured_frame = None

# Reference images paths
REFERENCE_IMAGES = [
    "images/reference_1.jpg",
    "images/reference_2.jpg",
    "images/reference_3.jpg"
]

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def load_reference_images():
    """Load reference images from files"""
    images = []
    for img_path in REFERENCE_IMAGES:
        try:
            img = Image.open(img_path)
            images.append(img)
        except:
            return None
    return images

class VideoProcessor(VideoTransformerBase):
    """Video processor to capture frames from webcam"""
    def __init__(self):
        self.frame_to_capture = None
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Store the latest frame
        self.frame_to_capture = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def analyze_progress_with_gpt4(reference_images, current_img, api_key):
    """Use GPT-4 Vision to analyze Brickz building progress"""
    try:
        client = OpenAI(api_key=api_key)
        
        # Encode current image
        curr_base64 = encode_image_to_base64(current_img)
        
        # Build message content with multiple reference images
        content = [
            {
                "type": "text", 
                "text": """You are analyzing LEGO building progress. I will show you 3 reference images of the completed LEGO structure (from different angles), followed by the user's current progress image.

Please analyze:
1. What percentage of the structure is complete? (0-100%)
2. Which parts are built correctly?
3. Which parts are missing or incomplete?
4. Any differences in color, position, or brick placement?

Provide your response in this exact format:
PROGRESS: [number]%
COMPLETED_PARTS: [description]
MISSING_PARTS: [description]
NOTES: [any additional observations in concise form]

Here are the reference images of the completed structure:"""
            }
        ]
        
        # Add all reference images
        for i, ref_img in enumerate(reference_images, 1):
            ref_base64 = encode_image_to_base64(ref_img)
            content.append({
                "type": "text",
                "text": f"\n\nReference Image {i}:"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{ref_base64}",
                    "detail": "high"
                }
            })
        
        # Add current progress image
        content.append({
            "type": "text",
            "text": "\n\nUser's Current Progress Image:"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{curr_base64}",
                "detail": "high"
            }
        })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content
        
        # Parse the response
        progress_percent = 0
        completed = ""
        missing = ""
        notes = ""
        
        for line in result_text.split('\n'):
            if line.startswith('PROGRESS:'):
                progress_str = line.replace('PROGRESS:', '').strip().replace('%', '')
                try:
                    progress_percent = int(progress_str)
                except:
                    progress_percent = 0
            elif line.startswith('COMPLETED_PARTS:'):
                completed = line.replace('COMPLETED_PARTS:', '').strip()
            elif line.startswith('MISSING_PARTS:'):
                missing = line.replace('MISSING_PARTS:', '').strip()
            elif line.startswith('NOTES:'):
                notes = line.replace('NOTES:', '').strip()
        
        return {
            'progress': progress_percent,
            'completed': completed,
            'missing': missing,
            'notes': notes,
            'raw_response': result_text
        }
        
    except Exception as e:
        st.error(f"Error analyzing images: {str(e)}")
        return None

# Title and header
col1, col2 = st.columns([1, 10])
with col1:
    st.markdown("# 🧱")
with col2:
    st.markdown("# Brickz-AI:Your Intelligent Progress Tracker")

st.markdown("---")

# Check for API key
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    st.error("⚠️ OpenAI API key not found in secrets.toml")
    st.info("Please add your API key to `.streamlit/secrets.toml`:")
    st.code('OPENAI_API_KEY = "your-api-key-here"', language="toml")
    st.stop()

# Load reference images
reference_images = load_reference_images()

# Display reference images at the top
# st.markdown('<div class="reference-box">', unsafe_allow_html=True)
st.markdown("### Reference: Complete Brickz Structure")
st.markdown("*Use these images as a guide for your build*")

if reference_images:
    cols = st.columns(3)
    labels = ["Front View", "Side View", "Top View"]
    
    for i, (col, img, label) in enumerate(zip(cols, reference_images, labels)):
        with col:
            st.image(img, caption=label, use_container_width=True)
else:
    st.warning("⚠️ Reference images not found. Please add 'reference_1.jpg', 'reference_2.jpg', and 'reference_3.jpg' to your project directory.")
    st.info("💡 You can upload your reference images below:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ref1 = st.file_uploader("Upload Front View", type=['jpg', 'jpeg', 'png'], key="ref1")
    with col2:
        ref2 = st.file_uploader("Upload Side View", type=['jpg', 'jpeg', 'png'], key="ref2")
    with col3:
        ref3 = st.file_uploader("Upload Top View", type=['jpg', 'jpeg', 'png'], key="ref3")
    
    if ref1 and ref2 and ref3:
        reference_images = [Image.open(ref1), Image.open(ref2), Image.open(ref3)]
        st.success("✅ Reference images loaded! (Note: Save these as reference_1.jpg, reference_2.jpg, reference_3.jpg in your project directory for permanent use)")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Camera capture section
st.markdown("## 📸 Capture Your Current Progress")
# st.markdown('<div class="camera-box">', unsafe_allow_html=True)
st.info("📷 Position your Brickz structure in front of the camera and click the capture button.")

# Simple camera component using st.camera_input
camera_photo = st.camera_input("Take a picture of your Brickz structure")

if camera_photo:
    # Convert the uploaded bytes to PIL Image
    current_img = Image.open(camera_photo)
    st.session_state.current_image = current_img
    
    st.success("✅ Photo captured successfully!")
    
st.markdown('</div>', unsafe_allow_html=True)

# Display captured image and analysis
if st.session_state.current_image:
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Your Current Progress")
        st.image(st.session_state.current_image, use_container_width=True)
    
    with col2:
        st.markdown("### 🔍 Quick Reference")
        if reference_images:
            st.image(reference_images[0], caption="Target Structure", use_container_width=True)
        else:
            st.info("Upload reference images above to see comparison")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if reference_images and st.button("🤖 Analyze My Progress with AI", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is analyzing your Brickz structure against the reference..."):
                result = analyze_progress_with_gpt4(
                    reference_images,
                    st.session_state.current_image,
                    api_key
                )
                
                if result:
                    st.session_state.progress_result = result
    
    with col2:
        if st.button("🔄 Take New Photo", use_container_width=True):
            st.session_state.current_image = None
            st.session_state.progress_result = None
            st.rerun()
    
    # Display analysis results
    if st.session_state.progress_result:
        result = st.session_state.progress_result
        
        st.markdown("---")
        
        # Progress bar
        # st.markdown('<div class="progress-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Building Progress")
        st.progress(result['progress'] / 100)
        st.markdown(f"<h1 style='text-align: center; color: #10B981;'>{result['progress']}%</h1>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Completed Parts")
            st.success(result['completed'])
        
        with col2:
            st.markdown("### ⏳ Missing Parts")
            st.warning(result['missing'])
        
        if result['notes']:
            st.markdown("### 📝 Additional Notes")
            st.info(result['notes'])
        
        with st.expander("🔍 View Full AI Analysis"):
            st.text(result['raw_response'])

# Sidebar with instructions
with st.sidebar:
    st.markdown("## 📋 How It Works")
    st.markdown("""
    ### Simple 3-Step Process:
    
    1. **📚 View Reference**: Check the 3 reference images showing the completed Brickz structure from different angles
    
    2. **📸 Capture Photo**: Position your Brickz structure in front of your webcam and click "Take Photo"
    
    3. **🤖 Analyze**: Click "Analyze My Progress" to get AI feedback on:
       - Completion percentage
       - Correctly built parts
       - Missing components
       - Building tips
    """)
    
    st.markdown("---")
    
    if st.button("🔄 Reset Everything"):
        st.session_state.current_image = None
        st.session_state.progress_result = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🤖 Powered by")
    st.markdown("OpenAI GPT-4o-mini Vision")
    st.markdown("### 📦 Model")
    st.markdown("`gpt-4o-mini`")
