V# ResearchVisionPro 🔬

**ResearchVisionPro** is a professional, full-stack web application purpose-built for academic researchers. Inspired by Adobe Photoshop but tailored for the rigorous demands of computer vision and image processing research, it provides a comprehensive suite of tools for spatial filtering, frequency domain analysis, morphological operations, segmentation, color processing, deep learning inference, and remote sensing.

This tool is specifically designed to support the daily research workflow of faculty and students working in Deep Learning, Computer Vision, Hyperspectral Imaging, and Machine Learning.

---

## ✨ Features

ResearchVisionPro implements 42+ image processing operations, organized intuitively into research categories:

### 📐 Spatial Processing (Ch. 3)
*   Histogram Equalization & CLAHE
*   Contrast Stretching
*   Gamma & Log Transformations
*   Spatial Filtering (Smoothing, Sharpening)
*   Unsharp Masking

### 🌊 Frequency Domain (Ch. 4)
*   2D Fast Fourier Transform (FFT) Magnitude Spectrum
*   Frequency Domain Filtering (Ideal, Butterworth, Gaussian, Notch)
*   Inverse FFT Reconstruction

### 🧹 Image Restoration (Ch. 5)
*   Synthetic Noise Generation (Gaussian, Salt & Pepper, Speckle, Poisson)
*   Classical Spatial Denoising (Mean, Median, Adaptive, Contra-harmonic)
*   Wiener Filter Deblurring
*   Motion Blur Removal

### 🎨 Color Processing (Ch. 6)
*   Color Space Conversion (HSV, HSI, LAB, YCbCr)
*   Pseudocolor / False Color Mapping
*   Per-Channel Histogram Equalization
*   HSV-based Color Segmentation
*   Channel Splitting

### 🦠 Morphological Processing (Ch. 9)
*   Erosion, Dilation, Opening, Closing
*   Morphological Gradient
*   Top-Hat & Black-Hat Transforms
*   Skeletonization (Thinning)

### ✂️ Image Segmentation (Ch. 10)
*   Thresholding (Global, Otsu, Adaptive)
*   Edge Detection (Sobel, Prewitt, Roberts, Canny, Laplacian)
*   Hough Transform (Lines, Circles)
*   Region Growing
*   Watershed Segmentation
*   Connected Component Labeling

### 🧠 Deep Learning Inference
*   **Super Resolution:** SRCNN (div2k trained) for ×2 or ×4 upscaling
*   **Blind Denoising:** DnCNN (BSD68 trained) for noise removal
*   *Note: Inference runs strictly on CPU via ONNX Runtime.*

### 🛰️ Remote Sensing
*   NDVI Computation (from NIR + Red bands)
*   False-Color Compositing
*   Per-Band Statistics & Single Band Viewing
*   CLAHE Enhancement for Satellite Imagery
*   Hyperspectral Spectral Profiling

---

## 🛠️ Tech Stack

**Frontend:**
*   **Framework:** Next.js 14+ (App Router)
*   **Library:** React 18+
*   **Styling:** Tailwind CSS 3+
*   **Networking:** Axios
*   **Utils:** react-dropzone

**Backend:**
*   **Framework:** FastAPI 0.110+ (Python 3.10+)
*   **Server:** Uvicorn
*   **Image Processing:** OpenCV 4.9+, NumPy 1.26+, Pillow 10+, scikit-image 0.22+
*   **Deep Learning:** onnxruntime 1.17+ (CPU Execution Provider)

---

## 📋 Prerequisites

Before setting up the project, ensure you have the following installed on your machine:

1.  **Node.js** (v18.x or higher) and `npm`
2.  **Python** (v3.10 or higher)
3.  **Git**

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/research-vision-pro.git
cd research-vision-pro
```

### 2. Deep Learning Models (Colab Training)

ResearchVisionPro uses PyTorch models that have been pre-trained on Google Colab and exported to the `.onnx` format. This allows the backend to perform CPU-only inference without requiring a local GPU.

1.  Open the provided Jupyter notebooks in Google Colab:
    *   `colab/train_srcnn.ipynb`
    *   `colab/train_dncnn.ipynb`
2.  Run the notebooks to train the models and export them.
3.  Download the resulting `.onnx` files:
    *   `srcnn_x2.onnx`
    *   `srcnn_x4.onnx`
    *   `dncnn.onnx`
4.  Place the downloaded `.onnx` files into the `backend/models/` directory. Create the directory if it does not exist.

### 3. Backend Setup

Open a terminal and navigate to the `backend` directory:

```bash
cd backend
```

Create and activate a virtual environment (recommended):

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

*The backend will now be running at `http://localhost:8000`.*

### 4. Frontend Setup

Open a **new** terminal window and navigate to the `frontend` directory:

```bash
cd frontend
```

Install the required Node.js dependencies:

```bash
npm install
```

Start the Next.js development server:

```bash
npm run dev
```

*The frontend will now be running at `http://localhost:3000`. Open this URL in your browser to use the application.*

---

## 📖 API Documentation

The FastAPI backend automatically provides interactive API documentation. While the backend server is running, navigate to:

👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

Here you can view all endpoint schemas, parameter requirements, and test the API directly from your browser.

---

## 📂 Folder Structure

```text
research-vision-pro/
├── frontend/                   # Next.js React Frontend
│   ├── app/                    # Routing, main page, and global styles
│   ├── components/             # Reusable React components (UI, Canvas, Cards)
│   ├── lib/                    # API client definitions
│   └── public/                 # Static assets
│
├── backend/                    # FastAPI Server
│   ├── main.py                 # Application entry point & exception handlers
│   ├── models/                 # Pre-trained ONNX models (srcnn, dncnn)
│   ├── processing/             # Core image processing logic
│   │   ├── spatial.py
│   │   ├── frequency.py
│   │   ├── restoration.py
│   │   ├── color.py
│   │   ├── morphology.py
│   │   ├── segmentation.py
│   │   ├── deeplearning.py
│   │   └── remote_sensing.py
│   └── test_all_endpoints.py   # Comprehensive backend test suite
│
├── colab/                      # Jupyter notebooks for model training
│   ├── train_srcnn.ipynb
│   └── train_dncnn.ipynb
│
├── prompt_history/             # Development prompt logs
├── CONTEXT.md                  # Master project memory & constraints
├── CONVENTIONS.md              # Codebase style and rules
└── README.md                   # This file
```

---

## 📚 Gonzalez & Woods Reference Mapping

The core operations in this tool are strictly mapped to the foundational textbook: **Digital Image Processing (4th Edition) by Rafael C. Gonzalez and Richard E. Woods**.

| Chapter | Topic | Backend Module |
| :--- | :--- | :--- |
| **Ch. 3** | Intensity Transformations & Spatial Filtering | `processing/spatial.py` |
| **Ch. 4** | Filtering in the Frequency Domain | `processing/frequency.py` |
| **Ch. 5** | Image Restoration and Reconstruction | `processing/restoration.py` |
| **Ch. 6** | Color Image Processing | `processing/color.py` |
| **Ch. 9** | Morphological Image Processing | `processing/morphology.py` |
| **Ch. 10** | Image Segmentation | `processing/segmentation.py` |

---

## 🙏 Acknowledgements

*   This project was developed to support the research initiatives of **Assistant Professors and Research Scholars at IIIT Vadodara (IIITV)** working in the domains of Deep Learning, Computer Vision, and Remote Sensing.
*   The algorithmic foundation and theoretical architecture of the classical image processing operations are entirely indebted to **Rafael C. Gonzalez and Richard E. Woods** and their definitive text, *Digital Image Processing*.
# ipa-bonus
