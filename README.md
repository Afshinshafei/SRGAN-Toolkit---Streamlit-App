# 🌍 SRGAN Toolkit - Streamlit App

Interactive web application for downloading ERA5 data, preprocessing, and running SRGAN super-resolution inference.

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the app:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📦 Deployment to Streamlit Cloud

### Step 1: Push to GitHub

1. Create a new GitHub repository
2. Push all files to the repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository and branch
5. Set **Main file path** to `app.py`
6. Click **"Deploy"**

That's it! Your app will be live in a few minutes.

## 📋 Features

The app includes 5 main sections:

1. **📥 ERA5 Download** - Download climate data from ECMWF
2. **🔍 File Visualization** - Explore and plot NetCDF/HDF5 files
3. **🔄 NC to H5 Conversion** - Convert file formats
4. **⚙️ Preprocessing Pipeline** - Prepare data for SRGAN
5. **🚀 SRGAN Inference** - Run super-resolution model

## 🔧 Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`
- For ERA5 downloads: CDS API key from [ECMWF CDS](https://cds.climate.copernicus.eu)

## 📝 File Structure

```
srgan_streamlit/
├── app.py              # Main Streamlit application
├── utils.py            # Utility functions
├── model.py            # SRGAN model architecture
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 💡 Usage Tips

- **ERA5 Download**: You need to accept terms on the CDS website before downloading
- **File Uploads**: Files are temporarily stored during the session
- **Model Files**: Upload your trained model files for inference
- **Large Files**: Processing may take time for large datasets

## 🐛 Troubleshooting

- **Import errors**: Make sure all dependencies are installed
- **CDS API errors**: Verify your API key and that you've accepted terms
- **Memory errors**: Reduce batch size or process smaller datasets
- **File errors**: Ensure uploaded files are valid NetCDF/HDF5 format

## 📄 License

This project is provided as-is for research and educational purposes.

---

*SRGAN Toolkit v1.0 - Streamlit Edition*

