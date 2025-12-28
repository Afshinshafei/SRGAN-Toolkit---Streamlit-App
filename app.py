"""Streamlit app for SRGAN Toolkit."""
import streamlit as st
import os
import tempfile
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import torch
import torch.utils.data as data
import warnings
warnings.filterwarnings('ignore')

# Imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils import (
    explore_h5_file, explore_nc_file, get_statistics,
    find_closest_index, check_netcdf_integrity, open_netcdf_robust,
    transform_data, IMG_HEIGHT, IMG_WIDTH
)
from model import Generator, InferenceDataset

# Page config
st.set_page_config(
    page_title="SRGAN Toolkit",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}


def save_uploaded_file(uploaded_file, file_type):
    """Save uploaded file to temp directory."""
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1]
        suffix = file_ext if file_ext else ('.nc' if file_type == 'nc' else '.h5')
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    return None


def setup_cdsapi(key):
    """Setup CDS API credentials."""
    cdsapirc_content = f"""url: https://cds.climate.copernicus.eu/api
key: {key}
"""
    cdsapirc_path = os.path.expanduser('~/.cdsapirc')
    os.makedirs(os.path.dirname(cdsapirc_path), exist_ok=True)
    with open(cdsapirc_path, 'w') as f:
        f.write(cdsapirc_content)


def denormalize_with_training_stats(output_tensor, mean, std, training_stats):
    """Properly denormalize using actual training min/max statistics."""
    output_numpy = output_tensor.cpu().numpy()
    batch_size = output_numpy.shape[0]
    denormalized_outputs = []

    for b in range(batch_size):
        output_sample = output_numpy[b]
        pr_min = training_stats['lr_pr_min']
        pr_max = training_stats['lr_pr_max']
        tas_min = training_stats['lr_tas_min']
        tas_max = training_stats['lr_tas_max']

        pr_normalized = (output_sample[0] + 1) * (pr_max - pr_min) / 2 + pr_min
        tas_normalized = (-output_sample[1] + 1) * (tas_max - tas_min) / 2 + tas_min

        pr_denormalized = pr_normalized * std[0] + mean[0]
        tas_denormalized = tas_normalized * std[1] + mean[1]

        pr_final = np.expm1(pr_denormalized)
        tas_final = tas_denormalized

        tas_mean = np.mean(tas_final)
        tas_final = 2 * tas_mean - tas_final

        denormalized_outputs.append([pr_final, tas_final])

    return np.array(denormalized_outputs)


def denormalize_approximation(output_tensor, mean, std):
    """Approximation method when training statistics are not available."""
    output_numpy = output_tensor.cpu().numpy()
    batch_size = output_numpy.shape[0]
    denormalized_outputs = []

    for b in range(batch_size):
        output_sample = output_numpy[b]
        normalized_range = 4.0
        pr_normalized = output_sample[0] * normalized_range
        tas_normalized = output_sample[1] * normalized_range

        pr_denormalized = pr_normalized * std[0] + mean[0]
        tas_denormalized = tas_normalized * std[1] + mean[1]

        pr_final = np.expm1(pr_denormalized)
        tas_final = tas_denormalized

        denormalized_outputs.append([pr_final, tas_final])

    return np.array(denormalized_outputs)


# Sidebar navigation
st.sidebar.title("🌍 SRGAN Toolkit")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Choose Section",
    [
        "🏠 Home",
        "📥 ERA5 Download",
        "🔍 File Visualization",
        "🔄 NC to H5 Conversion",
        "⚙️ Preprocessing",
        "🚀 SRGAN Inference"
    ]
)

# Home page
if page == "🏠 Home":
    st.title("🌍 Interactive ERA5 & SRGAN Super-Resolution Toolkit")
    st.markdown("---")
    
    st.markdown("""
    Welcome to the interactive toolkit for downloading ERA5 data, preprocessing, and running SRGAN super-resolution inference.
    
    ## 📋 Available Sections
    
    | Section | Purpose |
    |---------|---------|
    | **📥 ERA5 Download** | Download climate data from ECMWF |
    | **🔍 File Visualization** | Explore and plot NetCDF/HDF5 files |
    | **🔄 NC to H5 Conversion** | Convert file formats |
    | **⚙️ Preprocessing Pipeline** | Prepare data for SRGAN |
    | **🚀 SRGAN Inference** | Run super-resolution model |
    
    ## 🎯 Typical Workflow
    
    1. **📥 ERA5 Download**: Download ERA5 data
    2. **🔍 File Visualization**: Verify downloaded files (optional)
    3. **⚙️ Preprocessing**: Prepare data for SRGAN
    4. **🚀 SRGAN Inference**: Run super-resolution model
    
    **💡 Tip**: Use the sidebar to navigate between sections.
    """)

# Section 1: ERA5 Download
elif page == "📥 ERA5 Download":
    st.title("📥 ERA5 Data Download")
    st.markdown("Download ERA5 total precipitation (hourly) and 2m temperature (6-hourly) data from ECMWF Climate Data Store.")
    
    st.info("""
    **📋 How to Get Your CDS API Key:**
    1. Go to [ECMWF Climate Data Store](https://cds.climate.copernicus.eu)
    2. Register for a free account (if you don't have one) or login
    3. Once logged in, your API key will be displayed on the page. Copy it
    4. Paste it below
    
    **Important:** You must accept the terms of use for each dataset before downloading. 
    Visit the [ERA5 dataset page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) 
    and scroll to the bottom of the download form to accept the terms.
    """)
    
    api_key = st.text_input("API Key", type="password", help="Enter your CDS API key")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2022, 1, 1).date())
    with col2:
        end_date = st.date_input("End Date", value=datetime(2022, 1, 31).date())
    
    variables = st.multiselect(
        "Variables",
        ["Total Precipitation (hourly)", "2m Temperature (6-hourly)"],
        default=["Total Precipitation (hourly)", "2m Temperature (6-hourly)"]
    )
    
    if st.button("📥 Download ERA5 Data", type="primary"):
        if not api_key:
            st.error("❌ Please enter your API Key!")
        elif not variables:
            st.error("❌ Please select at least one variable!")
        else:
            try:
                setup_cdsapi(api_key)
                import cdsapi
                c = cdsapi.Client()
                
                # Generate date range
                current_date = datetime(start_date.year, start_date.month, start_date.day)
                end_datetime = datetime(end_date.year, end_date.month, end_date.day)
                date_list = []
                while current_date <= end_datetime:
                    date_list.append(current_date)
                    current_date += timedelta(days=1)
                
                years = sorted(list(set([str(d.year) for d in date_list])))
                months = sorted(list(set([f"{d.month:02d}" for d in date_list])))
                days = sorted(list(set([f"{d.day:02d}" for d in date_list])))
                
                st.info(f"Downloading data for: {start_date} to {end_date}")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                downloaded_files = []
                
                # Download Total Precipitation
                if 'Total Precipitation (hourly)' in variables:
                    status_text.text("📥 Downloading Total Precipitation (hourly)...")
                    progress_bar.progress(0.3)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
                        output_path = tmp_file.name
                    
                    c.retrieve(
                        'reanalysis-era5-single-levels',
                        {
                            'product_type': 'reanalysis',
                            'format': 'netcdf',
                            'variable': 'total_precipitation',
                            'year': years,
                            'month': months,
                            'day': days,
                            'time': [f"{h:02d}:00" for h in range(24)],
                        },
                        output_path
                    )
                    downloaded_files.append(('Precipitation', output_path))
                    st.success(f"✅ Total Precipitation downloaded!")
                
                # Download 2m Temperature
                if '2m Temperature (6-hourly)' in variables:
                    status_text.text("📥 Downloading 2m Temperature (6-hourly)...")
                    progress_bar.progress(0.7)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
                        output_path = tmp_file.name
                    
                    c.retrieve(
                        'reanalysis-era5-single-levels',
                        {
                            'product_type': 'reanalysis',
                            'format': 'netcdf',
                            'variable': '2m_temperature',
                            'year': years,
                            'month': months,
                            'day': days,
                            'time': ['00:00', '06:00', '12:00', '18:00'],
                        },
                        output_path
                    )
                    downloaded_files.append(('Temperature', output_path))
                    st.success(f"✅ 2m Temperature downloaded!")
                
                progress_bar.progress(1.0)
                status_text.text("✅ Download complete!")
                
                # Provide download buttons
                for name, file_path in downloaded_files:
                    with open(file_path, 'rb') as f:
                        st.download_button(
                            label=f"📥 Download {name} File",
                            data=f.read(),
                            file_name=f"era5_{name.lower().replace(' ', '_')}.nc",
                            mime="application/netcdf"
                        )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("""
                **Troubleshooting:**
                1. Check your API Key is correct
                2. Make sure you accepted ERA5 terms on CDS website before downloading
                3. Check your internet connection
                """)

# Section 2: File Visualization
elif page == "🔍 File Visualization":
    st.title("🔍 File Visualization")
    st.markdown("Visualize `.h5` and `.nc` files with automatic detection of variables and dimensions.")
    
    uploaded_file = st.file_uploader("Upload file", type=["nc", "h5", "nc4", "hdf5"])
    
    if uploaded_file:
        # Save uploaded file
        file_path = save_uploaded_file(uploaded_file, 'nc' if uploaded_file.name.endswith('.nc') else 'h5')
        st.session_state.uploaded_files[uploaded_file.name] = file_path
        
        try:
            # Check file integrity
            if file_path.endswith('.nc') or file_path.endswith('.nc4'):
                is_valid, check_msg = check_netcdf_integrity(file_path)
                if not is_valid:
                    st.error(f"❌ {check_msg}")
                    st.stop()
            
            file_size = os.path.getsize(file_path) / (1024*1024)
            st.info(f"📊 File size: {file_size:.2f} MB")
            
            # Explore file
            if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
                file_type = 'h5'
                info = explore_h5_file(file_path)
                st.subheader("📁 HDF5 File Information")
                st.json({"Keys": info['keys'], "Datasets": info['datasets']})
                variables = list(info['datasets'].keys())
            else:
                file_type = 'nc'
                info = explore_nc_file(file_path)
                st.subheader("📁 NetCDF File Information")
                st.json({"Dimensions": info['dimensions'], "Variables": info['variables']})
                variables = [v for v, props in info['variables'].items() if len(props['shape']) >= 2]
            
            if variables:
                var_name = st.selectbox("Select Variable", variables)
                timestep = st.slider("Timestep", 0, max(0, info['variables'].get(var_name, {}).get('shape', [0])[0] - 1) if file_type == 'nc' else max(0, info['datasets'].get(var_name, {}).get('shape', [0])[0] - 1), 0)
                flip_data = st.checkbox("Flip data upside down")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Plot"):
                        try:
                            if file_type == 'h5':
                                with h5py.File(file_path, 'r') as f:
                                    data = f[var_name][:]
                                    if len(data.shape) == 3:
                                        plot_data_2d = data[timestep, :, :]
                                    else:
                                        plot_data_2d = data[:, :]
                            else:
                                ds = open_netcdf_robust(file_path)
                                data = ds[var_name].values
                                if len(data.shape) >= 3:
                                    plot_data_2d = data[timestep, :, :]
                                else:
                                    plot_data_2d = data[:, :]
                                ds.close()
                            
                            if flip_data:
                                plot_data_2d = np.flipud(plot_data_2d)
                            
                            fig = plt.figure(figsize=(12, 8))
                            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                            
                            extent = None
                            if file_type == 'nc':
                                ds = open_netcdf_robust(file_path)
                                lons = None
                                lats = None
                                for lon_name in ['longitude', 'lon', 'x', 'rlon']:
                                    if lon_name in ds.variables or lon_name in ds.coords:
                                        lons = ds[lon_name].values
                                        break
                                for lat_name in ['latitude', 'lat', 'y', 'rlat']:
                                    if lat_name in ds.variables or lat_name in ds.coords:
                                        lats = ds[lat_name].values
                                        break
                                
                                if lons is not None and lats is not None:
                                    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
                                ds.close()
                            
                            if extent:
                                im = ax.imshow(plot_data_2d, extent=extent, origin='lower', cmap='viridis', transform=ccrs.PlateCarree())
                            else:
                                im = ax.imshow(plot_data_2d, origin='lower', cmap='viridis')
                            
                            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
                            ax.add_feature(cfeature.LAND, alpha=0.1)
                            
                            plt.colorbar(im, ax=ax, label=var_name, shrink=0.8)
                            plt.title(f'{var_name} - Timestep {timestep}')
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"❌ Error plotting: {str(e)}")
                
                with col2:
                    if st.button("📈 Show Statistics"):
                        try:
                            if file_type == 'h5':
                                with h5py.File(file_path, 'r') as f:
                                    data = f[var_name][:]
                                    stats = get_statistics(data)
                            else:
                                ds = open_netcdf_robust(file_path)
                                data = ds[var_name].values
                                stats = get_statistics(data)
                                ds.close()
                            
                            st.subheader(f"📊 Statistics for {var_name}")
                            st.json({
                                "Shape": list(data.shape),
                                "Min": f"{stats['min']:.6f}",
                                "Max": f"{stats['max']:.6f}",
                                "Mean": f"{stats['mean']:.6f}",
                                "Std": f"{stats['std']:.6f}"
                            })
                        except Exception as e:
                            st.error(f"❌ Error calculating statistics: {str(e)}")
            else:
                st.warning("No plottable variables found in this file.")
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

# Section 3: NC to H5 Conversion
elif page == "🔄 NC to H5 Conversion":
    st.title("🔄 NC to H5 Conversion")
    st.markdown("Convert NetCDF files to HDF5 format.")
    
    uploaded_nc = st.file_uploader("Upload NetCDF file", type=["nc", "nc4"])
    
    if uploaded_nc:
        file_path = save_uploaded_file(uploaded_nc, 'nc')
        
        try:
            info = explore_nc_file(file_path)
            variables = list(info['variables'].keys())
            
            st.subheader("📋 Select Variables to Convert")
            selected_vars = st.multiselect("Variables", variables, default=variables)
            
            output_name = st.text_input("Output filename", value=uploaded_nc.name.replace('.nc', '.h5').replace('.nc4', '.h5'))
            
            if st.button("🔄 Convert to H5", type="primary"):
                if not selected_vars:
                    st.error("❌ Please select at least one variable!")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5', delete=False) as tmp_file:
                        out_path = tmp_file.name
                    
                    try:
                        ds = open_netcdf_robust(file_path)
                        with h5py.File(out_path, 'w') as hf:
                            for i, var_name in enumerate(selected_vars):
                                status_text.text(f"Converting {var_name}...")
                                progress_bar.progress((i + 1) / len(selected_vars))
                                if var_name in ds.data_vars or var_name in ds.coords:
                                    data = ds[var_name].values
                                    hf.create_dataset(var_name, data=data, chunks=True)
                        ds.close()
                        
                        st.success("✅ Conversion complete!")
                        
                        with open(out_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download H5 File",
                                data=f.read(),
                                file_name=output_name,
                                mime="application/x-hdf5"
                            )
                    except Exception as e:
                        st.error(f"❌ Error during conversion: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Section 4: Preprocessing
elif page == "⚙️ Preprocessing":
    st.title("⚙️ Preprocessing Pipeline")
    st.markdown("Prepare ERA5 data for SRGAN inference.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        precip_file = st.file_uploader("Precipitation File", type=["nc", "h5", "nc4", "hdf5"])
        precip_var = st.text_input("Precipitation Variable Name", value="tp")
    
    with col2:
        temp_file = st.file_uploader("Temperature File", type=["nc", "h5", "nc4", "hdf5"])
        temp_var = st.text_input("Temperature Variable Name", value="t2m")
    
    st.subheader("⚙️ Processing Options")
    do_accumulate = st.checkbox("6-hour precipitation accumulation", value=True)
    do_extract_italy = st.checkbox("Extract Italy region (34.8-48.59N, 3.91-19.93E)", value=True)
    do_convert_units = st.checkbox("Convert precipitation m to mm", value=True)
    
    output_name = st.text_input("Output filename", value="preprocessed_era5.h5")
    
    if st.button("▶️ Run Preprocessing", type="primary"):
        if not precip_file or not temp_file:
            st.error("❌ Please upload both precipitation and temperature files!")
        else:
            precip_path = save_uploaded_file(precip_file, 'nc' if precip_file.name.endswith('.nc') else 'h5')
            temp_path = save_uploaded_file(temp_file, 'nc' if temp_file.name.endswith('.nc') else 'h5')
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Load data
                status_text.text("📥 Step 1: Loading and merging variables...")
                progress_bar.progress(0.1)
                
                if precip_path.endswith('.nc') or precip_path.endswith('.nc4'):
                    ds = open_netcdf_robust(precip_path)
                    pr_data_all = ds[precip_var].values
                    ds.close()
                else:
                    with h5py.File(precip_path, 'r') as f:
                        pr_data_all = f[precip_var][:]
                
                if temp_path.endswith('.nc') or temp_path.endswith('.nc4'):
                    ds = open_netcdf_robust(temp_path)
                    tas_data_all = ds[temp_var].values
                    ds.close()
                else:
                    with h5py.File(temp_path, 'r') as f:
                        tas_data_all = f[temp_var][:]
                
                status_text.text(f"   Loaded precipitation: shape={pr_data_all.shape}")
                status_text.text(f"   Loaded temperature: shape={tas_data_all.shape}")
                progress_bar.progress(0.2)
                
                # Step 2: Transform
                status_text.text("🔄 Step 2: Transforming data...")
                progress_bar.progress(0.3)
                
                for t in range(pr_data_all.shape[0]):
                    pr_data_all[t, :, :] = transform_data(pr_data_all[t, :, :])
                
                for t in range(tas_data_all.shape[0]):
                    tas_data_all[t, :, :] = transform_data(tas_data_all[t, :, :])
                
                progress_bar.progress(0.5)
                
                # Step 3: Extract Italy
                if do_extract_italy:
                    status_text.text("🗺️ Step 3: Extracting Italy region...")
                    north, south, west, east = 48.59, 34.8, 3.91, 19.93
                    lons = np.linspace(-180, 180, tas_data_all.shape[2])
                    lats = np.linspace(-90, 90, tas_data_all.shape[1])
                    
                    north_idx = find_closest_index(lats, north)
                    south_idx = find_closest_index(lats, south)
                    west_idx = find_closest_index(lons, west)
                    east_idx = find_closest_index(lons, east)
                    
                    tas_subregion_all = np.zeros((tas_data_all.shape[0], north_idx - south_idx + 1, east_idx - west_idx + 1))
                    pr_subregion_all = np.zeros((pr_data_all.shape[0], north_idx - south_idx + 1, east_idx - west_idx + 1))
                    
                    for t in range(tas_data_all.shape[0]):
                        tas_subregion_all[t, :, :] = tas_data_all[t, south_idx:north_idx+1, west_idx:east_idx+1]
                        pr_subregion_all[t, :, :] = pr_data_all[t, south_idx:north_idx+1, west_idx:east_idx+1]
                    
                    pr_data_all = pr_subregion_all
                    tas_data_all = tas_subregion_all
                
                progress_bar.progress(0.6)
                
                # Step 4: Convert units
                if do_convert_units:
                    status_text.text("📐 Step 4: Converting units...")
                    pr_data_all = pr_data_all * 1000
                
                progress_bar.progress(0.7)
                
                # Step 5: Accumulate
                if do_accumulate:
                    status_text.text("⏱️ Step 5: Computing 6-hour accumulation...")
                    n_timesteps = pr_data_all.shape[0]
                    n_6h_periods = n_timesteps // 6
                    pr_accumulated = np.zeros((n_6h_periods, pr_data_all.shape[1], pr_data_all.shape[2]))
                    
                    for i in range(n_6h_periods):
                        pr_accumulated[i] = np.sum(pr_data_all[i*6:(i+1)*6], axis=0)
                    
                    pr_data_all = pr_accumulated
                    
                    if tas_data_all.shape[0] > pr_data_all.shape[0]:
                        tas_data_all = tas_data_all[:pr_data_all.shape[0]]
                    elif tas_data_all.shape[0] < pr_data_all.shape[0]:
                        pr_data_all = pr_data_all[:tas_data_all.shape[0]]
                
                progress_bar.progress(0.9)
                
                # Step 6: Save
                status_text.text("💾 Step 6: Saving preprocessed data...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                    out_path = tmp_file.name
                
                with h5py.File(out_path, 'w') as f:
                    f.create_dataset('pr', data=pr_data_all.astype('float32'), chunks=True)
                    f.create_dataset('tas', data=tas_data_all.astype('float32'), chunks=True)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Preprocessing complete!")
                
                st.success(f"✅ Preprocessing complete! Final shapes: pr={pr_data_all.shape}, tas={tas_data_all.shape}")
                
                with open(out_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Preprocessed File",
                        data=f.read(),
                        file_name=output_name,
                        mime="application/x-hdf5"
                    )
                
            except Exception as e:
                st.error(f"❌ Error during preprocessing: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Section 5: SRGAN Inference
elif page == "🚀 SRGAN Inference":
    st.title("🚀 SRGAN Inference")
    st.markdown("Run super-resolution inference on preprocessed ERA5 data using the trained SRGAN model.")
    
    st.subheader("📂 Model Files")
    dem_file = st.file_uploader("DEM File (dem.nc)", type=["nc", "nc4"])
    stats_file = st.file_uploader("Stats File (train_stats_srgan_logtransform.pkl)", type=["pkl"])
    model_file = st.file_uploader("Model File (GAN_Generator_logtransform_bestmodel.pt)", type=["pt"])
    training_stats_file = st.file_uploader("Training Stats (training_minmax_statistics.pkl) - Optional", type=["pkl"])
    
    st.subheader("📊 Input Data")
    input_h5 = st.file_uploader("Input H5 File (preprocessed data)", type=["h5", "hdf5"])
    
    st.subheader("⚙️ Settings")
    batch_size = st.slider("Batch Size", 1, 32, 1)
    output_name = st.text_input("Output filename", value="srgan_output.h5")
    num_plots = st.slider("Number of timesteps to visualize", 1, 10, 2)
    
    if st.button("🚀 Run Inference", type="primary"):
        if not all([dem_file, stats_file, model_file, input_h5]):
            st.error("❌ Please upload all required files!")
        else:
            # Save files
            dem_path = save_uploaded_file(dem_file, 'nc')
            stats_path = save_uploaded_file(stats_file, 'pkl')
            model_path = save_uploaded_file(model_file, 'pt')
            input_path = save_uploaded_file(input_h5, 'h5')
            training_stats_path = save_uploaded_file(training_stats_file, 'pkl') if training_stats_file else None
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🚀 Running SRGAN Inference...")
                progress_bar.progress(0.1)
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                status_text.text(f"Using device: {device}")
                progress_bar.progress(0.2)
                
                # Load stats
                with open(stats_path, 'rb') as f:
                    stats = pickle.load(f)
                lr_out_mean = np.array([stats['pr_mean'], stats['tas_mean']])
                lr_out_std = np.array([stats['pr_std'], stats['tas_std']])
                
                # Load training stats
                use_training_stats = False
                training_stats = None
                if training_stats_path and os.path.exists(training_stats_path):
                    with open(training_stats_path, 'rb') as f:
                        training_stats = pickle.load(f)
                    use_training_stats = True
                
                progress_bar.progress(0.3)
                
                # Load dataset
                status_text.text("📂 Loading dataset...")
                inference_dataset = InferenceDataset(input_path, dem_path, stats_path)
                inference_loader = data.DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                progress_bar.progress(0.4)
                
                # Load model
                status_text.text("🧠 Loading model...")
                model = Generator(num_channels=3, num_res_blocks=8, scale_factor=8)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model = model.to(device)
                model.eval()
                progress_bar.progress(0.5)
                
                # Inference
                all_output_pr = []
                all_output_tas = []
                num_time_steps = len(inference_dataset)
                
                status_text.text(f"⚙️ Processing {num_time_steps} timesteps...")
                
                for i, input_tensor in enumerate(inference_loader):
                    progress_bar.progress(0.5 + 0.4 * (i / num_time_steps))
                    
                    input_tensor = input_tensor.to(device)
                    
                    with torch.no_grad():
                        output_tensor = model(input_tensor)
                    
                    if use_training_stats:
                        denormalized_outputs = denormalize_with_training_stats(output_tensor, lr_out_mean, lr_out_std, training_stats)
                    else:
                        denormalized_outputs = denormalize_approximation(output_tensor, lr_out_mean, lr_out_std)
                    
                    all_output_pr.append(denormalized_outputs[0, 0])
                    all_output_tas.append(denormalized_outputs[0, 1])
                
                all_output_pr = np.array(all_output_pr)
                all_output_tas = np.array(all_output_tas)
                
                progress_bar.progress(0.9)
                
                # Save output
                status_text.text("💾 Saving results...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                    out_path = tmp_file.name
                
                with h5py.File(out_path, 'w') as h5f:
                    h5f.create_dataset('pr', data=all_output_pr, compression="gzip", compression_opts=9)
                    h5f.create_dataset('tas', data=all_output_tas, compression="gzip", compression_opts=9)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Inference complete!")
                
                st.success(f"✅ Inference complete! Output shapes: pr={all_output_pr.shape}, tas={all_output_tas.shape}")
                
                # Show statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Precipitation Min", f"{all_output_pr.min():.3f} mm")
                    st.metric("Precipitation Max", f"{all_output_pr.max():.3f} mm")
                    st.metric("Precipitation Mean", f"{all_output_pr.mean():.3f} mm")
                with col2:
                    st.metric("Temperature Min", f"{all_output_tas.min():.2f} K")
                    st.metric("Temperature Max", f"{all_output_tas.max():.2f} K")
                    st.metric("Temperature Mean", f"{all_output_tas.mean():.2f} K")
                
                # Visualizations
                st.subheader("📊 Visualizations")
                num_to_plot = min(num_plots, len(all_output_pr))
                
                for t in range(num_to_plot):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(all_output_pr[t], cmap='jet', origin='upper')
                        plt.colorbar(im, ax=ax, label='Precipitation (mm)')
                        ax.set_title(f'Precipitation - Timestep {t}')
                        st.pyplot(fig)
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(all_output_tas[t], cmap='coolwarm', origin='upper')
                        plt.colorbar(im, ax=ax, label='Temperature (K)')
                        ax.set_title(f'Temperature - Timestep {t}')
                        st.pyplot(fig)
                
                # Download button
                with open(out_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Output",
                        data=f.read(),
                        file_name=output_name,
                        mime="application/x-hdf5"
                    )
                
                inference_dataset.close()
                
            except Exception as e:
                st.error(f"❌ Error during inference: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                try:
                    inference_dataset.close()
                except:
                    pass

