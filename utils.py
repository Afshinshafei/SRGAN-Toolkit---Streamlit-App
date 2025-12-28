"""Utility functions for SRGAN toolkit."""
import os
import numpy as np
import h5py
import netCDF4 as nc
import xarray as xr
from scipy.interpolate import RectBivariateSpline

# Constants
IMG_HEIGHT = 431
IMG_WIDTH = 339


def explore_h5_file(file_path):
    """Explore structure of an HDF5 file."""
    info = {'keys': [], 'datasets': {}}
    with h5py.File(file_path, 'r') as f:
        info['keys'] = list(f.keys())
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                info['datasets'][key] = {
                    'shape': item.shape,
                    'dtype': str(item.dtype)
                }
            elif isinstance(item, h5py.Group):
                for sub_key in item.keys():
                    sub_item = item[sub_key]
                    if isinstance(sub_item, h5py.Dataset):
                        info['datasets'][f"{key}/{sub_key}"] = {
                            'shape': sub_item.shape,
                            'dtype': str(sub_item.dtype)
                        }
    return info


def explore_nc_file(file_path):
    """Explore structure of a NetCDF file."""
    info = {'dimensions': {}, 'variables': {}}
    
    # Try multiple engines in order of robustness for ERA5 files
    engines_to_try = [
        ('h5netcdf', 'h5netcdf (pure Python, most compatible)'),
        ('netcdf4', 'netcdf4 (standard)'),
        ('scipy', 'scipy (fallback)')
    ]
    
    errors = []
    
    for engine, description in engines_to_try:
        try:
            ds = xr.open_dataset(file_path, engine=engine)
            for dim_name, dim_size in ds.dims.items():
                info['dimensions'][dim_name] = dim_size
            for var_name, var in ds.data_vars.items():
                info['variables'][var_name] = {
                    'shape': var.shape,
                    'dtype': str(var.dtype),
                    'dimensions': var.dims
                }
            # Also include coordinates
            for var_name, var in ds.coords.items():
                info['variables'][var_name] = {
                    'shape': var.shape,
                    'dtype': str(var.dtype),
                    'dimensions': var.dims
                }
            ds.close()
            return info
        except Exception as e:
            errors.append(f"{engine}: {str(e)[:80]}")
            continue
    
    # If all xarray engines fail, try direct netCDF4
    try:
        with nc.Dataset(file_path, 'r') as ds:
            for dim_name, dim in ds.dimensions.items():
                info['dimensions'][dim_name] = len(dim)
            for var_name, var in ds.variables.items():
                info['variables'][var_name] = {
                    'shape': var.shape,
                    'dtype': str(var.dtype),
                    'dimensions': var.dimensions
                }
        return info
    except Exception as e:
        errors.append(f"direct netCDF4: {str(e)[:80]}")
    
    # All methods failed
    raise Exception(f"Failed to open file with all available engines:\n" + "\n".join(f"  - {err}" for err in errors))


def get_statistics(data):
    """Calculate statistics for data array."""
    return {
        'min': float(np.nanmin(data)),
        'max': float(np.nanmax(data)),
        'mean': float(np.nanmean(data)),
        'std': float(np.nanstd(data))
    }


def find_closest_index(arr, value):
    """Find the closest index in array for a given value."""
    arr = np.asarray(arr)
    idx = (np.abs(arr - value)).argmin()
    return idx


def check_netcdf_integrity(file_path):
    """Check if a NetCDF file is valid and not corrupted."""
    try:
        # Try to read file header
        with open(file_path, 'rb') as f:
            header = f.read(4)
            # NetCDF files start with 'CDF' (NetCDF3) or '\x89HDF' (NetCDF4/HDF5)
            if header[:3] == b'CDF' or header == b'\x89HDF':
                return True, "File appears to be a valid NetCDF format"
            else:
                return False, f"File does not have a valid NetCDF header (found: {header})"
    except Exception as e:
        return False, f"Cannot read file header: {str(e)}"


def open_netcdf_robust(file_path):
    """Open a NetCDF file using the most compatible engine available."""
    engines = ['h5netcdf', 'netcdf4', 'scipy']
    
    for engine in engines:
        try:
            return xr.open_dataset(file_path, engine=engine)
        except:
            continue
    
    # All failed, raise error
    raise Exception(f"Unable to open NetCDF file with any available engine: {file_path}")


def transform_data(data):
    """Transform data by flipping upside-down and shifting to center the map correctly."""
    # Flip the data upside-down to have the North Pole at the top
    data_flipped = np.flipud(data)
    
    # First shift to center the Atlantic
    data_shifted_atlantic = np.roll(data_flipped, shift=-720//2, axis=1)
    
    # Additional shift to move the USA to the left and Asia to the right
    data_shifted = np.roll(data_shifted_atlantic, shift=-1440//4, axis=1)
    
    return data_shifted

