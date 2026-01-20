import numpy as np

def time_series_to_gaf(time_series, image_size=64, method='summation'):
    """
    Convert a time series to a Gramian Angular Field (GAF) image.
    
    Args:
        time_series (np.ndarray): 1D array of time series data.
        image_size (int): The size of the output image (image_size x image_size).
                          If len(time_series) > image_size, Piecewise Aggregate Approximation (PAA) 
                          will be used to downsample.
        method (str): 'summation' for GASF or 'difference' for GADF.
        
    Returns:
        np.ndarray: GAF image of shape (image_size, image_size).
    """
    # 1. Normalization to [-1, 1]
    min_val = np.min(time_series)
    max_val = np.max(time_series)
    
    # Handle constant signal case
    if max_val - min_val == 0:
        scaled_series = np.zeros_like(time_series)
    else:
        scaled_series = ((time_series - max_val) + (time_series - min_val)) / (max_val - min_val)
    
    # 2. Piecewise Aggregate Approximation (PAA) if needed
    n = len(scaled_series)
    if n > image_size:
        # Split into segments and take the mean
        segments = np.array_split(scaled_series, image_size)
        paa_series = np.array([np.mean(seg) for seg in segments])
    elif n < image_size:
        # Interpolate if smaller (though usually not the case for this project)
        paa_series = np.interp(np.linspace(0, n, image_size), np.arange(n), scaled_series)
    else:
        paa_series = scaled_series
        
    # 3. Convert to polar coordinates
    # Clip to ensure values are within [-1, 1] for arccos
    paa_series = np.clip(paa_series, -1, 1)
    phi = np.arccos(paa_series)
    
    # 4. Compute GAF
    # Create meshgrid
    phi_i, phi_j = np.meshgrid(phi, phi)
    
    if method == 'summation':
        # GASF: cos(phi_i + phi_j)
        gaf = np.cos(phi_i + phi_j)
    elif method == 'difference':
        # GADF: sin(phi_i - phi_j)
        gaf = np.sin(phi_i - phi_j)
    else:
        raise ValueError("Method must be 'summation' or 'difference'")
        
    return gaf
