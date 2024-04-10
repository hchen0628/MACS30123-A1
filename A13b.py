import numpy as np
import rasterio
import time

def compute_ndvi_serial(red_band_path, nir_band_path):
    with rasterio.open(red_band_path) as red_band:
        red = red_band.read(1).astype('float64')
    with rasterio.open(nir_band_path) as nir_band:
        nir = nir_band.read(1).astype('float64')
    ndvi = (nir - red) / (nir + red)
    return ndvi

if __name__ == "__main__":
    red_band_path = '/project2/macs30123/landsat8/LC08_B4.tif'
    nir_band_path = '/project2/macs30123/landsat8/LC08_B5.tif'

    # Serial CPU computation
    start_time_cpu = time.time()
    ndvi_cpu = compute_ndvi_serial(red_band_path, nir_band_path)
    end_time_cpu = time.time()
    print(f"Serial CPU NDVI calculation took {end_time_cpu - start_time_cpu:.4f} seconds.")
