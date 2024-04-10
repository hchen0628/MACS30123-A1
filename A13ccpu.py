import numpy as np
import rasterio
import time

def compute_ndvi_serial(red, nir):
    # Serial NDVI calculation
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    return ndvi

if __name__ == "__main__":
    red_band_path = '/project2/macs30123/landsat8/LC08_B4.tif'
    nir_band_path = '/project2/macs30123/landsat8/LC08_B5.tif'

    with rasterio.open(red_band_path) as red_band, rasterio.open(nir_band_path) as nir_band:
        red = red_band.read(1).astype('float64')
        nir = nir_band.read(1).astype('float64')

    for factor in [50, 100, 150]:
        print(f"Simulating for {factor} Landsat scenes")
        red_tiled = np.tile(red, (factor, 1))
        nir_tiled = np.tile(nir, (factor, 1))

        start_time = time.time()
        ndvi_cpu = compute_ndvi_serial(red_tiled, nir_tiled)
        print(f"Serial computation for {factor} scenes took {time.time() - start_time:.4f} seconds.")
