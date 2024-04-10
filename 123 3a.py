import numpy as np
import rasterio
import pyopencl as cl
import time

def compute_ndvi_pyopencl(red_band_path, nir_band_path):
    # Load the red and NIR bands using rasterio
    with rasterio.open(red_band_path) as red_band:
        red = red_band.read(1).astype('float32')
    with rasterio.open(nir_band_path) as nir_band:
        nir = nir_band.read(1).astype('float32')

    # Initialize PyOpenCL context and queue
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Create PyOpenCL buffers
    mf = cl.mem_flags
    red_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=red)
    nir_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nir)
    ndvi_buf = cl.Buffer(context, mf.WRITE_ONLY, red.nbytes)

    # NDVI calculation kernel
    kernel_code = """
    __kernel void calculate_ndvi(__global const float *red, __global const float *nir, __global float *ndvi) {
        int idx = get_global_id(0);
        float denom = nir[idx] + red[idx];
        if (denom > 0)
            ndvi[idx] = (nir[idx] - red[idx]) / denom;
        else
            ndvi[idx] = 0.0f;  // Avoid division by zero
    }
    """
    program = cl.Program(context, kernel_code).build()

    # Execute the kernel
    global_size = (red.size,)
    program.calculate_ndvi(queue, global_size, None, red_buf, nir_buf, ndvi_buf)

    # Retrieve the result
    ndvi = np.empty_like(red)
    cl.enqueue_copy(queue, ndvi, ndvi_buf)
    queue.finish()

    return ndvi

if __name__ == "__main__":
    red_band_path = '/project2/macs30123/landsat8/LC08_B4.tif'
    nir_band_path = '/project2/macs30123/landsat8/LC08_B5.tif'

    start_time = time.time()
    ndvi = compute_ndvi_pyopencl(red_band_path, nir_band_path)
    end_time = time.time()

    print(f"NDVI calculation with PyOpenCL took {end_time - start_time} seconds.")