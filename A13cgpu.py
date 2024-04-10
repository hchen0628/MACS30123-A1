import numpy as np
import rasterio
import pyopencl as cl
import time

def compute_ndvi_pyopencl(red, nir, context, queue):
    mf = cl.mem_flags
    red_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=red)
    nir_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nir)
    ndvi_buf = cl.Buffer(context, mf.WRITE_ONLY, red.nbytes)

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

    with rasterio.open(red_band_path) as red_band, rasterio.open(nir_band_path) as nir_band:
        red = red_band.read(1).astype('float64')
        nir = nir_band.read(1).astype('float64')

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)


    for factor in [50, 100, 150]:
        red_tiled = np.tile(red, (factor, 1)).astype('float32')
        nir_tiled = np.tile(nir, (factor, 1)).astype('float32')

        start_time = time.time()
        ndvi_gpu = compute_ndvi_pyopencl(red_tiled, nir_tiled, context, queue)
        print(f"Parallel GPU computation for {factor} scenes took {time.time() - start_time:.4f} seconds.")
