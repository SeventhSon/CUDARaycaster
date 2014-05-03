__global__ void Clear(TColor *dst, int imageW, int imageH) {
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < imageW && iy < imageH) {
		dst[imageW * iy + ix] = make_color(0.9, 0.9, 1.0, 0);
	}
}

extern "C" void cuda_Clear(TColor *d_dst, int imageW, int imageH) {
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	Clear<<<grid, threads>>>(d_dst, imageW, imageH);
}
