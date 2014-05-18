__global__ void Clear(TColor *dst, int imageW, int imageH) {
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < imageW && iy < imageH) {
		dst[imageW * iy + ix] = make_color(0.9, 0.5, 1.0, 1.0);
	}
}

__global__ void rayCast(TColor *dst, int imageW, int imageH, const Scene& scene, const Camera& camera) {
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	Radiance3 L_o;
	const Ray& R = computeEyeRay(ix + 0.5f, iy + 0.5f, imageW,
			imageH, camera);
	float distance = INFINITY;
	dst[imageW * iy + ix] = make_color(1.0,1.0,1.0,1.0);
	for (unsigned int t = 0; t < scene.triangleCount; ++t) {
		const Triangle& T = scene.triangles[t];
		if (sampleRayTriangle(scene, ix, iy, R, T, L_o, distance)) {
			if (ix < imageW && iy < imageH) {
				dst[imageW * iy + ix] = make_color(1.0, 0.5, 0.75, 1.0);
			}
		}
	}
}

extern "C" void cuda_Clear(TColor *d_dst, int imageW, int imageH) {
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	Clear<<<grid, threads>>>(d_dst, imageW, imageH);
}

extern "C" void cuda_rayCast(TColor *d_dst, int imageW, int imageH,const Scene& scene, const Camera& camera) {
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	rayCast<<<grid, threads>>>(d_dst, imageW, imageH, scene, camera);
}
