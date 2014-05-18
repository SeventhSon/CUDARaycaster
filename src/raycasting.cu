#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raycasting.h"

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
float Max(float x, float y) {
	return (x > y) ? x : y;
}

float Min(float x, float y) {
	return (x < y) ? x : y;
}

int iDivUp(int a, int b) {
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__ float lerpf(float a, float b, float c) {
	return a + (b - a) * c;
}

__device__ float vecLen(float4 a, float4 b) {
	return ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)
			+ (b.z - a.z) * (b.z - a.z));
}

__device__ TColor make_color(float r, float g, float b, float a) {
	return ((int) (a * 255.0f) << 24) | ((int) (b * 255.0f) << 16)
			| ((int) (g * 255.0f) << 8) | ((int) (r * 255.0f) << 0);
}

//Raycasting device functions
__device__ Ray computeEyeRay(float x, float y, int width, int height,
		const Camera& camera) {
	const float aspect = float(height) / width;

	// Compute the side of a square at z = -1 based on our
	// horizontal left-edge-to-right-edge field of view
	const float s = -2.0f * tan(camera.fieldOfViewX * 0.5f);
	const Vector3& start = Vector3((x / width - 0.5f) * s,
			-(y / height - 0.5f) * s * aspect, 1.0f) * camera.zNear;
	return Ray(start, start.direction());
}

__device__ bool sampleRayTriangle(const Scene& scene, int x, int y,
		const Ray& R, const Triangle& T, Radiance3& radiance, float& distance) {
	float weight[3];
	const float d = intersect(R, T, weight);
	if (d >= distance) {
		return false;
	}
	// This intersection is closer than the previous one
	// Intersection point
	const Point3& P = R.origin() + R.direction() * d;
	// Find the interpolated vertex normal at the intersection
	const Vector3& n = (T.normal(0) * weight[0] + T.normal(1) * weight[1]
			+ T.normal(2) * weight[2]).direction();
	const Vector3& w_o = -R.direction();

	//shade(scene, T, P, n, w_o, radiance);

	// Debugging intersect: set to white on any intersection
	//radiance = Radiance3(1, 1, 1);

	// Debugging barycentric
	//radiance = Radiance3(weight[0], weight[1], weight[2]) / 15;

	return true;
}

__device__ float intersect(const Ray& R, const Triangle& T, float weight[3]) {
	const Vector3& e1 = T.vertex(1) - T.vertex(0);
	const Vector3& e2 = T.vertex(2) - T.vertex(0);
	const Vector3& q = R.direction().cross(e2);

	const float a = e1.dot(q);
	const Vector3& s = R.origin() - T.vertex(0);
	const Vector3& r = s.cross(e1);

	// Barycentric vertex weights
	weight[1] = s.dot(q) / a;
	weight[2] = R.direction().dot(r) / a;
	weight[0] = 1.0f - (weight[1] + weight[2]);
	const float dist = e2.dot(r) / a;
	static const float epsilon = 1e-7f;

	static const float epsilon2 = 1e-10;

	if ((a <= epsilon) || (weight[0] < -epsilon2) || (weight[1] < -epsilon2)
			|| (weight[2] < -epsilon2) || (dist <= 0.0f)) {
		// The ray is nearly parallel to the triangle, or the
		// intersection lies outside the triangle or behind
		// the ray origin: "infinite" distance until intersection.
		return INFINITY;
	} else {
		return dist;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//Texture reference and channel descriptor for image texture
texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();

//CUDA array descriptor
cudaArray *a_Src;

////////////////////////////////////////////////////////////////////////////////
// kernels
////////////////////////////////////////////////////////////////////////////////
#include "raycasting_kernel.cuh"

extern "C" cudaError_t CUDA_Bind2TextureArray() {
	return cudaBindTextureToArray(texImage, a_Src);
}

extern "C" cudaError_t CUDA_UnbindTexture() {
	return cudaUnbindTexture(texImage);
}

extern "C" cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW,
		int imageH) {
	cudaError_t error;

	error = cudaMallocArray(&a_Src, &uchar4tex, imageW, imageH);
	error = cudaMemcpyToArray(a_Src, 0, 0, *h_Src,
			imageW * imageH * sizeof(uchar4), cudaMemcpyHostToDevice);

	return error;
}

extern "C" cudaError_t CUDA_FreeArray() {
	return cudaFreeArray(a_Src);
}

