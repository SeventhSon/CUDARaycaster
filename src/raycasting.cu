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

////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//Texture reference and channel descriptor for image texture
texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();

//CUDA array descriptor
cudaArray *a_Src;

#include "utility.cuh"
////////////////////////////////////////////////////////////////////////////////
//Raycasting Classes' functions
////////////////////////////////////////////////////////////////////////////////
CUDA_CALLABLE_MEMBER const Vector2 Vector2::operator*(const float &q) const {
	return (Vector2(this->x * q, this->y * q));
}

CUDA_CALLABLE_MEMBER const Vector2 Vector2::operator/(const float &q) const {
	return (Vector2(this->x / q, this->y / q));
}

CUDA_CALLABLE_MEMBER const Vector2 Vector2::operator+(const Vector2& q) const {
	return (Vector2(this->x + q.x, this->y + q.y));
}

CUDA_CALLABLE_MEMBER const Vector2 Vector2::operator-(const Vector2& q) const {
	return (Vector2(x - q.x, y - q.y));
}

CUDA_CALLABLE_MEMBER const Vector2 Vector2::direction() const {
	float length = sqrtf((this->x * this->x) + (this->y * this->y));
	return Vector2(this->x / length, this->y / length);
}
CUDA_CALLABLE_MEMBER float Vector2::length() const{
	return sqrtf((this->x * this->x) + (this->y * this->y));
}

CUDA_CALLABLE_MEMBER const Vector3 Vector3::operator*(const float q) const {
	return (Vector3(this->x * q, this->y * q, this->z * q));
}

CUDA_CALLABLE_MEMBER const Vector3 Vector3::operator/(const float &q) const {
	return (Vector3(this->x / q, this->y / q, this->z / q));
}

CUDA_CALLABLE_MEMBER const Vector3 Vector3::operator+(const Vector3& q) const {
	return (Vector3(this->x + q.x, this->y + q.y, this->z + q.z));
}

CUDA_CALLABLE_MEMBER const Vector3 Vector3::operator-(const Vector3& q) const {
	return (Vector3(this->x - q.x, this->y - q.y, this->z - q.z));
}
CUDA_CALLABLE_MEMBER const Vector3 Vector3::operator-() const{
	return Vector3(-(this->x), -(this->y), -(this->z));
}

CUDA_CALLABLE_MEMBER const Vector3 Vector3::direction() const {
	float length = sqrtf(
			(this->x * this->x) + (this->y * this->y) + (this->z * this->z));
	return Vector3(this->x / length, this->y / length, this->z / length);
}

CUDA_CALLABLE_MEMBER float Vector3::dot(const Vector3& q) const{
	return this->x*q.x+this->y*q.y+this->z*q.z;
}

CUDA_CALLABLE_MEMBER const Vector3 Vector3::cross(const Vector3& q) const{
	return Vector3(this->y*q.z - this->z*q.y,this->z*q.x-this->x*q.z,this->x*q.y-this->y*q.x);
}

CUDA_CALLABLE_MEMBER float Vector3::length() const{
	return sqrtf((this->x * this->x) + (this->y * this->y) + (this->z * this->z));
}

CUDA_CALLABLE_MEMBER const Vector3& Ray::origin() const {
	return m_origin;
}
CUDA_CALLABLE_MEMBER const Vector3& Ray::direction() const {
	return m_direction;
}

CUDA_CALLABLE_MEMBER const Vector3& Triangle::vertex(int i) const {
	return m_vertex[i];
}

CUDA_CALLABLE_MEMBER const Vector3& Triangle::normal(int i) const {
	return m_normal[i];
}

CUDA_CALLABLE_MEMBER const BSDF& Triangle::bsdf() const {
	return m_bsdf;
}

CUDA_CALLABLE_MEMBER const Color3 Color3::operator*(const float &q) const{
	return Color3(min(this->r*q,1.0f),min(this->g*q,1.0f),min(this->b*q,1.0f));
}

CUDA_CALLABLE_MEMBER const Color3 Color3::operator*(const Color3 &q) const{
	return Color3(this->r*q.r,this->g*q.g,this->b*q.b);
}

CUDA_CALLABLE_MEMBER const Color3 Color3::operator/(const float &q) const{
	return Color3(this->r/q,this->g/q,this->b/q);
}

CUDA_CALLABLE_MEMBER const Color3 Color3::operator+(const Color3 &q) const{
	return Color3(this->r+q.r,this->g+q.g,this->b+q.b);
}

CUDA_CALLABLE_MEMBER Color3 BSDF::evaluateFiniteScatteringDensity(const Vector3& w_i,const Vector3& w_o, const Vector3& n) const {
		const Vector3& w_h = (w_i + w_o).direction();
		return k_L/PI;
		//return (k_L + k_G * ((s + 8.0f) * powf(max(0.0f, w_h.dot(n)), s) / 8.0f)) /PI;
}

////////////////////////////////////////////////////////////////////////////////
//Raycasting device functions
////////////////////////////////////////////////////////////////////////////////
__device__ Ray computeEyeRay(float x, float y, int width, int height,
		const Camera& camera) {
	const float aspect = float(height) / width;

	// Compute the side of a square at z = -1 based on our
	// horizontal left-edge-to-right-edge field of view
	//-2.0f* tan(camera.fieldOfViewX * 0.5f)
	const float s = -2.0f;
	const Vector3& start = Vector3((x / width - 0.5f) * s,
			-(y / height - 0.5f) * s * aspect, 1.0f) * camera.zNear;
	return Ray(start, start.direction());
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
	const float epsilon = 1e-7f;

	const float epsilon2 = 1e-10;

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
__device__ void shade(const Triangle& T, const Vector3& P,const Vector3& n, const Vector3& w_o, Radiance3& L_o, Light& light) {

	const Vector3& offset = light.position - P;
	const float distanceToLight = offset.length();
	const Vector3& w_i = offset / distanceToLight;
	L_o = light.power;// / (4 * PI * distanceToLight*distanceToLight);

	// Scatter the light
	L_o = L_o*T.bsdf().evaluateFiniteScatteringDensity(w_i, w_o,n) * max(0.0, w_i.dot(n));
}

__device__ bool sampleRayTriangle(const Ray& R, const Triangle& T,
		Radiance3& radiance, float& distance, Light& light) {
	float weight[3];
	const float d = intersect(R, T, weight);
	if (d >= distance) {
		return false;
	}
	// This intersection is closer than the previous one
	// Intersection point
	const Vector3& P = R.origin() + R.direction() * d;
	// Find the interpolated vertex normal at the intersection
	const Vector3& n = (T.normal(0) * weight[0] + T.normal(1) * weight[1]
			+ T.normal(2) * weight[2]).direction();
	const Vector3& w_o = -R.direction();

	shade(T, P, n, w_o, radiance,light);

	// Debugging barycentric
	//radiance = Radiance3(weight[0], weight[1], weight[2])*0.14f;

	return true;
}

////////////////////////////////////////////////////////////////////////////////
// kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void Clear(TColor *dst, int imageW, int imageH) {
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < imageW && iy < imageH) {
		dst[imageW * iy + ix] = make_color(0.9, 0.5, 1.0, 1.0);
	}
}

__global__ void rayCast(TColor *dst, int imageW, int imageH,
		Camera camera, Light light, unsigned int triangleCount,Triangle* d_triangles) {
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;

	extern __shared__ Triangle triangles[];

	if (threadIdx.x < warpSize)
		for(int i = threadIdx.x; i  <triangleCount; i += warpSize)
			triangles[i] = d_triangles[i];
	__syncthreads();

	Radiance3 L_o;
	const Ray& R = computeEyeRay(ix + 0.5f, iy + 0.5f, imageW, imageH, camera);
	float distance = INFINITY;
	for (unsigned int t = 0; t < triangleCount; ++t) {
		const Triangle& T = triangles[t];
		if (sampleRayTriangle(R, T, L_o, distance,light)) {
			if (ix < imageW && iy < imageH) {
				dst[imageW * iy + ix] = make_color(L_o.r, L_o.g, L_o.b, 1.0);
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// CUDA code handles
////////////////////////////////////////////////////////////////////////////////
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

extern "C" void cuda_Clear(TColor *d_dst, int imageW, int imageH) {
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	Clear<<<grid, threads>>>(d_dst, imageW, imageH);
}

extern "C" void cuda_rayCasting(TColor *d_dst, int imageW, int imageH,
		Camera camera, Light light, unsigned int triangleCount, Triangle* d_triangles) {
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	rayCast<<<grid, threads, triangleCount * sizeof(Triangle)>>>(d_dst, imageW,
			imageH, camera, light, triangleCount, d_triangles);
}
