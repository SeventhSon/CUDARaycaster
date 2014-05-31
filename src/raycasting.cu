#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raycasting.h"

#define ALIGN 4
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
// Vector2
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
CUDA_CALLABLE_MEMBER float Vector2::length() const {
	return sqrtf((this->x * this->x) + (this->y * this->y));
}

////////////////////////////////////////////////////////////////////////////////
// Vector3
////////////////////////////////////////////////////////////////////////////////

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
CUDA_CALLABLE_MEMBER const Vector3 Vector3::operator-() const {
	return Vector3(-(this->x), -(this->y), -(this->z));
}

CUDA_CALLABLE_MEMBER const Vector3 Vector3::direction() const {
	float length = sqrtf(
			(this->x * this->x) + (this->y * this->y) + (this->z * this->z));
	return Vector3(this->x / length, this->y / length, this->z / length);
}

CUDA_CALLABLE_MEMBER float Vector3::dot(const Vector3& q) const {
	return this->x * q.x + this->y * q.y + this->z * q.z;
}

CUDA_CALLABLE_MEMBER const Vector3 Vector3::cross(const Vector3& q) const {
	return Vector3(this->y * q.z - this->z * q.y, this->z * q.x - this->x * q.z,
			this->x * q.y - this->y * q.x);
}

CUDA_CALLABLE_MEMBER float Vector3::length() const {
	return sqrtf(
			(this->x * this->x) + (this->y * this->y) + (this->z * this->z));
}

////////////////////////////////////////////////////////////////////////////////
// Ray
////////////////////////////////////////////////////////////////////////////////

CUDA_CALLABLE_MEMBER const Vector3& Ray::origin() const {
	return m_origin;
}
CUDA_CALLABLE_MEMBER const Vector3& Ray::direction() const {
	return m_direction;
}

////////////////////////////////////////////////////////////////////////////////
// Triangle
////////////////////////////////////////////////////////////////////////////////
CUDA_CALLABLE_MEMBER const Vector3& Triangle::vertex(int i) const {
	return m_vertex[i];
}

CUDA_CALLABLE_MEMBER const Vector3& Triangle::normal(int i) const {
	return m_normal[i];
}

CUDA_CALLABLE_MEMBER const BSDF& Triangle::bsdf() const {
	return m_bsdf;
}
////////////////////////////////////////////////////////////////////////////////
// Color3
////////////////////////////////////////////////////////////////////////////////
CUDA_CALLABLE_MEMBER const Color3 Color3::operator*(const float &q) const {
	return Color3(this->r * q, this->g * q, this->b * q);
}

CUDA_CALLABLE_MEMBER const Color3 Color3::operator*(const Color3 &q) const {
	return Color3(this->r * q.r, this->g * q.g, this->b * q.b);
}

CUDA_CALLABLE_MEMBER const Color3 Color3::operator/(const float &q) const {
	return Color3(this->r / q, this->g / q, this->b / q);
}

CUDA_CALLABLE_MEMBER const Color3 Color3::operator+(const Color3 &q) const {
	return Color3(this->r + q.r, this->g + q.g, this->b + q.b);
}

CUDA_CALLABLE_MEMBER Color3 BSDF::evaluateFiniteScatteringDensity(
		const Vector3& w_i, const Vector3& w_o, const Vector3& n) const {
	const Vector3& w_h = (w_i + w_o).direction();
	//return k_L / PI;
	return (k_L + k_G * ((s + 8.0f) * powf(max(0.0f, w_h.dot(n)), s) / 8.0f))
			/ PI;
}

CUDA_CALLABLE_MEMBER Ray Camera::computeEyeRay(float x, float y, int width,
		int height) {
	const float aspect = float(height) / width;

	const float s = 1.0f * fieldOfView;
	Vector3 image_point = right * (x / width - 0.5f) * s
			+ up * (y / height - 0.5f) * aspect * s + position + direction;
	Vector3 ray_direction = image_point - position;
	return Ray(position, ray_direction.direction());
}

CUDA_CALLABLE_MEMBER void Camera::setPosition(const Vector3& position) {
	this->position = position;
}
CUDA_CALLABLE_MEMBER void Camera::lookAt(const Vector3& point,
		const Vector3& up) {
	direction = (point - position).direction();
	right = direction.cross(up).direction();
	this->up = right.cross(direction).direction();
}
////////////////////////////////////////////////////////////////////////////////
// BVH creation
////////////////////////////////////////////////////////////////////////////////
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ unsigned int expandBits(unsigned int v) {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int morton3D(float x, float y, float z) {
	x = min(max(x * 1024.0f, 0.0f), 1023.0f);
	y = min(max(y * 1024.0f, 0.0f), 1023.0f);
	z = min(max(z * 1024.0f, 0.0f), 1023.0f);
	unsigned int xx = expandBits((unsigned int) x);
	unsigned int yy = expandBits((unsigned int) y);
	unsigned int zz = expandBits((unsigned int) z);
	return xx * 4 + yy * 2 + zz;
}
////////////////////////////////////////////////////////////////////////////////
// Raycasting device functions
////////////////////////////////////////////////////////////////////////////////

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

	const float epsilon2 = 1e-10f;

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
__device__ void shade(const Triangle& T, const Vector3& P, const Vector3& n,
		const Vector3& w_o, Radiance3& L_o, Light& light) {

	const Vector3& offset = light.position - P;
	const float distanceToLight = offset.length();
	//Normalize the offset vector
	const Vector3& w_i = offset / distanceToLight;

	// Scatter the light
	L_o = (light.power / (4 * PI * distanceToLight * distanceToLight))
			* T.bsdf().evaluateFiniteScatteringDensity(w_i, w_o, n)
			* max(0.0, w_i.dot(n));
}

__device__ bool sampleRayTriangle(const Ray& R, const Triangle& T,
		Radiance3& radiance, float& distance, Light& light) {
	float weight[3];
	const float d = intersect(R, T, weight);
	if (d >= distance) {
		return false;
	}
	distance = d;
	// This intersection is closer than the previous one
	// Intersection point
	const Vector3& P = R.origin() + R.direction() * d;
	// Find the interpolated vertex normal at the intersection
	const Vector3& n = (T.normal(0) * weight[0] + T.normal(1) * weight[1]
			+ T.normal(2) * weight[2]).direction();
	const Vector3& w_o = -R.direction();

	shade(T, P, n, w_o, radiance, light);

	// Debugging barycentric
	// radiance = Radiance3(weight[0], weight[1], weight[2])*0.14f;

	return true;
}
__device__ Vector3 getVector(unsigned int i, float* data) {
	return Vector3(data[i], data[i + 1], data[i + 2]);
}

////////////////////////////////////////////////////////////////////////////////
// kernels
////////////////////////////////////////////////////////////////////////////////

//Shared memory handle
extern __shared__ float sharedData[];

__global__ void bvhCreate(unsigned int faceCount, unsigned int vertexCount,
		unsigned int normalCount, unsigned int* d_faces, float* d_vertices,
		float* d_normals) {

}

__global__ void rayCast(TColor *dst, int imageW, int imageH, Camera camera,
		Light light, unsigned int faceCount, unsigned int vertexCount,
		unsigned int normalCount, unsigned int* d_faces, float* d_vertices,
		float*d_normals) {
	//Global x, y image coordinates
	const int ix = blockDim.x * blockIdx.x + threadIdx.x;
	const int iy = blockDim.y * blockIdx.y + threadIdx.y;
	//Number of threads in a block
	const int threads = blockDim.x * blockDim.y;
	//The serial id of a thread in its block
	const int serialId = threadIdx.x + threadIdx.y * blockDim.x;

	//Define sharedMemory array handlers for conviniece
	float* sh_vertices = (float*) &sharedData;
	float* sh_normals = (float*) &sh_vertices[vertexCount * 3 + ALIGN
			- (vertexCount * 3) % ALIGN];
	unsigned int* sh_faces = (unsigned int*) &sh_normals[normalCount * 3 + ALIGN
			- (normalCount * 3) % ALIGN];

	//Loading triangle data to sharedMemory
	for (int i = serialId; i < vertexCount * 3; i += threads)
		sh_vertices[i] = d_vertices[i];
	for (int i = serialId; i < normalCount * 3; i += threads)
		sh_normals[i] = d_normals[i];
	for (int i = serialId; i < faceCount * 6; i += threads)
		sh_faces[i] = d_faces[i];
	//Wait for everyone to be ready
	__syncthreads();

	Triangle T;
	//Color of our pixel
	Radiance3 L_o(0.04f, 0.04f, 0.04f);
	const Ray& R = camera.computeEyeRay(ix + 0.5f, iy + 0.5f, imageW, imageH);
	//Now find the closest triangle that intersects with our ray
	float distance = INFINITY;
	for (unsigned int t = 0; t < faceCount * 6; t += 6) {
		//Construct a triangle from sharedMemory based on face index and its data.
		//Each face is composed of 6 floats (3 vertex indices, 3 normal indices).
		//Each vertex and normal is composed of 3 floats (x,y,z) and reside in their
		//respective arrays sh_vertices, sh_normals, use getVector to grab data
		T.load(getVector((sh_faces[t] - 1) * 3, sh_vertices),
				getVector((sh_faces[t + 1] - 1) * 3, sh_vertices),
				getVector((sh_faces[t + 2] - 1) * 3, sh_vertices),
				getVector((sh_faces[t + 3] - 1) * 3, sh_normals),
				getVector((sh_faces[t + 4] - 1) * 3, sh_normals),
				getVector((sh_faces[t + 5] - 1) * 3, sh_normals),
				BSDF(Color3(0.2f, 1.0f, 0.2f), Color3(0.2f, 0.2f, 0.2f),
						100.0f));
		//Try this triangle and our ray
		sampleRayTriangle(R, T, L_o, distance, light);
	}
	//Draw our pixel if we are not outside of the buffer!
	if (ix < imageW && iy < imageH) {
		dst[imageW * iy + ix] = make_color(min(L_o.r, 1.0f), min(L_o.g, 1.0f),
				min(L_o.b, 1.0f), 1.0);
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

extern "C" void cuda_rayCasting(TColor *d_dst, int imageW, int imageH,
		Camera camera, Light light, unsigned int faceCount,
		unsigned int vertexCount, unsigned int normalCount,
		unsigned int* d_faces, float* d_vertices, float*d_normals) {
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	//printf("MEm needed %d\n",
	//		((vertexCount * 3 + normalCount * 3 + faceCount * 6) * sizeof(float)));
	unsigned int aligned_v_count = vertexCount * 3;
	unsigned int aligned_n_count = normalCount * 3;
	unsigned int aligned_f_count = faceCount * 6;
	aligned_f_count += ALIGN - aligned_f_count % ALIGN;
	aligned_v_count += ALIGN - aligned_v_count % ALIGN;
	aligned_n_count += ALIGN - aligned_n_count % ALIGN;
	//printf("v %d n %d f %d\n",aligned_v_count,aligned_n_count,aligned_f_count);
	rayCast<<<grid, threads,
			(aligned_f_count * sizeof(int)
					+ (aligned_v_count + aligned_n_count) * sizeof(float))>>>(
			d_dst, imageW, imageH, camera, light, faceCount, vertexCount,
			normalCount, d_faces, d_vertices, d_normals);
}
