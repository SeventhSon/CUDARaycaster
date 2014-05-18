#ifndef RAYCASTING_H
#define RAYCASTING_H

typedef unsigned int TColor;

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

# define PI 3.14159265358979323846

#ifndef MAX
#define MAX(a,b) ((a < b) ? b : a)
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

//Raycasting Classes
class Vector2 {
public:
	float x, y;
};

class Vector3 {
public:
	float x, y, z;
};

typedef Vector2 Point2;
typedef Vector3 Point3;

class Color3 {
public:
	float r, g, b;
};

typedef Color3 Radiance3;
typedef Color3 Power3;

class Ray {
private:
	Point3 m_origin;
	Vector3 m_direction;
public:
	Ray(const Point3& org, const Vector3& dir) :
			m_origin(org), m_direction(dir) {
	}
	const Point3& origin() const {
		return m_origin;
	}
	const Vector3& direction() const {
		return m_direction;
	}
};

class Triangle {
private:
	float3 m_vertex[3];
	float3 m_normal[3];
public:
	const float3& vertex(int i) const {
		return m_vertex[i];
	}
	const float3& normal(int i) const {
		return m_normal[i];
	}
};

class Light {
public:
	float3 position;
	TColor color;
};

class Camera {
public:
	float zNear;
	float zFar;
	float fieldOfViewX;
	Camera() :
			zNear(-0.1f), zFar(-100.0f), fieldOfViewX(PI / 2.0f) {
	}
};

class Scene {
	Triangle* triangles;
	Light* lights;
	unsigned int triangleCount;
	unsigned int lightCount;
};

// CUDA wrapper functions for allocation/freeing texture arrays
extern "C" cudaError_t CUDA_Bind2TextureArray();
extern "C" cudaError_t CUDA_UnbindTexture();
extern "C" cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH);
extern "C" cudaError_t CUDA_FreeArray();

// CUDA kernel functions
extern "C" void cuda_Clear(TColor *d_dst, int imageW, int imageH);
extern "C" void cuda_rayCast(TColor *dst, int imageW, int imageH, const Scene& scene,
		const Camera& camera);

#endif
