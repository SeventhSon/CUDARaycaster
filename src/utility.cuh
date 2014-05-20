/*
 * utility.cuh
 *
 *  Created on: May 20, 2014
 *      Author: guru
 */

#ifndef UTILITY_CUH_
#define UTILITY_CUH_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

//Raycasting Classes
class Vector3 {
public:
	float x, y, z;
	CUDA_CALLABLE_MEMBER Vector3(float xx, float yy, float zz) :
			y(yy), x(xx), z(zz) {
	}
	CUDA_CALLABLE_MEMBER const Vector3 operator*(const float q) const;
	CUDA_CALLABLE_MEMBER const Vector3 operator+(const Vector3& q) const;
	CUDA_CALLABLE_MEMBER const Vector3 operator-(const Vector3& q) const;
	CUDA_CALLABLE_MEMBER const Vector3 operator-() const;
	CUDA_CALLABLE_MEMBER const Vector3 direction() const;
	CUDA_CALLABLE_MEMBER float dot(const Vector3& q) const;
	CUDA_CALLABLE_MEMBER const Vector3 cross(const Vector3& q) const;
};

class Vector2 {
public:
	float x, y;

	CUDA_CALLABLE_MEMBER Vector2(float xx, float yy) :
			y(yy), x(xx) {
	}
	CUDA_CALLABLE_MEMBER const Vector2 operator*(const float &q) const;
	CUDA_CALLABLE_MEMBER const Vector2 operator+(const Vector2& q) const;
	CUDA_CALLABLE_MEMBER const Vector2 operator-(const Vector2& q) const;
	CUDA_CALLABLE_MEMBER const Vector2 direction() const;
};

class Color3 {
public:
	float r, g, b;
	CUDA_CALLABLE_MEMBER Color3(): r(0.0f),g(0.0f),b(0.0f){

	}
	CUDA_CALLABLE_MEMBER Color3(float rr, float gg, float bb) : r(rr), g(gg), b(bb){

	}
};

typedef Color3 Radiance3;
typedef Color3 Power3;

class Ray {
private:
	Vector3 m_origin;
	Vector3 m_direction;
public:
	CUDA_CALLABLE_MEMBER Ray(const Vector3& org, const Vector3& dir) :
			m_origin(org), m_direction(dir) {
	}
	CUDA_CALLABLE_MEMBER const Vector3& origin() const;
	CUDA_CALLABLE_MEMBER const Vector3& direction() const;

};

class Triangle {
private:
	Vector3 m_vertex[3];
	Vector3 m_normal[3];
public:
	CUDA_CALLABLE_MEMBER const Vector3& vertex(int i) const;
	CUDA_CALLABLE_MEMBER const Vector3& normal(int i) const;
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
	CUDA_CALLABLE_MEMBER Camera() :
			zNear(-0.1f), zFar(-100.0f), fieldOfViewX(PI / 2.0f) {
	}
};

class Scene {
	Triangle* triangles;
	Light* lights;
	unsigned int triangleCount;
	unsigned int lightCount;
};


#endif /* UTILITY_CUH_ */
