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
	CUDA_CALLABLE_MEMBER Vector3() :
			x(0.0f),y(0.0f),z(0.0f){
	}
	CUDA_CALLABLE_MEMBER Vector3(float x, float y, float z) :
			y(y), x(x), z(z) {
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

	CUDA_CALLABLE_MEMBER Vector2(float x, float y) :
			y(y), x(x) {
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
	Color3 m_color;
public:
	CUDA_CALLABLE_MEMBER Triangle(const Vector3& v1,const Vector3& v2, const Vector3& v3, const Vector3& n1, const Vector3& n2, const Vector3& n3, const Color3& color){
		m_vertex[0]=v1;
		m_vertex[1]=v2;
		m_vertex[2]=v3;
		m_normal[0]=n1;
		m_normal[1]=n2;
		m_normal[2]=n3;
		m_color = color;
	}
	CUDA_CALLABLE_MEMBER const Vector3& vertex(int i) const;
	CUDA_CALLABLE_MEMBER const Vector3& normal(int i) const;
};


class Light {
public:
	Vector3 position;
	Color3 color;
	CUDA_CALLABLE_MEMBER Light(const Vector3& position, const Color3& color) :
			position(position), color(color){
	}
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


#endif /* UTILITY_CUH_ */
