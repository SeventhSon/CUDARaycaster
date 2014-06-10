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

# define PI 3.14159265358979323846f

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
	CUDA_CALLABLE_MEMBER const Vector3 operator/(const float &q) const;
	CUDA_CALLABLE_MEMBER const Vector3 operator+(const Vector3& q) const;
	CUDA_CALLABLE_MEMBER const Vector3 operator-(const Vector3& q) const;
	CUDA_CALLABLE_MEMBER const Vector3 operator-() const;
	CUDA_CALLABLE_MEMBER const Vector3 direction() const;
	CUDA_CALLABLE_MEMBER float dot(const Vector3& q) const;
	CUDA_CALLABLE_MEMBER const Vector3 cross(const Vector3& q) const;
	CUDA_CALLABLE_MEMBER float length() const;
};

class Vector2 {
public:
	float x, y;

	CUDA_CALLABLE_MEMBER Vector2(float x, float y) :
			y(y), x(x) {
	}
	CUDA_CALLABLE_MEMBER const Vector2 operator*(const float &q) const;
	CUDA_CALLABLE_MEMBER const Vector2 operator/(const float &q) const;
	CUDA_CALLABLE_MEMBER const Vector2 operator+(const Vector2& q) const;
	CUDA_CALLABLE_MEMBER const Vector2 operator-(const Vector2& q) const;
	CUDA_CALLABLE_MEMBER const Vector2 direction() const;
	CUDA_CALLABLE_MEMBER float length() const;
};

class Color3 {
public:
	float r, g, b;
	CUDA_CALLABLE_MEMBER Color3(): r(0.0f),g(0.0f),b(0.0f){

	}
	CUDA_CALLABLE_MEMBER Color3(float rr, float gg, float bb) : r(rr), g(gg), b(bb){

	}
	CUDA_CALLABLE_MEMBER const Color3 operator*(const float &q) const;
	CUDA_CALLABLE_MEMBER const Color3 operator*(const Color3 &q) const;
	CUDA_CALLABLE_MEMBER const Color3 operator/(const float &q) const;
	CUDA_CALLABLE_MEMBER const Color3 operator+(const Color3 &q) const;
};

typedef Color3 Radiance3;
typedef Color3 Power3;

class BSDF {
public:
	Color3 k_L;
	Color3 k_G;
	float s;
	CUDA_CALLABLE_MEMBER BSDF() :
			k_L(Color3(0.0f,0.0f,0.0f)), k_G(Color3(0.0f,0.0f,0.0f)), s(100.0f){

	}

	CUDA_CALLABLE_MEMBER BSDF(const Color3& k_L, const Color3& k_G, float s) :
		k_L(k_L), k_G(k_G), s(s){

	}
	CUDA_CALLABLE_MEMBER Color3 evaluateFiniteScatteringDensity(const Vector3& w_i,const Vector3& w_o, const Vector3& n) const ;
};

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
	BSDF m_bsdf;
public:
	CUDA_CALLABLE_MEMBER Triangle(){

	}
	CUDA_CALLABLE_MEMBER Triangle(const Vector3& v1,const Vector3& v2, const Vector3& v3, const Vector3& n1, const Vector3& n2, const Vector3& n3, const BSDF& bsdf){
		m_vertex[0]=v1;
		m_vertex[1]=v2;
		m_vertex[2]=v3;
		m_normal[0]=n1;
		m_normal[1]=n2;
		m_normal[2]=n3;
		m_bsdf = bsdf;
	}
	CUDA_CALLABLE_MEMBER void load(const Vector3& v1,const Vector3& v2, const Vector3& v3, const Vector3& n1, const Vector3& n2, const Vector3& n3, const BSDF& bsdf){
			m_vertex[0]=v1;
			m_vertex[1]=v2;
			m_vertex[2]=v3;
			m_normal[0]=n1;
			m_normal[1]=n2;
			m_normal[2]=n3;
			m_bsdf = bsdf;
	}
	CUDA_CALLABLE_MEMBER const Vector3& vertex(int i) const;
	CUDA_CALLABLE_MEMBER const Vector3& normal(int i) const;
	CUDA_CALLABLE_MEMBER const BSDF& bsdf() const;
};


class Light {
public:
	Vector3 position;
	Power3 power;
	CUDA_CALLABLE_MEMBER Light(const Vector3& position, const Power3& power) :
			position(position), power(power){
	}
};

class Camera {
private:
	Vector3 position;
	Vector3 direction;
	Vector3 up;
	Vector3 right;
	float aspect;
	float fieldOfView;
	float d;
public:
	CUDA_CALLABLE_MEMBER Camera(float height, float width,float d, const Vector3& position) :
			fieldOfView(1.0f),position(position),aspect(height/width),d(d){

	}
	CUDA_CALLABLE_MEMBER void setPosition(const Vector3& position);
	CUDA_CALLABLE_MEMBER void lookAt(const Vector3& point,const Vector3& up);
	CUDA_CALLABLE_MEMBER Ray computeEyeRay(
			float x, float y, int width, int height);
};

class AABoundingBox{
public:
	float minX, minY, minZ;
	float maxX, maxY, maxZ;
	CUDA_CALLABLE_MEMBER AABoundingBox(float minX,float minY,float minZ,float maxX, float maxY, float maxZ) :
		minX(minX),minY(minY),minZ(minZ),maxX(maxX),maxY(maxY),maxZ(maxZ){

	}
	CUDA_CALLABLE_MEMBER AABoundingBox(const Vector3& v1,const Vector3& v2,const Vector3& v3){
		if(v2.x>v1.x){
			minX=v1.x;
			maxX=v2.x;
		}else{
			minX=v2.x;
			maxX=v1.x;
		}
		if(v3.x<minX)
			minX=v3.x;
		else if(v3.x>maxX)
			maxX=v3.x;


		if(v2.y>v1.y){
			minY=v1.y;
			maxY=v2.y;
		}else{
			minY=v2.y;
			maxY=v1.y;
		}
		if(v3.y<minY)
			minY=v3.y;
		else if(v3.y>maxY)
			maxY=v3.y;


		if(v2.z>v1.z){
			minZ=v1.z;
			maxZ=v2.z;
		}else{
			minZ=v2.z;
			maxZ=v1.z;
		}
		if(v3.z<minZ)
			minZ=v3.z;
		else if(v3.z>maxZ)
			maxZ=v3.z;
	}
	CUDA_CALLABLE_MEMBER Vector3 getCenter() const;
	CUDA_CALLABLE_MEMBER const AABoundingBox operator+(const AABoundingBox &q) const;
};

class BVHNode{
public:
	AABoundingBox aabb;
	bool isLeaf;
	unsigned int objectId;
	unsigned int left, right;
	CUDA_CALLABLE_MEMBER BVHNode(AABoundingBox aabb): aabb(aabb){

	}
};
#endif /* UTILITY_CUH_ */
