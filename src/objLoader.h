/*
 * objLoader.h
 *
 *  Created on: May 20, 2014
 *      Author: guru
 */

#ifndef OBJLOADER_H_
#define OBJLOADER_H_

#include "utility.cuh"

// OBJ reader
class objLoader {
public:
	float *triangles_arr;
	float *normals_arr;
	float *vertices_arr;
	unsigned int vertexCount, faceCount, normalCount;
	bool parseOBJ(const char* file);
};

class Face{
public:
	int v1,v2,v3,n1,n2,n3;
	Face(int v1, int v2, int v3, int n1, int n2, int n3):v1(v1),v2(v2),v3(v3),n1(n1),n2(n2),n3(n3){

	}
};

#endif /* OBJLOADER_H_ */
