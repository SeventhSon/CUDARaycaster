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
	unsigned int *triangles_arr;
	float *normals_arr;
	float *vertices_arr;
	unsigned int vertexCount, faceCount, normalCount;
	bool parseOBJ(const char* file);
};

class Face{
public:
	unsigned int v1,v2,v3,n1,n2,n3;
	Face(unsigned int v1, unsigned int v2, unsigned int v3, unsigned int n1, unsigned int n2, unsigned int n3):v1(v1),v2(v2),v3(v3),n1(n1),n2(n2),n3(n3){

	}
};

#endif /* OBJLOADER_H_ */
