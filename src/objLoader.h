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
	Face *faces_arr;
	float *normals_arr;
	float *vertices_arr;
	unsigned int vertexCount, faceCount, normalCount;
	bool parseOBJ(const char* file);
};

#endif /* OBJLOADER_H_ */
