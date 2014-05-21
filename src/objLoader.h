/*
 * objLoader.h
 *
 *  Created on: May 20, 2014
 *      Author: guru
 */

#ifndef OBJLOADER_H_
#define OBJLOADER_H_

// OBJ reader
class objLoader {
public:
	float triangle_count, normal_count, vertex_count;
	float *faces;
	float *normals;
	float *vertices;
	void parseOBJ(char* file);
};

#endif /* OBJLOADER_H_ */
