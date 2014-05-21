#include "objLoader.h"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "utility.cuh"
bool objLoader::parseOBJ(const char* path) {
	std::vector<Face> triangles;
	std::vector<Vector3> normals;
	std::vector<Vector3> vertices;
	FILE* file = fopen(path, "r");
	if (file == NULL) {
		printf("Error opening the file !\n");
		return false;
	}
	while (1) {
		char lineHeader[128];
		int res = fscanf(file, "%s", lineHeader);
		if (res == EOF)
			break;
		if (strcmp(lineHeader, "v") == 0) {
			Vector3 vertex;
			fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
			vertices.push_back(vertex);
			//printf("Vertex %f %f %f\n", vertex.x, vertex.y, vertex.z);
		} else if (strcmp(lineHeader, "vn") == 0) {
			Vector3 normal;
			fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
			normals.push_back(normal);
			//printf("Normal %f %f %f\n", normal.x, normal.y, normal.z);
		} else if (strcmp(lineHeader, "f") == 0) {
			unsigned int vertexIndex[3], normalIndex[3];
			int matches = fscanf(file, "%d//%d %d//%d %d//%d\n",
					&vertexIndex[0], &normalIndex[0], &vertexIndex[1],
					&normalIndex[1], &vertexIndex[2], &normalIndex[2]);
			if (matches != 6) {
				printf("File not supported\n");
				return false;
			}
			triangles.push_back(
					Face(vertexIndex[0], vertexIndex[1], vertexIndex[2],
							normalIndex[0], normalIndex[1], normalIndex[2]));
			//printf("Face v%d %d %d n%d %d %d\n", vertexIndex[0], vertexIndex[1],
			//		vertexIndex[2], normalIndex[0], normalIndex[1],
			//		normalIndex[2]);
		}
	}
	printf("Successfully parsed %s\nCreating arrays for transport!\n", path);
	triangles_arr = (float*) malloc(sizeof(float) * triangles.size() * 6);
	vertices_arr = (float*) malloc(sizeof(float) * vertices.size() * 3);
	normals_arr = (float*) malloc(sizeof(float) * normals.size() * 3);

	vertexCount = vertices.size();
	int j = 0;
	for (int i = 0; i < vertexCount * 3; i += 3) {
		vertices_arr[i] = vertices[j].x;
		vertices_arr[i + 1] = vertices[j].y;
		vertices_arr[i + 2] = vertices[j].z;
		j++;
	}

	normalCount = normals.size();
	j = 0;
	for (int i = 0; i < normalCount * 3; i += 3) {
		normals_arr[i] = normals[j].x;
		normals_arr[i + 1] = normals[j].y;
		normals_arr[i + 2] = normals[j].z;
		//printf("%f %f %f\n", normals_arr[i], normals_arr[i + 1],
		//		normals_arr[i + 2]);
		j++;
	}

	faceCount = triangles.size();
	j = 0;
	for (int i = 0; i < faceCount * 6; i += 6) {
		triangles_arr[i] = triangles[j].v1;
		triangles_arr[i + 1] = triangles[j].v2;
		triangles_arr[i + 2] = triangles[j].v3;
		triangles_arr[i + 3] = triangles[j].n1;
		triangles_arr[i + 4] = triangles[j].n2;
		triangles_arr[i + 5] = triangles[j].n3;
		j++;
	}
	return true;
}
