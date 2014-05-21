#include "objLoader.h"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "utility.cuh"
bool objLoader::parseOBJ(const char* path) {
	std::vector<Face> faces;
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
			printf("Vertex %f %f %f\n", vertex.x, vertex.y, vertex.z);
		} else if (strcmp(lineHeader, "vn") == 0) {
			Vector3 normal;
			fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
			normals.push_back(normal);
			printf("Normal %f %f %f\n", normal.x, normal.y, normal.z);
		} else if (strcmp(lineHeader, "f") == 0) {
			unsigned int vertexIndex[3], normalIndex[3];
			int matches = fscanf(file, "%d//%d %d//%d %d//%d\n",
					&vertexIndex[0], &normalIndex[0], &vertexIndex[1],
					&normalIndex[1], &vertexIndex[2], &normalIndex[2]);
			if (matches != 6) {
				printf("File not supported\n");
				return false;
			}
			faces.push_back(
					Face(vertexIndex[0], vertexIndex[1], vertexIndex[2],
							normalIndex[0], normalIndex[1], normalIndex[2],
							BSDF(Color3(0.4f, 0.1f, 0.8f),
									Color3(0.1f, 0.1f, 0.1f), 20.0f)));
			printf("Face v%d %d %d n%d %d %d\n", vertexIndex[0], vertexIndex[1],
					vertexIndex[2], normalIndex[0], normalIndex[1],
					normalIndex[2]);
		}
	}
	printf("Successfully parsed %s\n Creating arrays for transport!\n", path);
	faces_arr = (Face*) malloc(sizeof(Face) * faces.size());
	vertices_arr = (float*) malloc(sizeof(float) * vertices.size() * 3);
	normals_arr = (float*) malloc(sizeof(float) * normals.size() * 3);

	vertexCount = vertices.size();
	for (int i = 0; i < vertexCount; i += 3) {
		vertices_arr[i] = vertices[i].x;
		vertices_arr[i + 1] = vertices[i].y;
		vertices_arr[i + 2] = vertices[i].z;
	}

	normalCount = normals.size();
	for (int i = 0; i < normalCount; i += 3) {
		normals_arr[i] = normals[i].x;
		normals_arr[i + 1] = normals[i].y;
		normals_arr[i + 2] = normals[i].z;
	}

	faceCount = faces.size();
	for (int i = 0; i < faceCount; i += 3) {
		faces_arr[i] = faces[i];
		faces_arr[i + 1] = faces[i];
		faces_arr[i + 2] = faces[i];
	}
	return true;
}
