#ifndef RAYCASTING_H
#define RAYCASTING_H

typedef unsigned int TColor;

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

#ifndef MAX
#define MAX(a,b) ((a < b) ? b : a)
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

// CUDA wrapper functions for allocation/freeing texture arrays
extern "C" cudaError_t CUDA_Bind2TextureArray();
extern "C" cudaError_t CUDA_UnbindTexture();
extern "C" cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH);
extern "C" cudaError_t CUDA_FreeArray();

#include "utility.cuh"
// CUDA kernel functions
extern "C" void cuda_rayCasting(TColor *d_dst, int imageW, int imageH,
		Camera camera, Light light, unsigned int faceCount, unsigned int vertexCount,
		unsigned int normalCount,unsigned int* d_faces,float* d_vertices, float*d_normals,
		unsigned int* d_objectIds, AABoundingBox* d_aabbs, unsigned int* d_mortonCodes);

#endif
