// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raycasting.h" //
#include "objLoader.h" //simple OBJ model loader
// includes, project
#include <helper_functions.h> // includes for helper utility functions
#include <helper_cuda.h>      // includes for cuda error checking and initialization
////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
//Source image on the host side
uchar4 *h_Src;
int imageW = 1360, imageH = 768;
GLuint shader;

//Host side scene definition arrays
objLoader loader;
float *d_normals, *d_vertices;
unsigned int* d_faces, *d_objectIds, *d_mortonCodes, *h_mortonCodes,
		*h_objectIds;
AABoundingBox* d_aabbs, *h_aabbs;
BVHNode* d_bvhNodes, *h_bvhNodes;
Light light(Vector3(1.0f, 1.0f, 1.0f), Power3(100.0f, 100.0f, 100.0f));
Camera cam(imageW, imageH, 1.0f, Vector3(0.0f, 0.0f, 0.0f));
bool R = true;
float angle = 0, anglestep = 0.01f;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
bool g_FPS = false;
bool g_Diag = false;
StopWatchInterface *timer = NULL;

const int frameN = 24;
int frameCounter = 0;

//No idea what the hell was this for
//#define BUFFER_DATA(i) ((char *)0 + i)

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
unsigned int frameCount = 0;

int *pArgc = NULL;
char **pArgv = NULL;

#define REFRESH_DELAY     10 //ms
//Handy function displaying GPU CUDA info
void displayCUDAInfo() {
	const int kb = 1024;
	const int mb = kb * kb;
	printf("CUDA INFO:\n=========\n\nCUDA version:   v%d\n", CUDART_VERSION);

	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("CUDA Devices: \n\n");

	for (int i = 0; i < devCount; ++i) {
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		printf("%d : %s:%d.%d\n", i, props.name, props.major, props.minor);
		printf("  Global memory:   %dmb\n", props.totalGlobalMem / mb);
		printf("  Shared memory:   %dkb\n", props.sharedMemPerBlock / kb);
		printf("  Constant memory: %dkb\n", props.totalConstMem / kb);
		printf("  Block registers: %d\n", props.regsPerBlock);

		printf("  Warp size:         %d\n", props.warpSize);
		printf("  Threads per block: %d\n", props.maxThreadsPerBlock);
		printf("  Max block dimensions: [ %d, %d, %d ]\n",
				props.maxThreadsDim[0], props.maxThreadsDim[1],
				props.maxThreadsDim[2]);
		printf("  Max grid dimensions:  [ %d, %d, %d ]\n\n=========\n\n",
				props.maxGridSize[0], props.maxGridSize[1],
				props.maxGridSize[2]);
	}
}

void computeFPS() {
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit) {
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		sprintf(fps, "CudaRaycaster: %3.1f fps", ifps);

		glutSetWindowTitle(fps);
		fpsCount = 0;

		fpsLimit = (int)MAX(ifps, 1.f);
		sdkResetTimer(&timer);
	}
}

void displayFunc(void) {
	sdkStartTimer(&timer);
	TColor *d_dst = NULL;
	size_t num_bytes;

	if (frameCounter++ == 0) {
		sdkResetTimer(&timer);
	}

	// DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_dst, gl_PBO));
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	getLastCudaError("cudaGraphicsMapResources failed");
	checkCudaErrors(
			cudaGraphicsResourceGetMappedPointer((void ** )&d_dst, &num_bytes,
					cuda_pbo_resource));
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

	checkCudaErrors(CUDA_Bind2TextureArray());

	//////////////////////////////////////////////////////////////////////////////
	//Run the kernel!

	if (R) {
		angle += anglestep;
		light.position = Vector3(cosf(angle) * 8.0f, 0.0f, sinf(angle) * 8.0f);
		cam.setPosition(Vector3(cosf(angle) * 8.0f, 0.0f, sinf(angle) * 8.0f));
		cam.lookAt(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 1.0f, 0.0f));
	}
	cuda_rayCasting(d_dst, imageW, imageH, cam, light, loader.faceCount,
			loader.vertexCount, loader.normalCount, d_faces, d_vertices,
			d_normals, d_objectIds, d_mortonCodes, d_aabbs, d_bvhNodes, h_aabbs);
	/*checkCudaErrors(
			cudaMemcpy(h_mortonCodes, d_mortonCodes,
					loader.faceCount * sizeof(unsigned int),
					cudaMemcpyDeviceToHost));
	checkCudaErrors(
			cudaMemcpy(h_objectIds, d_objectIds,
					loader.faceCount * sizeof(unsigned int),
					cudaMemcpyDeviceToHost));
	checkCudaErrors(
			cudaMemcpy(h_bvhNodes, d_bvhNodes,
					(loader.faceCount * 2 - 1) * sizeof(BVHNode),
					cudaMemcpyDeviceToHost));
	for (int i=0; i<loader.faceCount;i++){
		printf("%u %u\n", h_objectIds[i], h_mortonCodes[i]);
		printf("%f %f %f %f %f %f\n", h_aabbs[h_objectIds[i]].minX,
						h_aabbs[h_objectIds[i]].minY, h_aabbs[h_objectIds[i]].minZ,
						h_aabbs[h_objectIds[i]].maxX, h_aabbs[h_objectIds[i]].maxY,
						h_aabbs[h_objectIds[i]].maxZ);
		printf("%f %f %f\n", h_aabbs[h_objectIds[i]].getCenter().x,h_aabbs[h_objectIds[i]].getCenter().y,h_aabbs[h_objectIds[i]].getCenter().z);
	}
	for (int i = 0; i < loader.faceCount*2 -1; i++) {
		printf("ID: %d\tparent: %d\tsplit: %d\tIsLeaf: %d\tObjId: %u\tleft %d  \tright %d  \tstart %d  \tstop %d\n", i, h_bvhNodes[i].parent, h_bvhNodes[i].visited, h_bvhNodes[i].isLeaf,h_bvhNodes[i].objectId,h_bvhNodes[i].left,h_bvhNodes[i].right,h_bvhNodes[i].start,h_bvhNodes[i].stop);
		printf("AABB %f %f %f %f %f %f\n",h_bvhNodes[i].aabb.minX,
				h_bvhNodes[i].aabb.minY, h_bvhNodes[i].aabb.minZ,
				h_bvhNodes[i].aabb.maxX, h_bvhNodes[i].aabb.maxY,
				h_bvhNodes[i].aabb.maxZ);

	}*/
	getLastCudaError("Raycasting kernel execution failed.\n");
	//////////////////////////////////////////////////////////////////////////////

	checkCudaErrors(CUDA_UnbindTexture());
	// DEPRECATED: checkCudaErrors(cudaGLUnmapBufferObject(gl_PBO));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

	// Common display code path
	{
		glClear(GL_COLOR_BUFFER_BIT);

		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA,
				GL_UNSIGNED_BYTE, NULL);
		glBegin(GL_TRIANGLES);
		glTexCoord2f(0, 0);
		glVertex2f(-1, -1);
		glTexCoord2f(2, 0);
		glVertex2f(3, -1);
		glTexCoord2f(0, 2);
		glVertex2f(-1, 3);
		glEnd();
		glFinish();
	}

	if (frameCounter == frameN) {
		frameCounter = 0;

		if (g_FPS) {
			printf("FPS: %3.1f\n", frameN / (sdkGetTimerValue(&timer) * 0.001));
			g_FPS = false;
		}
	}

	glutSwapBuffers();

	sdkStopTimer(&timer);
	computeFPS();
}

void timerEvent(int value) {
	glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void shutDown(unsigned char k, int /*x*/, int /*y*/) {
	switch (k) {
	case '\033':
	case 'Q':
		printf("Shutting down...\n");

		sdkStopTimer(&timer);
		sdkDeleteTimer(&timer);

		checkCudaErrors(CUDA_FreeArray());
		free(h_Src);

		exit(EXIT_SUCCESS);
		break;
		//Light manipulation
	case 'a':
		light.position = Vector3(light.position.x - 0.4f, light.position.y,
				light.position.z);
		printf("Light (%f, %f, %f)\n", light.position.x, light.position.y,
				light.position.z);
		break;
	case 'd':
		light.position = Vector3(light.position.x + 0.4f, light.position.y,
				light.position.z);
		printf("Light (%f, %f, %f)\n", light.position.x, light.position.y,
				light.position.z);
		break;
	case 's':
		light.position = Vector3(light.position.x, light.position.y - 0.4f,
				light.position.z);
		printf("Light (%f, %f, %f)\n", light.position.x, light.position.y,
				light.position.z);
		break;
	case 'w':
		light.position = Vector3(light.position.x, light.position.y + 0.4f,
				light.position.z);
		printf("Light (%f, %f, %f)\n", light.position.x, light.position.y,
				light.position.z);
		break;
	case 'e':
		light.position = Vector3(light.position.x, light.position.y,
				light.position.z - 0.4f);
		printf("Light (%f, %f, %f)\n", light.position.x, light.position.y,
				light.position.z);
		break;
	case 'q':
		light.position = Vector3(light.position.x, light.position.y,
				light.position.z + 0.4f);
		printf("Light (%f, %f, %f)\n", light.position.x, light.position.y,
				light.position.z);
		break;
	case 'r':
		if (R) {
			R = false;
			printf("Rotation OFF\n");
		} else {
			R = true;
			printf("Rotation ON\n");
		}
		break;
	case '+':
		anglestep += 0.01;
		break;
	case '-':
		anglestep -= 0.01;
		break;
	}
}

int initGL(int *argc, char **argv) {
	printf("Initializing GLUT...\n");
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(imageW, imageH);
	glutInitWindowPosition(512 - imageW / 2, 384 - imageH / 2);
	glutCreateWindow(argv[0]);
	printf("OpenGL window created.\n");

	glewInit();
	printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));

	if (!glewIsSupported(
			"GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
		fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
		fprintf(stderr, "This sample requires:\n");
		fprintf(stderr, "  OpenGL version 1.5\n");
		fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
		fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
		fflush(stderr);
		return false;
	}

	return 0;
}

// shader for displaying floating-point texture
static const char *shader_code = "!!ARBfp1.0\n"
		"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
		"END";

GLuint compileASMShader(GLenum program_type, const char *code) {
	GLuint program_id;
	glGenProgramsARB(1, &program_id);
	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB,
			(GLsizei) strlen(code), (GLubyte *) code);

	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

	if (error_pos != -1) {
		const GLubyte *error_string;
		error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
		fprintf(stderr, "Program error at position: %d\n%s\n", (int) error_pos,
				error_string);
		return 0;
	}

	return program_id;
}

void initOpenGLBuffers() {
	printf("Creating GL texture...\n");
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imageW, imageH, 0, GL_RGBA,
			GL_UNSIGNED_BYTE, h_Src);
	printf("Texture created.\n");

	printf("Creating PBO...\n");
	glGenBuffers(1, &gl_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, imageW * imageH * 4, h_Src,
			GL_STREAM_COPY);
	//While a PBO is registered to CUDA, it can't be used
	//as the destination for OpenGL drawing calls.
	//But in our particular case OpenGL is only used
	//to display the content of the PBO, specified by CUDA kernels,
	//so we need to register/unregister it only once.
	// DEPRECATED: checkCudaErrors(cudaGLRegisterBufferObject(gl_PBO) );
	checkCudaErrors(
			cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO,
					cudaGraphicsMapFlagsWriteDiscard));
	GLenum gl_error = glGetError();

	if (gl_error != GL_NO_ERROR) {
		fprintf(stderr, "GL Error in file '%s' in line %d :\n", __FILE__,
				__LINE__);
		fprintf(stderr, "%s\n", gluErrorString(gl_error));
		exit(EXIT_FAILURE);
	}

	printf("PBO created.\n");

	// load shader program
	shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void cleanup() {
	sdkDeleteTimer(&timer);

	glDeleteProgramsARB(1, &shader);
}

int main(int argc, char **argv) {
	char *dump_file = NULL;
	int clearColorbit = 255 << 24 | 0 << 16 | 0 << 8 | 0;

	pArgc = &argc;
	pArgv = argv;

	displayCUDAInfo();

	printf("Raycaster starting...\n\n");

	initGL(&argc, argv);
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

	h_Src = (uchar4*) malloc(imageH * imageW * 4);
	memset(h_Src, clearColorbit, imageH * imageW * 4);
	checkCudaErrors(CUDA_MallocArray(&h_Src, imageW, imageH));

	initOpenGLBuffers();

	//Let's parse our object!
	if (loader.parseOBJ("data/suzzane2.obj")) {
		//Allocating arrays for our data
		checkCudaErrors(
				cudaMalloc(&d_faces,
						sizeof(unsigned int) * loader.faceCount * 6));
		checkCudaErrors(
				cudaMalloc(&d_normals, sizeof(float) * loader.normalCount * 3));
		checkCudaErrors(
				cudaMalloc(&d_vertices,
						sizeof(float) * loader.vertexCount * 3));
		checkCudaErrors(
				cudaMalloc(&d_bvhNodes,
						sizeof(BVHNode)
								* (loader.faceCount + loader.faceCount - 1)));
		checkCudaErrors(
				cudaMalloc(&d_objectIds,
						sizeof(unsigned int) * loader.faceCount));
		checkCudaErrors(
				cudaMalloc(&d_mortonCodes,
						sizeof(unsigned int) * loader.faceCount));
		checkCudaErrors(
				cudaMalloc(&d_aabbs, sizeof(AABoundingBox) * loader.faceCount));
		h_mortonCodes = (unsigned int*) malloc(
				sizeof(unsigned int) * loader.faceCount);
		h_objectIds = (unsigned int*) malloc(
				sizeof(unsigned int) * loader.faceCount);
		h_aabbs = (AABoundingBox*) malloc(
				sizeof(AABoundingBox) * loader.faceCount);
		h_bvhNodes = (BVHNode*) malloc(sizeof(BVHNode) * loader.faceCount * 2 - 1);
		//Copying data to device
		checkCudaErrors(
				cudaMemcpy(d_faces, loader.triangles_arr,
						loader.faceCount * sizeof(unsigned int) * 6,
						cudaMemcpyHostToDevice));
		checkCudaErrors(
				cudaMemcpy(d_normals, loader.normals_arr,
						loader.normalCount * sizeof(float) * 3,
						cudaMemcpyHostToDevice));
		checkCudaErrors(
				cudaMemcpy(d_vertices, loader.vertices_arr,
						loader.vertexCount * sizeof(float) * 3,
						cudaMemcpyHostToDevice));
	} else
		exit(EXIT_FAILURE);
	printf("Starting GLUT main loop...\n");

	glutDisplayFunc(displayFunc);
	glutKeyboardFunc(shutDown);

	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	glutMainLoop();

// cudaDeviceReset causes the driver to clean up all state. While
// not mandatory in normal operation, it is good practice.  It is also
// needed to ensure correct operation when the application is being
// profiled. Calling cudaDeviceReset causes all profile data to be
// flushed before the application exits
	cudaDeviceReset();
	exit(EXIT_SUCCESS);
}
