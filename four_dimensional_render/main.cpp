

#include <helper_gl.h>
#include <Gl/wglew.h>
#include <GL/freeglut.h>
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "helper_functions.h"
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <vector>
#include "kernel.h"
#include <algorithm>
#include <iterator>


struct Point5D {
	float cords[5];
	Point4D to4D() {
		return Point4D{ cords[0] / cords[4],
					    cords[1] / cords[4],
					    cords[2] / cords[4],
					    cords[3] / cords[4], };
	}
};

Point5D operator*(Point5D a, float r)
{
	return Point5D{ a.cords[0] * r, a.cords[1] * r,a.cords[2] * r,a.cords[3] * r ,a.cords[4] * r };
}

Point5D operator*(Point5D a, Point5D b)
{
	return Point5D{ a.cords[0] * b.cords[0],
				    a.cords[1] * b.cords[1],
				    a.cords[2] * b.cords[2],
				    a.cords[3] * b.cords[3],
				    a.cords[4] * b.cords[4]};
}

struct Matrix5x5
{
	Point5D col[5];
};

Point5D operator* (Matrix5x5 a, Point5D b)
{
	Point5D res{ 0.0, 0.0, 0.0, 0.0, 0.0 };
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			res.cords[i] += a.col[i].cords[j] * b.cords[j];
		}
	}
	return res;
}

Matrix5x5 operator* (Matrix5x5 a, Matrix5x5 b)
{
	Matrix5x5 res{
		Point5D{0.0, 0.0, 0.0, 0.0, 0.0},
		Point5D{0.0, 0.0, 0.0, 0.0, 0.0},
		Point5D{0.0, 0.0, 0.0, 0.0, 0.0},
		Point5D{0.0, 0.0, 0.0, 0.0, 0.0},
		Point5D{0.0, 0.0, 0.0, 0.0, 0.0} };
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			for (int k = 0; k < 5; k++)
			{
				res.col[i].cords[j] += a.col[i].cords[k] * b.col[k].cords[j];
			}
		}
	}
	return res;
}

struct TetraLabel {
	int pos[4];
	char visibility;
};

struct HyperShape {
	std::vector<Point5D> points;
	std::vector<TetraLabel> tetrahedron_connections;

	HyperShape(std::vector<Point5D> points, std::vector<TetraLabel> tetrahedron_connections)
	{
		this->points = points;
		this->tetrahedron_connections = tetrahedron_connections;
	}

	std::vector<Tetrahedron> get_tetrahedrons() {
		std::vector<Tetrahedron> res{};
		for (auto conn : tetrahedron_connections)
		{
			res.push_back(Tetrahedron{ {points[conn.pos[0]].to4D(), points[conn.pos[1]].to4D(),
									  points[conn.pos[2]].to4D(), points[conn.pos[3]].to4D() },
									 conn.visibility });
		}
		return res;
	}
};

Matrix5x5 identity_matrix()
{
	return Matrix5x5{
		Point5D{1.0, 0.0, 0.0, 0.0, 0.0},
		Point5D{0.0, 1.0, 0.0, 0.0, 0.0},
		Point5D{0.0, 0.0, 1.0, 0.0, 0.0},
		Point5D{0.0, 0.0, 0.0, 1.0, 0.0},
		Point5D{0.0, 0.0, 0.0, 0.0, 1.0} };
}

Matrix5x5 rotation_matrix(float angle, int axis1, int axis2)
{
	Matrix5x5 res = identity_matrix();
	res.col[axis1].cords[axis1] = std::cos(angle);
	res.col[axis1].cords[axis2] = std::sin(angle);
	res.col[axis2].cords[axis1] = -std::sin(angle);
	res.col[axis2].cords[axis2] = std::cos(angle);
	return res;
}

Matrix5x5 translation(Point4D shift)
{
	Matrix5x5 res = identity_matrix();
	for (int i = 0; i < 4; i++)
	{
		res.col[i].cords[4] = shift.cord[i];
	}
	return res;
}

Point4D operator* (float r, Point4D a)
{
	return Point4D{ a.cord[0] * r, a.cord[1] * r, a.cord[2] * r, a.cord[3] * r };
}

Point4D operator* (Matrix4x4 a, Point4D b)
{
	Point4D res{ 0.0, 0.0, 0.0, 0.0 };
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			res.cord[i] += a.col[i].cord[j] * b.cord[j];
		}
	}
	return res;
}

Tetrahedron operator*(Matrix4x4 a, Tetrahedron b)
{
	return Tetrahedron{ a * b.p[0], a*b.p[1], a*b.p[2] , a*b.p[3] };
}

FILE* stream;
char g_ExecPath[300];

GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource *cuda_pbo_resource;

uchar4 *h_Src = 0;
uchar4 *d_dst = NULL;
int imageW = 400;
int imageH = 400;

StopWatchInterface *hTimer = NULL;

#define REFRESH_DELAY     10

typedef BOOL(WINAPI *PFNWGLSWAPINTERVALFARPROC)(int);

#define BUFFER_DATA(i) ((char *)0 + i)

void setVsync(int interval)
{
	if (WGL_EXT_swap_control)
	{
		wglSwapIntervalEXT = (PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress("wglSwapIntervalEXT");
		wglSwapIntervalEXT(interval);
	}
}

static const char *shader_code =
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

GLuint compileASMShader(GLenum program_type, const char* code)
{
	printf("OK");
	GLuint program_id;
	glGenProgramsARB(1, &program_id);
	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(code), (GLubyte *)code);

	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
	if (error_pos != -1)
	{
		const GLubyte *error_string;
		error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
		fprintf(stderr, "Program error at position: %d \n%s\n", (int)error_pos, error_string);
		return 0;
	}

	return program_id;
}

int fpsCount = 0;
int fpsLimit = 15;
unsigned int frameCount = 0;

void calculateFPS()
{
	fpsCount++;
	frameCount++;
	if (fpsCount == fpsLimit)
	{
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&hTimer) / 1000.f);
		sprintf(fps, "%3.1f fps\n", ifps);
		printf(fps, "%3.1f fps\n", ifps);
		glutSetWindowTitle(fps);
		fpsCount = 0;

		fpsLimit = MAX(1.f, (float)ifps);
		sdkResetTimer(&hTimer);
	}
}

void initOpenGLBuffers(int w, int h)
{
	if (h_Src)
	{
		free(h_Src);
		h_Src = 0;
	}

	if (gl_Tex)
	{
		glDeleteTextures(1, &gl_Tex);
		gl_Tex = 0;
	}

	if (gl_PBO)
	{
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &gl_PBO);
		gl_PBO = 0;
	}
	h_Src = (uchar4 *)malloc(w * h * 4);
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
	
	glGenBuffers(1, &gl_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO,
		cudaGraphicsMapFlagsWriteDiscard));
	gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}


void renderImage();

float frame_time = 1.0;

void displayFunc(void)
{
	sdkStartTimer(&hTimer);
	renderImage();
	float start_t = sdkGetTimerValue(&hTimer);

	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));

	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
	glEnable(GL_FRAGMENT_PROGRAM_ARB);
	glDisable(GL_DEPTH_TEST);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(0.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(0.0f, 1.0f);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_FRAGMENT_PROGRAM_ARB);
	frame_time = (sdkGetTimerValue(&hTimer) - start_t);
	sdkStopTimer(&hTimer);
	glutSwapBuffers();

	calculateFPS();
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

void reshapeFunc(int w, int h)
{
	glViewport(0, 0, w, h);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

	if (w != 0 && h != 0)  // Do not call when window is minimized that is when width && height == 0
		initOpenGLBuffers(w, h);

	imageW = w;
	imageH = h;
	//pass = 0;

	glutPostRedisplay();
}

int pixelX = 100;
int pixelY = 100;
float angleXT = 0.0;
float angleXY = 0.0;
float angleZT = 0.0;
float speed = 10.0;
void keyboardFunc(unsigned char k, int, int)
{
	switch (k)
	{
	case 'i':
		pixelY -= 1;
		break;
	case 'k':
		pixelY += 1;
		break;
	case 'j':
		pixelX -= 1;
		break;
	case 'l':
		pixelX += 1;
		break;
	case 'q':
		angleXY += speed * frame_time;
		break;
	case 'a':
		angleXY -= speed * frame_time;
		break;
	case 'w':
		angleXT += speed * frame_time;
		break;
	case 's':
		angleXT -= speed * frame_time;
		break;
	case 'e':
		angleZT += speed * frame_time;
		break;
	case 'd':
		angleZT -= speed * frame_time;
		break;
	}
}

void initGL(int *argc, char **argv)
{
	printf("Initializing GLUT....\n");
	glutInit(argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(imageW, imageH);
	glutInitWindowPosition(0, 0);
	glutCreateWindow(argv[0]);
	glewInit();

	glutDisplayFunc(displayFunc);
	glutKeyboardFunc(keyboardFunc);
	//glutMouseFunc()
	//glutMotionFunc()
	//glutReshapeFunc()
	glutReshapeFunc(reshapeFunc);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	//initMenus();
	
	printf("OpenGL window created.\n");
}

void cleanup()
{
	if (h_Src)
	{
		free(h_Src);
		h_Src = 0;
	}
	sdkStopTimer(&hTimer);
	sdkDeleteTimer(&hTimer);
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glDeleteBuffers(1, &gl_PBO);
	glDeleteTextures(1, &gl_Tex);
	glDeleteProgramsARB(1, &gl_Shader);
}


//cudaError_t CalculateImage(SceneData scene, Triangle* triangles, unsigned int size, int width, int height, uchar4* dst);
float angle = 0.0;

void renderImage()
{
	sdkResetTimer(&hTimer);
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, &num_bytes, cuda_pbo_resource));
	float range = 1.0f;
	//printf("%f", distX);
	SceneData scene{ Point4D{-2.0f, -0.3f, -0.3f, 0.0f},
	Point4D{0.125f, 0.125f, 0.125f, -10.0f},
	Point4D{2.0f, -range / imageW / 2, -range / imageH / 2, 0.0f},
	Point4D{0.0f, range / imageW, 0.0f, 0.0f},
	Point4D{0.0f, 0.0f, range / imageH, 0.0f} };
	float res = 0.0;

	Point4D center = Point4D{ 0.25, 0.25, 0.25, 0.25 };
	Matrix5x5 trans = translation(center) * rotation_matrix(angleZT, 2, 3) * rotation_matrix(angleXY, 0, 1) * rotation_matrix(angleXT, 0, 3) * translation((-1.0f) * center);

	std::vector<Point5D> hypercube_points{};
	for (int i = 0; i < 16; i++)
	{
		hypercube_points.push_back(trans * Point5D{0.5f * (i & 1), 0.5f * ((i & 2) != 0), 0.5f * ((i & 4) != 0), 0.5f * ((i & 8) != 0), 1.0});
	}
	std::vector<TetraLabel> labels_cube{ {{0, 4, 2, 7}, 0b110000},
										{{6, 4, 2, 7}, 0b111000},
										{{0, 2, 1, 7}, 0b110000},
										{{3, 2, 1, 7}, 0b111000},
										{{0, 1, 4, 7}, 0b110000},
										{{5, 1, 4, 7}, 0b111000} };
	std::vector<TetraLabel> hypercube_labels{};
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < labels_cube.size(); j++)
		{
			TetraLabel l = labels_cube[j];
			TetraLabel result;
			std::transform(std::begin(l.pos), std::end(l.pos), std::begin(result.pos),
				[i](int c)
			{int lower_bits = c & ((1 << (i / 2)) - 1);
			int higher_bits = c ^ lower_bits;
			return lower_bits | ((i & 1) << (i / 2)) | (higher_bits << 1);
			});
			result.visibility = l.visibility;
			hypercube_labels.push_back(result);
		}
	}
	HyperShape hyper_cube = HyperShape( hypercube_points, hypercube_labels);
	std::vector<Tetrahedron> all_tetra = hyper_cube.get_tetrahedrons();
	int num_of_tetra = all_tetra.size();
	/*
	int const num_of_tetra = 5;
	Point5D simp5[5]{
	{ 0.0f, 0.0f, 0.0f, 0.0f, 1.0f }
	,{ 0.5f, 0.0f, 0.0f, 0.0f, 1.0f}
	,{ 0.0f , 0.5f, 0.0f, 0.0f, 1.0f}
	,{ 0.0f , 0.0f, 0.5f, 0.0f, 1.0f}
	,{ 0.0f , 0.0f, 0.0f, 0.5f, 1.0f} };

	Point4D simp[5];
	for (int i = 0; i < num_of_tetra; i++)
	{
		simp[i] = (trans * simp5[i]).to4D();
	}

	Tetrahedron all_tetra[num_of_tetra]{
	Tetrahedron{ simp[0], simp[1], simp[2], simp[3], 127 },
	Tetrahedron{ simp[0], simp[1], simp[2], simp[4], 127 },
	Tetrahedron{ simp[0], simp[1], simp[3], simp[4], 127 },
	Tetrahedron{ simp[0], simp[2], simp[3], simp[4], 127 },
	Tetrahedron{ simp[1], simp[2], simp[3], simp[4], 127 }
	};
	Triangle all_triangles[num_of_tetra * 4];*/
	//printf("number of triangle %d\n", num_of_tetra);
	Triangle all_triangles[100 * 4];
	for (int i = 0; i < num_of_tetra; i++)
	{
		all_tetra[i].add_triangle_to_array(&all_triangles[4 * i]);
	}
	

	//printf("(%d %d)", pixelX, pixelY);
	CalculateImage(scene, all_triangles, num_of_tetra * 4, imageW, imageH, d_dst);
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

int main(int argc, char **argv)
{
	//initData(argc, argv);

	initGL(&argc, argv);
	printf("after initGL \n");
	initOpenGLBuffers(imageW, imageH);
	glutCloseFunc(cleanup);
	setVsync(0);
	sdkCreateTimer(&hTimer);
	sdkStartTimer(&hTimer);
	glutMainLoop();

	/*if (cudaStatus != cudaSuccess) {
		   fprintf(stderr, "addWithCuda failed!");
		   return 1;
	   }

	   printf("{result pixal: %f\n", res);

	   // cudaDeviceReset must be called before exiting in order for profiling and
	   // tracing tools such as Nsight and Visual Profiler to show complete traces.
	   cudaStatus = cudaDeviceReset();
	   if (cudaStatus != cudaSuccess) {
		   fprintf(stderr, "cudaDeviceReset failed!");
		   return 1;
	   }*/

	return 0;
}

