

#include <windows.h>
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
#include <map>
#include "kernel.h"
#include <algorithm>
#include <iterator>
#include <fstream>


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

struct TriangleLabel
{
	int a, b, c;
	char visibility;
};

const bool operator< (const TriangleLabel left, const TriangleLabel right)
{
	if (left.a != right.a)
		return left.a < right.a;
	if (left.b != right.b)
		return left.b < right.b;
	return left.c < right.c;
}

struct TrianglePart
{
	TriangleLabel triangle;
	int left_cell;
	int right_cell;
};

struct HyperShape2
{
	std::vector<Point5D> points;
	std::vector<TrianglePart> triangles;

	std::vector<TriangleFull> generate_triangles()
	{
		std::vector<TriangleFull> res{};
		for (auto t : triangles)
		{
			res.push_back({ {points[t.triangle.a].to4D(), 
				             points[t.triangle.b].to4D(), 
				             points[t.triangle.c].to4D()}, 
				            t.triangle.visibility, t.left_cell, t.right_cell });
		}
		return res;
	}
};

uchar4 generate_color(unsigned char val)
{
	//return uchar4{ (val % 3) * (255u / 3), ((val / 3) % 3) * (255u / 3), (val % 5) * (255u / 5) };
	return uchar4{ ((val/2)%2) *(255u),0,255,0 };
}

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
			int a = conn.pos[0], b = conn.pos[1], c = conn.pos[2], d = conn.pos[3];
			res.push_back(Tetrahedron{{points[a].to4D(), points[b].to4D(),
									   points[c].to4D(), points[d].to4D() },
									   conn.visibility, 
				                      {generate_color((c + d)%256),
									   generate_color((b + d)%256),
									   generate_color((b + c)%256),
									   generate_color((a + d)%256),
									   generate_color((a + c)%256),
									   generate_color((a + b)%256)
									   } });
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
StopWatchInterface *globalTimer = NULL;


#define REFRESH_DELAY     10

typedef BOOL(WINAPI *PFNWGLSWAPINTERVALFARPROC)(int);

#define BUFFER_DATA(i) ((char *)0 + i)

void setVsync(int interval)
{
	if (WGL_EXT_swap_control)
	{
		printf("sync cotntrol\n\n\n");
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
float prev_time = 0.0;
float cur_time = 0.0;
void displayFunc(void)
{
	sdkStartTimer(&hTimer);
	renderImage();
	
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
	prev_time = cur_time;
	cur_time = sdkGetTimerValue(&globalTimer);
	frame_time = (cur_time - prev_time)/1000.0f;
	//int diff = cur_time - prev_time;
	//int time_per_frame = 1000 / 50;
	//Sleep(diff > time_per_frame ? 0 : time_per_frame - diff);
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
float angleXZ = 0.0;
float angleYT = 0.0;
float angleZT = 0.0;
float speed = 1.0;
float move_time = 0.0;
float last_positionX = 0.0;
float last_positionY = 0.0;

void motionFunc(int x, int y)
{
	last_positionX = (float(x) / float(imageW) - 0.5);
	last_positionY = (float(y) / float(imageH) - 0.5);
	//printf("(%.1f, %.1f)", last_positionX, last_positionY);
}

Matrix5x5 motion_trans = identity_matrix();

void process_motion()
{
	float angleXY_dif = last_positionX * abs(last_positionX) * speed*frame_time * 5;
	float angleYZ_dif = last_positionY * abs(last_positionY) * speed*frame_time * 5;
	motion_trans = rotation_matrix(-angleYZ_dif, 0, 2) * rotation_matrix(angleXY_dif, 0, 1)  * motion_trans;
}

bool is_pressed_Q = false;
bool is_pressed_A = false;
bool is_pressed_W = false;
bool is_pressed_S = false;
bool is_pressed_E = false;
bool is_pressed_D = false;

struct Toggle_value
{
	bool value;
	bool previous_input;
	Toggle_value() {
		value = false;
		previous_input = false;
	}
	void input(bool is_key_pressed)
	{
		if (previous_input == false && is_key_pressed == true) {
			value = !value;
			if (value)
				std::cout << "program paused\n";
			else
				std::cout << "program unpaused\n";
		}
		previous_input = is_key_pressed;
	}
};

Toggle_value is_paused = Toggle_value();

void updateKeyPresses(unsigned char k, bool val)
{
	switch (k)
	{
	case 'q':
		is_pressed_Q = val;
		break;
	case 'a':
		is_pressed_A = val;
		break;
	case 'w':
		is_pressed_W = val;
		break;
	case 's':
		is_pressed_S = val;
		break;
	case 'e':
		is_pressed_E = val;
		break;
	case 'd':
		is_pressed_D = val;
		break;
	}
	is_paused.input(k == 'p' && val);
}

void keyboardFunc(unsigned char k, int, int)
{
	updateKeyPresses(k, true);
}

void keyboardFuncUp(unsigned char k, int, int)
{
	updateKeyPresses(k, false);
}

Matrix5x5 key_trans = identity_matrix();

void process_key_pressed()
{
	float angleYT_diff = 0.0;
	float angleXT_diff = 0.0;
	float angleZT_diff = 0.0;
	if (move_time > 1.0/12) {
	//if (frameCount % 5 == 0) {
		if (is_pressed_Q)
			angleYT_diff = speed * move_time;
		if (is_pressed_A)
			angleYT_diff = -speed * move_time;
		if (is_pressed_W)
			angleXT_diff = speed * move_time;
		if (is_pressed_S)
			angleXT_diff = -speed * move_time;
		if (is_pressed_E)
			angleZT_diff = speed * move_time;
		if (is_pressed_D)
			angleZT_diff = -speed * move_time;
		key_trans = rotation_matrix(angleYT_diff, 1, 3) * rotation_matrix(angleXT_diff, 0, 3) * rotation_matrix(angleZT_diff, 2, 3) * key_trans;
		move_time = 0.0f;
	}
	else
		move_time += frame_time;
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
	glutKeyboardUpFunc(keyboardFuncUp);
	//glutMouseFunc()
	glutMotionFunc(motionFunc);
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
	sdkStopTimer(&globalTimer);
	sdkDeleteTimer(&hTimer);
	sdkDeleteTimer(&globalTimer);
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glDeleteBuffers(1, &gl_PBO);
	glDeleteTextures(1, &gl_Tex);
	glDeleteProgramsARB(1, &gl_Shader);
}

HyperShape generate_hypercube() {
	std::vector<Point5D> hypercube_points{};
	for (int i = 0; i < 16; i++)
	{
		hypercube_points.push_back(Point5D{ 0.5f * (i & 1), 0.5f * ((i & 2) != 0), 0.5f * ((i & 4) != 0), 0.5f * ((i & 8) != 0), 1.0 });
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
	return HyperShape(hypercube_points, hypercube_labels);
}

//cudaError_t CalculateImage(SceneData scene, Triangle* triangles, unsigned int size, int width, int height, uchar4* dst);
float angle = 0.0;

HyperShape2 final_shape{ {},{} };

void renderImage()
{
	sdkResetTimer(&hTimer);
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, &num_bytes, cuda_pbo_resource));
	float range = 1.0f;
	//printf("%f", distX);
	SceneData scene{ Point4D{-2.0f, 0.25f, 0.25f, 0.0f},
	Point4D{0.125f, 0.125f, 0.125f, -10.0f},
	Point4D{2.0f, 0.0f, 0.0f, 0.0f},
	Point4D{0.0f, range / imageW, 0.0f, 0.0f},
	Point4D{0.0f, 0.0f, range / imageH, 0.0f} };
	float res = 0.0;
	process_key_pressed();
	process_motion();
	if (!is_paused.value)
	{
		//angleXY += 1.1 * frame_time;
		Point4D center = Point4D{ 0.25, 0.25, 0.25, 0.25 };
		Matrix5x5 trans = translation(center) * motion_trans * key_trans * translation((-1.0f) * center);

		HyperShape2 hyper_cube = final_shape;//generate_hypercube();
		for (int i = 0; i < hyper_cube.points.size(); i++)
		{
			hyper_cube.points[i] = trans * hyper_cube.points[i];
		}

		TriangleFull all_triangles[4 * 100];
		int num_of_triangles = 0;
		for (auto triangle : hyper_cube.generate_triangles())
		{
			all_triangles[num_of_triangles] = triangle;
			num_of_triangles += 1;
		}

		//printf("(%d %d)", pixelX, pixelY);
		CalculateImage(scene, all_triangles, num_of_triangles, imageW, imageH, d_dst);
		
	}
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}


HyperShape2 load_shape_file(std::string filename)
{
	std::ifstream file;
	file.open(filename);
	printf("opening a file\n");
	if (!file.is_open())
	{
		printf("reading a file failed");
		return HyperShape2{ {},{} };
	}
	int num_of_points;
	int num_of_cells;
	file >> num_of_points >> num_of_cells;
	//read points
	std::vector<Point5D> points{};
	for (int i = 0; i < num_of_points; i++)
	{
		float a, b, c, d;
		file >> a >> b >> c >> d;
		points.push_back(Point5D{ a,b,c,d,1.0 });
	}
	//read tetra connections
	std::map<TriangleLabel, std::pair<int, int>> labels{};
	
	for (int i = 0; i < num_of_cells; i++)
	{
		int num_of_triangles;
		file >> num_of_triangles;
		for (int j = 0; j < num_of_triangles; j++)
		{
			int a, b, c;
			file >> a >> b >> c;
			char c1 = 32;
			while (c1 != '0' && c1 != '1')
				file >> c1;
			char c2, c3;
			file >> c2 >> c3;
			char visibility = (c3 == '1') * 4 | (c2 == '1') * 2 | (c1 == '1');
			TriangleLabel current_label = TriangleLabel{ a,b,c, visibility };
			auto prev_pos = labels.find(current_label);

			if (prev_pos == std::end(labels))
				labels.insert({ current_label, { i, EMPTY_CELL } });
			else
			{
				prev_pos->second.second = i;
			}

		}
	}
	std::vector<TrianglePart> triangles{};
	for (auto v : labels)
	{
		//printf("[%d, %d, %d]", v.first.a, v.first.b, v.first.c);
		triangles.push_back({ v.first, v.second.first, v.second.second });
	}

	file.close();
	printf("file read\n");
	return HyperShape2{ points, triangles };
}

void write_shape_file(HyperShape shape, std::string filename)
{
	std::ofstream file;
	file.open(filename);
	file << shape.points.size() << " " << shape.tetrahedron_connections.size() << "\n";
	for (Point5D& p : shape.points)
	{
		file << p.cords[0] << " " << p.cords[1] << " " << p.cords[2] << " " << p.cords[3] << "\n";
	}
	file << "\n";
	for (TetraLabel& conn : shape.tetrahedron_connections)
	{
		file << conn.pos[0] << " " << conn.pos[1] << " " << conn.pos[2] << " " << conn.pos[3] << " ";
		int pos = 1;
		for (int i = 0; i < 6; i++)
		{
			file << ((conn.visibility & pos) / pos);
			pos = pos << 1;
		}
		file << "\n";
	}
	file.close();
}

int main(int argc, char **argv)
{
	//initData(argc, argv);
	std::string file_name;
	std::cout << "what file to load\n";
	std::cin >> file_name;
	//final_shape = load_shape_file("cell24");
	final_shape = load_shape_file(file_name);
	initGL(&argc, argv);
	printf("after initGL \n");
	initOpenGLBuffers(imageW, imageH);
	glutCloseFunc(cleanup);
	setVsync(1);
	sdkCreateTimer(&hTimer);
	sdkCreateTimer(&globalTimer);
	sdkStartTimer(&hTimer);
	sdkStartTimer(&globalTimer);
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

