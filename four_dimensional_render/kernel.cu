

#include "cuda_runtime.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "kernel.h"


/*
struct RayPoint{
	float x;
	float y;
	float visibility;
};

struct PointData
{
	float x;
	float y;
	float slope;
};

struct Point4D
{
	float cord[4];

};

struct Triangle
{
	Point4D p[3];
};

struct Matrix4x4
{
	Point4D col[4];
};

struct Tetrahedron
{
	Point4D p[4];

	void add_triangle_to_array(Triangle* array)
	{
		array[0] = Triangle{ p[0], p[1], p[2] };
		array[1] = Triangle{ p[0], p[1], p[3] };
		array[2] = Triangle{ p[0], p[2], p[3] };
		array[3] = Triangle{ p[1], p[2], p[3] };
	}
};

//Tetrahedron operator*(Matrix4x4 a, Tetrahedron b)
//{
//	return Tetrahedron{ a * b.p[0], a*b.p[1], a*b.p[2] , a*b.p[3] };
//}


struct SceneData
{
	Point4D origin;
	Point4D focal_point;
	Point4D direction;
	Point4D directionX;
	Point4D directionY;
};

struct TriangleTmpData
{
	Point4D a_x;
	Point4D a_y;
	Point4D a_0;
	float b_x;
	float b_y;
	float b_0;
};
*/
__device__ Point4D operator+ (Point4D a, Point4D b)
{
	return Point4D{ a.cord[0] + b.cord[0],
					a.cord[1] + b.cord[1],
					a.cord[2] + b.cord[2],
					a.cord[3] + b.cord[3] };
}

__device__ Point4D operator* (Point4D a,float r)
{
	return Point4D{ r*a.cord[0],r*a.cord[1], r*a.cord[2], r*a.cord[3] };
}
__device__ Point4D operator* (Point4D a, Point4D b)
{
	return Point4D{ a.cord[0] * b.cord[0],
					a.cord[1] * b.cord[1],
					a.cord[2] * b.cord[2],
					a.cord[3] * b.cord[3] };
}



/*__host__ Point4D operator* (Matrix4x4 a, Point4D b)
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
}*/


#define BlockSize1 64

// size has to be divisible by 2
__device__ bool VisibleCalcSection(float p_slope, float p_x, float p_y, bool is_ignored, int from, int count, const RayPoint *a)
{
	int i = threadIdx.x;
	__shared__ PointData points[BlockSize1];
	
	if (i < count)
	{
		if (from == 0)
		{
			points[i] = PointData{ p_x, p_y, p_slope };
		}
		else
		{
			float b_x = a[from + i].x;
			float b_y = a[from + i].y;
			points[i] = PointData{ b_x, b_y, b_x / b_y };
		}
	}
	__syncthreads();
	if (is_ignored) return true; //the code for coping can't be skipped as it's needed by other threads
	bool is_blocked = false;
	for (int j = 0; j < count; j += 2)
	{
		int k = points[j].slope < points[j ^ 1].slope ? j : j ^ 1;
		if (points[k].slope < p_slope && p_slope < points[k ^ 1].slope) // the point is in the range of a segment
		{
			if ((points[k].x - points[k ^ 1].x) * (p_y - points[k].y) -
				(points[k].y - points[k ^ 1].y) * (p_x - points[k].x) <= 0)
			{
				is_blocked = true;
				break;
			}
		}
	}
	//printf("ok %d %c %c\n", i, is_blocked ? 't' : 'f', is_ignored ? 't' : 'f');
	return is_blocked;
}

__device__ void VisibleKernel(uchar4 *c, const RayPoint *a, unsigned int size)
{
	int i = threadIdx.x;
	__shared__ float real_vis[BlockSize1];
	__shared__ bool show_state;
	if (i == 0) show_state = false;
	float p_x = 0.0;
	float p_y = 0.0;
	float p_slope = 0.0;
	float p_vis = 0.0;
	float final_vis = 0.0;
	bool is_blocked = false;
	for (int k = 0; k < (size - 1) / BlockSize1 + 1; k++)
	{
		int pos = i + k * BlockSize1;
		if (pos < size)
		{
			p_x = a[pos].x;
			p_y = a[pos].y;
			p_vis = a[pos].visibility;
			p_slope = p_x / p_y;
			if (p_x == 0.0) show_state = true;
		}
		else
		{
			p_vis = 0.0;
		}
		is_blocked = false;
		//if (pos < size) printf("position %d visibility %f is blocked %c \n", pos, p_vis, is_blocked ? 't' : 'f');
		int num_loops = (size - 1) / BlockSize1 + 1;
		for (int m = 0; m < num_loops - 1; m++)
		{
			__syncthreads();
			if (VisibleCalcSection(p_slope, p_x, p_y, (pos >= size) || is_blocked, m*BlockSize1, BlockSize1, a))
				is_blocked = true; 
			// I can't do a break, because all threads need to execute the VisibleCalcSection
			// to still ignore checking unnecessary conditions I pass var is_blocked to VisibleCalcSection 
		}
		is_blocked = VisibleCalcSection(p_slope, p_x, p_y, (pos >= size) || is_blocked, (num_loops - 1)*BlockSize1, size - (num_loops - 1)*BlockSize1, a);
		
		//real_vis[i] = p_vis;
		real_vis[i] = is_blocked ? 0.0 : p_vis;
		//if (pos < size) printf("position %d visibility %f is blocked %c \n", pos, p_vis, is_blocked ? 't' : 'f');
		__syncthreads();
		// calculate maximum 
		for (int range = 2; range < BlockSize1; range <<= 1)
		{
			if (i % range == 0)
			{
				real_vis[i] = real_vis[i] > real_vis[i + (range / 2)] ? real_vis[i] : real_vis[i + (range / 2)];
			}
			__syncthreads();
		}
		if (i == 0)
		{
			float tmp = real_vis[0] > real_vis[BlockSize1 / 2] ? real_vis[0] : real_vis[BlockSize1 / 2];
			final_vis = final_vis > tmp ? final_vis : tmp;
		}
	}
	if (i == 0)
	{
		if (show_state)
		{
			c->x = 255;
			c->y = 0;
			c->z = 0;
		}
		else
		{
			char val = (final_vis > 0.5) ? 255 : 0;
			c->x = val;
			c->y = val;
			c->z = val;
		}
	}
	__syncthreads();
}

// direction is a combination of two vectors D_x, D_y, this way the pixel position (x,y) on the screen
// coresponds to direction vector x * D_x + y*D_y + D_0
// then using linearity of determinant with respect to one column we can parametrize
// them with respect to x and y.

// then we can calculate vector of intersections by doing (x*a_x + y*a_y + a)/(x*b_x + y_b_y + b)

__device__ float sub_determinant(Matrix4x4 A, int i0, int i1, int i2)
{
	return A.col[0].cord[i0] * A.col[1].cord[i1] * A.col[2].cord[i2] +
		A.col[0].cord[i1] * A.col[1].cord[i2] * A.col[2].cord[i0] +
		A.col[0].cord[i2] * A.col[1].cord[i0] * A.col[2].cord[i1] -
		A.col[0].cord[i2] * A.col[1].cord[i1] * A.col[2].cord[i0] -
		A.col[0].cord[i1] * A.col[1].cord[i0] * A.col[2].cord[i2] -
		A.col[0].cord[i0] * A.col[1].cord[i2] * A.col[2].cord[i1];
}

__device__ float determinant(Matrix4x4 A)
{
	return -A.col[3].cord[0] * sub_determinant(A, 1, 2, 3) +
		A.col[3].cord[1] * sub_determinant(A, 0, 2, 3) -
		A.col[3].cord[2] * sub_determinant(A, 0, 1, 3) +
		A.col[3].cord[3] * sub_determinant(A, 0, 1, 2);
}

__device__ Point4D cramer_dets(Matrix4x4 a, Point4D b)
{
	return Point4D{ determinant(Matrix4x4{b, a.col[1], a.col[2], a.col[3]}),
					determinant(Matrix4x4{a.col[0], b, a.col[2], a.col[3]}),
					determinant(Matrix4x4{a.col[0], a.col[1], b, a.col[3]}),
					determinant(Matrix4x4{a.col[0], a.col[1], a.col[2], b}) };
}


// call with one dimensional threads and blocks
//solutions of vector equation
// [B-A  C-A  -D  F-O][u v x y]^T = F-C
// which comes from 
// A + (B-A)u + (C-A)v = P = Dx + (O-F)y + F
// 

__device__ void calc_2_eq(float a_x, float a_y, float a_0,
	                      float b_x, float b_y, float b_0, float *x, float *y)
{
	float det = a_x * b_y - a_y * b_x;
	*x = (a_0*b_y - a_y * b_0) / det;
	*y = (a_x*b_0 - a_0 * b_x) / det;
}

__global__ void calculate_triangle(Triangle* t, unsigned int size, SceneData scene, TriangleTmpData *result)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
	{
		Matrix4x4 m{ t[i].p[1] + t[i].p[0] * (-1),
					 t[i].p[2] + t[i].p[0] * (-1),
					 scene.direction*(-1),
					 scene.origin*(-1) + scene.focal_point };
		Point4D b = scene.focal_point + t[i].p[0] * (-1);
		Point4D a_0 = cramer_dets(m, b);
		float b_0 = determinant(m);
		m.col[2] = scene.directionX * (-1);
		Point4D a_x = cramer_dets(m, b);
		a_x = a_x * Point4D{ 1,1,0,1 };
		float b_x = determinant(m);

		m.col[2] = scene.directionY * (-1);
		Point4D a_y = cramer_dets(m, b);
		a_y = a_y * Point4D{ 1,1,0,1 };
		float b_y = determinant(m);
		
		float px, py, x_min, x_max, y_max, y_min;
		calc_2_eq(a_x.cord[0], a_y.cord[0], -a_0.cord[0],
			a_x.cord[1], a_y.cord[1], -a_0.cord[1], &px, &py); //u =0, v=0
		x_min = px;
		x_max = px;
		y_max = py;
		y_min = py;

		calc_2_eq(a_x.cord[0] - b_x, a_y.cord[0] - b_y, -a_0.cord[0] + b_0,
			a_x.cord[1], a_y.cord[1], -a_0.cord[1], &px, &py); //u =1, v=0
		x_min = px < x_min ? px : x_min;
		x_max = px > x_max ? px : x_max;
		y_min = py < y_min ? py : y_min;
		y_max = py > y_max ? py : y_max;

		calc_2_eq(a_x.cord[0], a_y.cord[0], -a_0.cord[0],
			a_x.cord[1] - b_x, a_y.cord[1] - b_y, -a_0.cord[1] + b_0, &px, &py); //u =0, v=1
		x_min = px < x_min ? px : x_min;
		x_max = px > x_max ? px : x_max;
		y_min = py < y_min ? py : y_min;
		y_max = py > y_max ? py : y_max;

		result[i] = TriangleTmpData{ a_x, a_y, a_0, b_x, b_y, b_0, t[i].visibility_flags, x_min, x_max, y_min, y_max };

	}
}

//template<typename T>
__device__ void gather_non_empty(bool is_full, RayPoint* data1, RayPoint* data2, RayPoint* results, int unsigned* res_size, unsigned int maksimal_size)
{
	unsigned int i = threadIdx.x;
	__shared__ unsigned int positions[BlockSize1];
	positions[i] = is_full ? 1 : 0;
	__syncthreads();
	for (unsigned int pos = 1; pos < BlockSize1; pos <<= 1)
	{
		if ((i& pos) != 0)
		{
			positions[i] += positions[(i & ((BlockSize1-1) ^ (pos*2 - 1))) + pos-1];
		}
		__syncthreads();
	}
	if (is_full)
	{
		results[2 * positions[i] - 2] = *data1;
		results[2 * positions[i] - 1] = *data2;
	}
	if (i == 0)
		*res_size += 2 * positions[BlockSize1 - 1];
	
	__syncthreads();
}
// triangle are put in the array in pairs of fours so that each 4 correspond to 1 tetrahedron 
__global__ void calculate_pixel(TriangleTmpData* t, int unsigned data_size, uchar4* pixels, int width)
{
	unsigned int i = threadIdx.x;
	int x = blockIdx.x;
	int y = blockIdx.y;
	__shared__ RayPoint intersections[2*BlockSize1];
	__shared__ unsigned int size;
	__shared__ unsigned int last_i;
	if (i == 0)
	{
		size = 0;
	}
	__syncthreads();
	for (int pos = 0; pos < (data_size - 1)/(BlockSize1*4) + 1; pos++) 
	{
		RayPoint a{ 0.0f, 0.0f, 0.0f };
		RayPoint b{ 0.0f, 0.0f, 0.0f };
		bool is_intersected = false;
		for (int m = 0; m < 4; m++)
		{
			int k = pos * BlockSize1 * 4 + i*4 + m;
			if (k < data_size)
			{
				if (t[k].x_min <= x && x <= t[k].x_max &&
					t[k].y_min <= y && y <= t[k].y_max) {
					float denominator = (t[k].b_x * x + t[k].b_y *y + t[k].b_0);
					float p_u = (t[k].a_x.cord[0] * x + t[k].a_y.cord[0] * y + t[k].a_0.cord[0]) / denominator;
					if (0 < p_u)
					{
						float p_v = (t[k].a_x.cord[1] * x + t[k].a_y.cord[1] * y + t[k].a_0.cord[1]) / denominator;
						if (0 < p_v && p_u + p_v < 1)
						{
							float p_x = (t[k].a_x.cord[2] * x + t[k].a_y.cord[2] * y + t[k].a_0.cord[2]) / denominator;
							float p_y = (t[k].a_x.cord[3] * x + t[k].a_y.cord[3] * y + t[k].a_0.cord[3]) / denominator;
							float width = 0.02;
							bool is_visible = ((t[k].visibility & 0b10) && p_u < width) ||
								((t[k].visibility & 0b1) && p_v < width) ||
								((t[k].visibility & 0b100) && p_u + p_v > 1 - width);
							if (!is_intersected) // if is set to false, than it has to be the first intersection.
							{
								a.x = p_x;
								a.y = p_y;
								a.visibility = is_visible ? 1.0 : 0.0;
							}
							else
							{
								b.x = p_x;
								b.y = p_y;
								b.visibility = is_visible ? 1.0 : 0.0;
							}
							is_intersected = true;
							
						}
					}
				}
			}
		}
		__syncthreads();
		gather_non_empty(is_intersected, &a, &b, &intersections[size], &size, data_size);
	}
	__syncthreads();
	if (size != 0)
	{
		VisibleKernel(&pixels[y * width + x], intersections, size);
	}
	else
	{
		if (i == 0)
		{
			pixels[y * width + x].x = 0;
			pixels[y * width + x].y = 0;
			pixels[y * width + x].z = 0;
		}
	}
}

void CalculateImage(SceneData scene, Triangle* triangles, unsigned int size, int width, int height, uchar4* dst)
{

	Triangle *dev_triangles = 0;
	TriangleTmpData *dev_t_data = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_triangles, size * sizeof(Triangle));

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMalloc((void**)&dev_t_data, size * sizeof(TriangleTmpData));

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_triangles, triangles, size * sizeof(Triangle), cudaMemcpyHostToDevice);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	calculate_triangle << <size, 1 >> > (dev_triangles, size, scene, dev_t_data);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calculate_triangle!\n", cudaStatus);
		goto Error;
	}
	uint3 blocks;
	blocks.x = width;
	blocks.y = height;
	blocks.z = 1;
	uint3 threads;
	threads.x = BlockSize1;
	threads.y = 1;
	threads.z = 1;
	calculate_pixel << <blocks, threads >> > (dev_t_data, size, dst, width);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching caculate_pixel!\n", cudaStatus);
		goto Error;
	}

Error:
	cudaFree(dev_triangles);
	cudaFree(dev_t_data);
}