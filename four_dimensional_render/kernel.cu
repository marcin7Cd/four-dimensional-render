

#include "cuda_runtime.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "kernel.h"


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

float calculate_norm(Point4D a)
{
	return a.cord[0] * a.cord[0] + a.cord[1] * a.cord[1] + a.cord[2] * a.cord[2] + a.cord[3] * a.cord[3];
}

#define BlockSize1 128

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
		/*float eps = 0.05;
		if (abs(points[j].x-p_x)< eps && abs(points[j].y - p_y) < eps)
			continue;
		if (abs(points[j+1].x - p_x) < eps && abs(points[j+1].y - p_y) < eps)
			continue;/**/
		
		int k = points[j].slope < points[j ^ 1].slope ? j : j ^ 1;
		//int k = points[j].slope < points[j ^ 1].slope ? j : j ^ 1;
		//if (p_y*points[k].x < p_x *points[k].y && points[k ^ 1].y*p_x < points[k^1].x* p_y) //assumes that y cord is always positive
		//float diff = points[k ^ 1].slope - points[k].slope;
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
	__shared__ float dist[BlockSize1];
	__shared__ bool show_state;
	if (i == 0) show_state = false;
	float p_x = 0.0;
	float p_y = 0.0;
	float p_slope = 0.0;
	float p_vis = 0.0;
	float final_vis = 0.0;
	float final_dist = 1e10f;
	uchar4 p_color{0,0,0,0};
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
			p_color = a[pos].color;
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
		//if (i == 0)
		//	printf("[%.1f]", p_slope*100.0);
		dist[i] = (is_blocked || p_vis == 0.0) ? 1e10f : p_slope;//p_slope * 5.0f;
		//if (pos < size) printf("position %d visibility %f is blocked %c \n", pos, p_vis, is_blocked ? 't' : 'f');
		__syncthreads();
		// calculate minimum 
		for (int range = 2; range < BlockSize1; range <<= 1)
		{
			if (i % range == 0)
			{
				dist[i] = dist[i] < dist[i + (range / 2)] ? dist[i] : dist[i + (range / 2)];
			}
			__syncthreads();
		}
		float tmp = dist[0] < dist[BlockSize1 / 2] ? dist[0] : dist[BlockSize1 / 2];
		final_dist = final_dist < tmp ? final_dist : tmp;
	}
	__shared__ unsigned int num_of_barriers;
	if (i == 0)
		num_of_barriers = 0;
	__syncthreads();
	if (!is_blocked) {
	//if (p_slope < final_dist && !is_blocked) {
		atomicInc(&num_of_barriers, 200);
	}/**/
	__syncthreads();
	float alpha = 1 - powf(0.9, num_of_barriers);
	//float alpha = num_of_barriers;
	unsigned char brightness = (int)(alpha*100);//(int)(5 * alpha);
	if (i == 0)
	{
		
		/*c->x = (int)(5*(alpha));
		c->y = (int)(5*(alpha));
		c->z = (int)(5*(alpha));
		/**/
		/*c->x = 0;
		c->y = 0;
		c->z = 0;/**/
		c->x = brightness;
		c->y = brightness;
		c->z = brightness;
	}
	if (final_dist == p_slope) // WRONG when there is more points processed than threads.
	{
		//printf("[%f]", p_slope);
		//*c = p_color;
		//c->x = (255 - val)*alpha; +140 * (1 - alpha);
		//c->y = (255-val)*alpha + 70*(1-alpha);
		//c->z = val*alpha + (100)*(1-alpha);
		/**/
		float p_vis = ((p_slope - 0.8) * 2.0f);
		unsigned char val = (p_vis > 1.0) ? 255 : (unsigned char)(p_vis * 255);
		c->x = int((255 - val)*alpha);
		c->y = int((255 - val)*alpha);
		c->z = int((val)*alpha);/**/
	}
	if (i == 0)
	{
		if (show_state)
		{
			c->x = 255;
			c->y = 0;
			c->z = 0;
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

__global__ void calculate_triangle(TriangleFull* t, unsigned int size, SceneData scene, TriangleTmpData *result)
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

		// x_min_int = x_min <= 0.0f ? 0 : (unsigned int)(x_min);
		//unsigned int y_min_int = y_min <= 0.0f ? 0 : (unsigned int)(y_min);
		//unsigned int x_max_int = x_max <= 0.0f ? 0 : (unsigned int)(x_max) + 1;
		//unsigned int y_max_int = y_max <= 0.0f ? 0 : (unsigned int)(y_max) + 1;
		result[i] = TriangleTmpData{ a_x, a_y, a_0, b_x, b_y, b_0, t[i].first_cell, t[i].second_cell, t[i].visibility,
			x_min, x_max, y_min, y_max };

	}
}

//template<typename T>
__device__ void gather_non_empty_v2(bool is_full, RayPoint* data, RayPoint* results, int unsigned* res_size, unsigned int maksimal_size)
{
	unsigned int i = threadIdx.x;
	__shared__ unsigned int num_filled;
	__shared__ unsigned int position[BlockSize1];
	__shared__ unsigned int is_filled[BlockSize1];
	position[i] = 0;
	is_filled[i] = 0;
	if (i == 0)
		num_filled = 0;
	__syncthreads();
	bool is_second = false;
	if (is_full)
	{
		unsigned int offset = atomicInc(&is_filled[i / 4], 2);
		if (offset == 0)
		{
			unsigned int spot = atomicInc(&num_filled, BlockSize1);
			results[spot * 2] = *data;
			position[i / 4] = spot;
		}
		else
		{
			// I will copy after the sync so that postion array will be fully updated.
			is_second = true;
		}
	}
	__syncthreads();
	if (is_second)
	{
		results[2 * position[i / 4] + 1] = *data;
	}
	if (i == 0)
	{
		*res_size += 2 * num_filled;
	}
	__syncthreads();
}

__device__ void gather_non_empty(bool is_full, RayPoint* data, RayPoint* results, int unsigned* res_size, unsigned int maksimal_size)
{
	unsigned int i = threadIdx.x;
	__shared__ unsigned int positions[BlockSize1];
	positions[i] = is_full ? 1 : 0;
	__syncthreads();
	for (unsigned int pos = 1; pos < BlockSize1; pos <<= 1)
	{
		if ((i& pos) != 0)
		{
			positions[i] += positions[(i & ((BlockSize1 - 1) ^ (pos * 2 - 1))) + pos - 1];
		}
		__syncthreads();
	}
	if (is_full)
		results[positions[i] - 1] = *data;
	if (i == 0)
		*res_size += positions[BlockSize1 - 1];
	__syncthreads();
}
#define LINE_LEN 400
// triangle are put in the array in pairs of fours so that each 4 correspond to 1 tetrahedron 
__global__ void calculate_pixel(TriangleTmpData* t, int unsigned data_size, uchar4* pixels, int width, int height,
	float dir_norm, float dirX_norm, float dirY_norm)
{
	unsigned int i = threadIdx.x;
	__shared__ RayPoint intersections[BlockSize1];
	__shared__ unsigned int cell_intersection_count[BlockSize1];
	__shared__ unsigned int cell_pos[BlockSize1];
	__shared__ unsigned int size;
	__syncthreads();
	TriangleTmpData tk[10];
	int y_im = blockIdx.y;
	int x_im = blockIdx.x * LINE_LEN;
	float y = y_im - height / 2;
	float x = x_im - width / 2;
	for (int pos = 0; pos < (data_size - 1) / (BlockSize1)+1; pos++)
	{
		int k = pos * BlockSize1 + i;
		if (k < data_size)
		{
			tk[pos] = t[k];
		}
	}
	for (int r = 0; r < LINE_LEN; r++)
	{
		cell_pos[i] = 0;
		cell_intersection_count[i] = 0;
		if (i == 0)
		{
			size = 0;
		}
		__syncthreads();
		x_im = blockIdx.x * LINE_LEN + r;
		x = x_im - (width / 2);
		for (int pos = 0; pos < (data_size - 1) / (BlockSize1)+1; pos++)
		{
			RayPoint a{ 0.0f, 0.0f, 0.0f };
			bool is_intersected = false;
			int k = pos * BlockSize1 + i;
			if (k < data_size)
			{
				if (tk[pos].y_min <= y && y <= tk[pos].y_max &&
					tk[pos].x_min <= x && x <= tk[pos].x_max) {
					float denominator = (tk[pos].b_x * x + tk[pos].b_y *y + tk[pos].b_0);
					float p_u = (tk[pos].a_x.cord[0] * x + tk[pos].a_y.cord[0] * y + tk[pos].a_0.cord[0]) / denominator;
					if (0 < p_u)
					{
						float p_v = (tk[pos].a_x.cord[1] * x + tk[pos].a_y.cord[1] * y + tk[pos].a_0.cord[1]) / denominator;
						if (0 < p_v && p_u + p_v < 1)
						{
							float p_x = (tk[pos].a_x.cord[2] * x + tk[pos].a_y.cord[2] * y + tk[pos].a_0.cord[2]) / denominator;
							//p_x = p_x / sqrt(dirX_norm*x*x + dirY_norm * y*y + dir_norm); //normalize
							float p_y = (tk[pos].a_x.cord[3] * x + tk[pos].a_y.cord[3] * y + tk[pos].a_0.cord[3]) / denominator;
							float width = 0.02;
							bool is_visible = ((tk[pos].visibility & 0b10) && p_u < width) ||
								((tk[pos].visibility & 0b1) && p_v < width) ||
								((tk[pos].visibility & 0b100) && p_u + p_v > 1 - width);
							int scale = 10000;
							a.x = float(int(p_x*scale))/ scale; //discreate
							a.y = float(int(p_y*scale))/ scale;
							a.visibility = is_visible ? 1.0 : 0.0;
							//a.visibility = 1.0;
							is_intersected = true;
							/*if ((tk[pos].visibility & 0b1) && p_v < width)
								a.color = t[pos].edge_colors[0];
							if ((tk[pos].visibility & 0b10) && p_u < width)
								a.color = t[pos].edge_colors[1];
							if ((tk[pos].visibility & 0b100) && p_u + p_v > 1 - width)
								a.color = t[pos].edge_colors[2];*/
						}
					}
				}
			}
			__syncthreads();
			bool is_second_l = false;
			bool is_second_r = false;
			if (is_intersected)
			{
				//if (tk[pos].left_cell == -1 || tk[pos].right_cell == -1)
				//	printf("WRONG");
				unsigned int occupied = atomicInc(&cell_intersection_count[tk[pos].left_cell], BlockSize1);
				if (occupied == 0)
				{
					int spot = atomicInc(&size, BlockSize1);
					cell_pos[tk[pos].left_cell] = spot;
					intersections[spot * 2] = a;
				}
				else
					is_second_l = true;

				if (tk[pos].right_cell != EMPTY_CELL)
				{
					unsigned int occupied = atomicInc(&cell_intersection_count[tk[pos].right_cell], BlockSize1);
					if (occupied == 0)
					{
						int spot = atomicInc(&size, BlockSize1);
						cell_pos[tk[pos].right_cell] = spot;
						intersections[spot * 2] = a;
					}
					else
						is_second_r = true;
					//if (occupied > 1)
					//	printf("[%d]", occupied);
				}
			}
			__syncthreads();
			if (is_second_l)
				intersections[2 * cell_pos[tk[pos].left_cell] + 1] = a;
			if (is_second_r)
				intersections[2 * cell_pos[tk[pos].right_cell] + 1] = a;
			//gather_non_empty_v2(is_intersected, &a, &intersections[size], &size, data_size);
		}
		__syncthreads();
		if (size != 0)
		{
			//if (i == 0 && size<3)
			//	printf("[%d]", size);
			VisibleKernel(&pixels[y_im * width + x_im], intersections, 2*size);
		}
		else
		{
			if (i == 0)
			{
				pixels[y_im * width + x_im].x = 0;
				pixels[y_im * width + x_im].y = 0;
				pixels[y_im * width + x_im].z = 0;
			}
		}
		__syncthreads();
	}
}

void CalculateImage(SceneData scene, TriangleFull* triangles, unsigned int size, int width, int height, uchar4* dst)
{

	TriangleFull *dev_triangles = 0;
	TriangleTmpData *dev_t_data = 0;
	cudaError_t cudaStatus;

	float dir_n = calculate_norm(scene.direction);
	float dirX_n = calculate_norm(scene.directionX);
	float dirY_n = calculate_norm(scene.directionY);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_triangles, size * sizeof(TriangleFull));

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
	cudaStatus = cudaMemcpy(dev_triangles, triangles, size * sizeof(TriangleFull), cudaMemcpyHostToDevice);

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
	blocks.x = width / LINE_LEN + 1;
	blocks.y = height;
	blocks.z = 1;
	uint3 threads;
	threads.x = BlockSize1;
	threads.y = 1;
	threads.z = 1;

	calculate_pixel <<<blocks, threads>>> (dev_t_data, size, dst, width, height, dirX_n,dirY_n, dir_n);

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