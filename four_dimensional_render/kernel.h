#pragma once

struct RayPoint
{
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
	char visibility_flags;
};

struct Matrix4x4
{
	Point4D col[4];
};

struct Tetrahedron
{
private:
	const static char AB_FLAG = 0b100000;
	const static char AC_FLAG = 0b10000;
	const static char AD_FLAG = 0b1000;
	const static char BC_FLAG = 0b100;
	const static char BD_FLAG = 0b10;
	const static char CD_FLAG = 0b1;
public:
	Point4D p[4];
	char visibility_flags;
	void add_triangle_to_array(Triangle* array)
	{
		Point4D a = p[0];
		Point4D b = p[1];
		Point4D c = p[2];
		Point4D d = p[3];
		array[0] = Triangle{{ a, b, c }, 
			((visibility_flags & AB_FLAG)!=0) | 
			(((visibility_flags & AC_FLAG)!=0)*2) |
		    (((visibility_flags & BC_FLAG)!=0)*4)};
		array[1] = Triangle{ {a, b, d} ,
		    ((visibility_flags & AB_FLAG)!=0) |
		    (((visibility_flags & AD_FLAG)!=0)*2) |
		    (((visibility_flags & BD_FLAG)!=0)*4)};
		array[2] = Triangle{ {a, c, d} ,
		    ((visibility_flags & AC_FLAG)!=0) |
		    (((visibility_flags & AD_FLAG)!=0)*2) |
		    (((visibility_flags & CD_FLAG)!=0)*4)};
		array[3] = Triangle{ {b, c, d },
			((visibility_flags & BC_FLAG) != 0) |
			(((visibility_flags & BD_FLAG) != 0) * 2) |
			(((visibility_flags & CD_FLAG) != 0) * 4) };
	}
};

/*Tetrahedron operator*(Matrix4x4 a, Tetrahedron b)
{
	return Tetrahedron{ a * b.p[0], a*b.p[1], a*b.p[2] , a*b.p[3] };
}*/


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
	char visibility;
	float x_min;
	float x_max;
	float y_min;
	float y_max;
};

extern "C" void CalculateImage(SceneData scene, Triangle* triangles, unsigned int size, int width, int height, uchar4* dst);
