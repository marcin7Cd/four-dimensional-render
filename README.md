# four-dimensional-render
experimental real time renderer (on GPU) of four dimensional objects written in c++ with cuda.

# How 4D is displayed
The 4D object is projected onto a 3D viewspaces and then its projected onto the screen (look at diagram bewlow). We have two two cameras one in 4D and one in 3D.

![projection explanation](https://github.com/marcin7Cd/four-dimensional-render/assets/34893204/ba7dda40-f0d7-4401-ae7e-406a7e82f117)

The rendered properly deals with occlusions in 4th dimension. The object, which is lower in 4th dimension can occlude the object placed higher. The 3D "faces" of 4D object are rendered in the following way: their edges are solid and colored according to the distance from the screen; the bondaries between 3D "faces" of the 4D object are semi-transparent with a white tint. The final effect looks something like this. Below I show a hypercube.



https://github.com/marcin7Cd/four-dimensional-render/assets/34893204/0139fd30-5b0c-4aa4-ac35-3f85b98d26fe


# How to compile it
Ideally, you would just compile & run it with Visual Studio 2017. You would need cuda and openGL installed to run it. But I haven't tested it or written it for different machines and systems, so you need to figure out how to run it on your own. I run it on Windows 10 (64-bit) with GeForce GTX 1050 graphics card and I had cuda 10.1 installed. 
# How to use it
In the first window you write the name of a file you would like to render. I included 3 files "hypercube", "cell16" and "cell24" (no file extentions). Then the 400x400 window appears (changing the window size is unsupported). With a mouse you can rotate the projection in the 3D space (click and drag). With a keybord using keys:
 + Q, A you can rotate it in a YT plane
 + W, S you can rotate it in a XT plane
 + E, D you can rotate it in a ZT plane

# Creating you own files to render
In the first line is the number of vertices n and number of faces f. 

The next n lines are coordinates of vertecies (4 numbers x y z t)

Then you write each face. The format of the face is as follows:

The first line is the number of triangles the face is decomposed into.

Each next line specifies the triangle. You write indices of each vertex of the triangle (numeration based on the list of vertices at the beggining and starts with 0). And at the end you write 0-1 sequence specifing, which edges should be drawn. Edges are considered in the order 0-1 then 0-2 then 1-2. So for example 

2 4 6 011

means the the triangle has vertices 2, 4 and 6, and the edges 2-6 and 4-6 are drawn.

# Idea behind the rendering
Instead of calculating projection onto 3D viewspace and then onto screen. I combine them. From each pixel on the screen I draw a "ray" plane that passes through 3 points: the pixel on the screen, the position of the 3d camera, and the position of the 4d camera. Then I draw on this plane the ordinary ray from 3D camera through the pixel on the screen. Then I caluclate all intersections with objects (the are 2D shapes on this plane) and decide, what part will be hit by the ordinary ray, when the object were to be projected onto the 3D space (it works, because this projection will be on this ray thanks to the choice of the ray plane).

![ray plane explanation](https://github.com/marcin7Cd/four-dimensional-render/assets/34893204/df28a46e-23d1-4583-a06f-9240d1303471)


