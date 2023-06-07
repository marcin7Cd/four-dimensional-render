# four-dimensional-render
experimental real time rendered of four dimensional objects written in c++ with cuda.

# How 4D is displayed
The 4D object is projected onto a 3D spaces and then its projected onto the screen. So there are two cameras one in 4D and one in 3D. The key feature is that the projection onto 3D space correctly calculates occlusions. (The object lower in 4th dimension can occlude object placed higher). The 3D "faces" of 4D object are rendered in the following way: their edges are solid and colored according to distance from the screen; their 2D faces are transparent and tinted white. The effect looks something like this for a hupercube.



https://github.com/marcin7Cd/four-dimensional-render/assets/34893204/0139fd30-5b0c-4aa4-ac35-3f85b98d26fe


# How to compile it
Ideally, you would just compile & run it with Visual Studio 2017. You would need cuda and openGL installed to run it. But I haven't tested it or written it for different machines and systems, so you need to figure out how to run it on your own. I used Windows 10  (64-bit) with GeForce GTX 1050 graphics card and I had cuda 10.1 installed. 
# How to use it
In the first window you write the name of a file you would like to render. I included 3 files "hypercube", "cell16" and "cell24" (no file extentions). Then the 400x400 window appears (changing the window size is broken). With a mouse you can rotate the projection in the 3D space (click and drag). With a keybord using keys:
 + Q, A you can rotate it in a YT plan
 + W, S you can rotate it in a XT plane
 + E, D you can rotate it in a ZT plane

# Creating you own files to render
In the first line is the number of vertices n and number of faces f
The next n lines are coordinates of vertecies (4 numbers)

Then you write each face. The format of the face is as follows:
The first line is the number of triangles the face is decomposed into.
Each next line specifies triangle. You write indices (based on the list at the beggining) of each vertex of the triangle. And at the end you write 0-1 sequence specifing, which edges should be drawn. Edges are considered in the order 0-1 then 0-2 then 1-2. So for example 
2 4 6 011
means the the triangle has vertices 2, 4 and 6, and the edges 2-6 and 4-6 are drawn.

# Idea behind the rendering
Instead of calculating projection onto 3D space and then onto 2D space. I combine them. From each pixel on the screen I draw a "ray" plane that passes through 3 points: the pixel on the screen, the eye position of the projection onto screen, and the eye position of the projection onto 3D space. Then I draw on this plane the ordinary ray from camera in 3D space through the pixel on the screen. Then I caluclate all intersections with objects (the are 2D shapes on this plane) and decide, what part will be hit by the ordinary ray, when the object were to be projected onto the 3D space (it works, because this projection will be on this ray thanks to the choice of the ray plane).
