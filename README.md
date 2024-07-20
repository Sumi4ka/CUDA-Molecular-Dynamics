# Name: CUDA Molecular Dynamics
## Description: 
This code is written to solve problems of dynamics of complex molecular structures.
A feature of the solution is the consideration of the structure as an integral body 
with uniform translational and rotational characteristics.
This solution is implemented in C#, but this solution uses CUDA C++ to implement parallelization on the GPU.

## Instruction:
- Download and install Visual Studio 2019 or higher;
- Download and install the CUDA Toolkit 11.0 or newer;
- Launch CntGPU7.sln;
- In "Data\" directory there are two files such as Fullerene.txt and Nanotube.txt. In ReadData function you need to select file name.
- in main function you need to configure calculation characteristic:
  - T - count of iterations
  - dt - step of time
  - Data of Molecules
