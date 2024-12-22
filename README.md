# Name: CUDA Molecular Dynamics
## Description: 
This code is written to solve problems of dynamics of complex molecular structures.

### The trajectory approach
A feature of the solution is the consideration of the structure as an integral body 
with uniform translational and rotational characteristics.
This solution is implemented in C#, but this solution uses CUDA C++ to implement parallelization on the GPU.

- In "data\" directory there are two files such as Fullerene.txt and Nanotube.txt. In ReadData function you need to select file name.
- "results\" directory will store the obtained data of the trajectories of the molecules
- In the file "Description.txt " the initial data for the calculation is stored. They are recorded in a human readable form. The program parses this file, so the TABULATION in the file must be strictly adhered to.
- By default, the program parses the "Description.txt" file, but you can add a command-line argument (use .bat files) to run the program as a path to the .txt file.


## Instruction:
- The program requires: C++17 standart or newer, CUDA Toolkit 11.0 or newer
- To compile all file: run compile.bat
- To run the program: run DescriptinScript.bat or any else
- Or work with VS
