# Name: CUDA Molecular Dynamics
## Description: 
This code is written to solve problems of dynamics of complex molecular structures.
A feature of the solution is the consideration of the structure as an integral body 
with uniform translational and rotational characteristics.
This solution is implemented in C#, but this solution uses CUDA C++ to implement parallelization on the GPU.

- In "Data\" directory there are two files such as Fullerene.txt and Nanotube.txt. In ReadData function you need to select file name.
- "Results\" directory will store the obtained data of the trajectories of the molecules
- In the file "Description.txt " the initial data for the calculation is stored. They are recorded in a human readable form. The program parses this file, so the TABULATION in the file must be strictly adhered to.
- By default, the program parses the "Description.txt" file, but you can add a command-line argument to run the program as a path to the .txt file.


## Instruction:
- Download and install Visual Studio 2019 or higher;
- Download and install the CUDA Toolkit 11.0 or newer;
- How to Compile and Run the Program
  To compile all files and run the program, execute the following command in the terminal:

  ```bash
  bash Compile_script.sh

