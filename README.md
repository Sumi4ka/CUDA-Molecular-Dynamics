# Name: CUDA Molecular Dynamics
## Description: 
This code is written to solve problems of dynamics of complex molecular structures.

### The trajectory approach
A feature of the solution is the consideration of the structure as an integral body 
with uniform translational and rotational characteristics.
This solution is implemented in C#, but this solution uses CUDA C++ to implement parallelization on the GPU.

- In "data\" directory there are two files such as Fullerene.txt and Nanotube.txt.
- "results\" directory will store the obtained data of the trajectories of the molecules
- In the file "DescriptionN.txt " the initial data for the calculation is stored. They are recorded in a human readable form. The program parses this file, so the TABULATION in the file must be strictly adhered to.
- By default, the program parses the "Description.txt" file, but you can add a command-line argument (use .bat files) to run the program as a path to the .txt file.
- In "matlab_scripts\" directory there are scripts for visualizing and animating the results.

## Instruction:
- The program requires: C++17 standart or newer, CUDA Toolkit 11.0 or newer
- Clone repository
  ```bash
  git clone https://github.com/Sumi4ka/CUDA-Molecular-Dynamics
- To compile all file: run compile.bat or in terminal:
  ```bash
  cd CUDA-Molecular-Dynamics
  nvcc -arch=sm_86 -O3 -std=c++17 -I./include src/main.cu src/solver.cu src/molecule.cu src/moleculeCPU.cu src/moleculeGPU.cu src/kernels.cu -o build/CUDA_MD
- To run the program: run DescriptionScript.bat or any else. Or run the build/CUDA_MD.exe
- Or work with VS (run .sln)
