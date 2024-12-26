cd /d "%~dp0"
nvcc -arch=sm_86 -O3 -std=c++17 -I./include src/main.cu src/solver.cu src/molecule.cu src/moleculeCPU.cu src/moleculeGPU.cu src/kernels.cu -o build/CUDA_MD