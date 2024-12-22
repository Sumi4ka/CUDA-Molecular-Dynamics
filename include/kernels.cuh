#pragma once
#include "device_launch_parameters.h"
__global__ void atomCoofTensor(float* ABCDEF, float* d_xA, float* d_yA, float* d_zA, int M);
__global__ void atomPrepareStepD(float* d_xA, float* d_yA, float* d_zA, float* d_xA0, float* d_yA0, float* d_zA0, float* d_xA1, float* d_yA1, float* d_zA1, float* d_xC, float* d_yC, float* d_zC, float xM, float yM, float zM, int M, int n);
__global__ void atomCalculateFLJ(float* d_xC, float* d_yC, float* d_zC, float* UVW, float* Lxyz, int d_N, int d_M, int d_n);
__global__ void atomVelocity(float* d_xA, float* d_yA, float* d_zA, float* d_xA1, float* d_yA1, float* d_zA1, float* d_uA, float* d_vA, float* d_wA, float omegaX, float omegaY, float omegaZ, int M, int dt);
__global__ void atomStep(float* d_xA, float* d_yA, float* d_zA, float* d_xA0, float* d_yA0, float* d_zA0, float* d_xC, float* d_yC, float* d_zC, float* d_uA, float* d_vA, float* d_wA, float xM, float yM, float zM, int n, int M, float dt);
