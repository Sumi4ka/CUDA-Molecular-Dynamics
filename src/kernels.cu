﻿#include "kernels.cuh"
__constant__ float d_KB = 1.380649e-23, d_massA = 1.9944733e-26, d_sigma = 0.34;
__constant__ float d_eps = 1.7258113e-22; //h_KB * 12.5
__global__ void atomCoofTensor(float* ABCDEF, float* d_xA, float* d_yA, float* d_zA, int M)
{
    int i = threadIdx.x;
    float massA = d_massA;
    if (i < 6) ABCDEF[i] = 0.0f;
    __syncthreads();
    if (i < M)
    {
        atomicAdd(&ABCDEF[0], massA * (d_yA[i] * d_yA[i] + d_zA[i] * d_zA[i]));
        atomicAdd(&ABCDEF[1], massA * (d_zA[i] * d_zA[i] + d_xA[i] * d_xA[i]));
        atomicAdd(&ABCDEF[2], massA * (d_xA[i] * d_xA[i] + d_yA[i] * d_yA[i]));
        atomicAdd(&ABCDEF[3], -massA * d_yA[i] * d_zA[i]);
        atomicAdd(&ABCDEF[4], -massA * d_xA[i] * d_zA[i]);
        atomicAdd(&ABCDEF[5], -massA * d_xA[i] * d_yA[i]);
    }
}
__global__ void atomPrepareStepD(float* d_xA, float* d_yA, float* d_zA, float* d_xA0, float* d_yA0, float* d_zA0, float* d_xA1, float* d_yA1, float* d_zA1, float* d_xC, float* d_yC, float* d_zC, float xM, float yM, float zM, int M, int n)
{
    int i = threadIdx.x;
    if (i < M)
    {
        d_xA1[i] = d_xA0[i] = d_xA[i];
        d_yA1[i] = d_yA0[i] = d_yA[i];
        d_zA1[i] = d_zA0[i] = d_zA[i];
        d_xC[n * M + i] = d_xA[i] + xM;
        d_yC[n * M + i] = d_yA[i] + yM;
        d_zC[n * M + i] = d_zA[i] + zM;
    }
}
__global__ void atomCalculateFLJ(float* d_xC, float* d_yC, float* d_zC, float* UVW, float* Lxyz, int d_N, int d_M, int d_n)
{
    int M = d_M;
    int N = d_N;
    int n = d_n;
    int i = threadIdx.x;
    float ro, c, FLJ, FLJx, FLJy, FLJz, sigma = d_sigma, eps = d_eps, x, y, z;
    __shared__ float xS[256];
    __shared__ float yS[256];
    __shared__ float zS[256];
    x = d_xC[i + n * M]; y = d_yC[i + n * M]; z = d_zC[i + n * M];
    FLJx = 0.f, FLJy = 0.f, FLJz = 0.f;
    for (int j = 0; j < N; j++)                            //Цикл блоков
    {
        if (j != n)
        {
            if (threadIdx.x < M)
            {
                xS[threadIdx.x] = d_xC[threadIdx.x + j * M];
                yS[threadIdx.x] = d_yC[threadIdx.x + j * M];
                zS[threadIdx.x] = d_zC[threadIdx.x + j * M];
            }
            else
            {
                xS[threadIdx.x] = 0.f;
                yS[threadIdx.x] = 0.f;
                zS[threadIdx.x] = 0.f;
            }
            __syncthreads();
            if (threadIdx.x < M)
                for (int k = 0; k < M; k++)                              //Цикл потоков (нитей)
                {
                    //if ((k + j * M) != i)
                    {
                        ro = sqrtf(powf(x - xS[k], 2.f) + powf(y - yS[k], 2.f) + powf(z - zS[k], 2.f));
                        c = powf(sigma / ro, 6.f);
                        FLJ = 24.f * eps * c * (2.f * c - 1.f) / ro / ro;
                        FLJx += FLJ * (x - xS[k]);
                        FLJy += FLJ * (y - yS[k]);
                        FLJz += FLJ * (z - zS[k]);
                    }
                }
            __syncthreads();
        }
    }
    //instead of sharedUVW we used xS yS zS 
    __shared__ float sharedLxyz0[256];
    __shared__ float sharedLxyz1[256];
    __shared__ float sharedLxyz2[256];
    if (i < M)
    {
        xS[i] = FLJx;
        yS[i] = FLJy;
        zS[i] = FLJz;
        sharedLxyz0[i] = y * FLJz - z * FLJy;
        sharedLxyz1[i] = z * FLJx - x * FLJz;
        sharedLxyz2[i] = x * FLJy - y * FLJx;
    }
    else
    {
        xS[i] = 0.0;
        yS[i] = 0.0;
        zS[i] = 0.0;
        sharedLxyz0[i] = 0.0;
        sharedLxyz1[i] = 0.0;
        sharedLxyz2[i] = 0.0;
    }
    __syncthreads();
    // Редукция внутри блока
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (i < stride) {
            xS[i] += xS[i + stride];
            yS[i] += yS[i + stride];
            zS[i] += zS[i + stride];
            sharedLxyz0[i] += sharedLxyz0[i + stride];
            sharedLxyz1[i] += sharedLxyz1[i + stride];
            sharedLxyz2[i] += sharedLxyz2[i + stride];
        }
        __syncthreads();
    }
    if (i < 3) { UVW[i] = 0.f; Lxyz[i] = 0.f; }
    __syncthreads();
    if (i == 0)
    {
        UVW[0] = xS[0];
        UVW[1] = yS[0];
        UVW[2] = zS[0];
        Lxyz[0] = sharedLxyz0[0];
        Lxyz[1] = sharedLxyz1[0];
        Lxyz[2] = sharedLxyz2[0];
    }
}
__global__ void atomVelocity(float* d_xA, float* d_yA, float* d_zA, float* d_xA1, float* d_yA1, float* d_zA1, float* d_uA, float* d_vA, float* d_wA, float omegaX, float omegaY, float omegaZ, int M, int dt)
{
    int i = threadIdx.x;
    if (i < M)
    {
        d_uA[i] = omegaY * d_zA[i] - omegaZ * d_yA[i];
        d_vA[i] = omegaZ * d_xA[i] - omegaX * d_zA[i];
        d_wA[i] = omegaX * d_yA[i] - omegaY * d_xA[i];
        d_xA1[i] += dt * d_uA[i];
        d_yA1[i] += dt * d_vA[i];
        d_zA1[i] += dt * d_wA[i];
    }
}
__global__ void atomStep(float* d_xA, float* d_yA, float* d_zA, float* d_xA0, float* d_yA0, float* d_zA0, float* d_xC, float* d_yC, float* d_zC, float* d_uA, float* d_vA, float* d_wA, float xM, float yM, float zM, int n, int M, float dt)
{
    int i = threadIdx.x;
    if (i < M)
    {
        d_xA[i] = d_xA0[i] + dt * d_uA[i];
        d_yA[i] = d_yA0[i] + dt * d_vA[i];
        d_zA[i] = d_zA0[i] + dt * d_wA[i];
        d_xC[n * M + i] = d_xA[i] + xM;
        d_yC[n * M + i] = d_yA[i] + yM;
        d_zC[n * M + i] = d_zA[i] + zM;
    }
}