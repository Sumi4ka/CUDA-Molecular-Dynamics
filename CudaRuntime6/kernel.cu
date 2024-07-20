#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <direct.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <chrono>
#include <cooperative_groups.h>
#define N 2
#define M 240

//reading Data from .../Data/name.txt
void readData(std::string name, double* x, double* y, double* z)
{
    std::string sData = "Data\\";
    std::ifstream in(sData + name);
    std::string s1, xData, yData, zData;
    in >> s1;
    if (s1 == "Carbon")
    {
        in >> s1;
        int is1 = stoi(s1);
        for (int j = 0; j < M; j++)
        {
            in >> xData >> yData >> zData;
            for (int i = 0; i < N; i++)
            {
                x[i * M + j] = stod(xData);
                y[i * M + j] = stod(yData);
                z[i * M + j] = stod(zData);
            }
        }
    }
    in.close();
}
//preparing data for calculating
void Average(double* x, double* y, double* z)
{
    for (int i = 0; i < N; i++)
    {
        double avgX, avgY, avgZ;
        avgX = avgY = avgZ = 0.0;
        for (int j = 0; j < M; j++)
        {
            avgX += x[i * M + j];
            avgY += y[i * M + j];
            avgZ += z[i * M + j];
        }
        avgX /= M; avgY /= M; avgZ /= M;
        for (int j = 0; j < M; j++)
        {
            x[i * M + j] -= avgX;
            y[i * M + j] -= avgY;
            z[i * M + j] -= avgZ;
        }
    }
}
//global memory
__device__ volatile double xC[N * M], yC[N * M], zC[N * M];
//main kernell
__global__ void Solver
(
    int T, double dt, double KB, double massA, double massM, double sigma, double eps,
    double* h_x, double* h_y, double* h_z, double* xData, double* yData, double* zData,
    double* h_xM, double* h_yM, double* h_zM, double* h_uM, double* h_vM, double* h_wM,
    double* h_OmegaX, double* h_OmegaY, double* h_OmegaZ
) {
    int i = blockIdx.x, j = threadIdx.x;
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    double x1, y1;
    double xA = 0, yA = 0, zA = 0;
    double xA0 = 0, yA0 = 0, zA0 = 0;
    double xA1 = 0, yA1 = 0, zA1 = 0;
    double xA2 = 0, yA2 = 0, zA2 = 0;
    double xA3 = 0, yA3 = 0, zA3 = 0;
    double xA4 = 0, yA4 = 0, zA4 = 0;
    double FLJx = 0, FLJy = 0, FLJz = 0;
    __shared__ volatile double xM, yM, zM, uM, vM, wM, OmegaX, OmegaY, OmegaZ;
    __shared__ volatile double xM0, yM0, zM0, uM0, vM0, wM0, Kx0, Ky0, Kz0;
    __shared__ volatile double xM1, yM1, zM1, uM1, vM1, wM1, Lx1, Ly1, Lz1;
    __shared__ volatile double xM2, yM2, zM2, uM2, vM2, wM2, Lx2, Ly2, Lz2;
    __shared__ volatile double xM3, yM3, zM3, uM3, vM3, wM3, Lx3, Ly3, Lz3;
    __shared__ volatile double xM4, yM4, zM4, uM4, vM4, wM4, Lx4, Ly4, Lz4;
    __shared__ volatile double A, B, C, D, E, F, Kx, Ky, Kz, ub, vb, wb, Lx, Ly, Lz;
    if (j < M)
    {
        xA = h_x[i * M + j];
        yA = h_y[i * M + j];
        zA = h_z[i * M + j];
    }
    __syncthreads();
    if (j == 0)
    {
        xM = h_xM[i], yM = h_yM[i], zM = h_zM[i], 
        uM = h_uM[i], vM = h_vM[i], wM = h_wM[i],
        OmegaX = h_OmegaX[i], OmegaY = h_OmegaY[i], OmegaZ = h_OmegaZ[i];
        xData[i * T] = xM;
        yData[i * T] = yM;
        zData[i * T] = zM;
        A = 0.0; B = 0.0; C = 0.0; D = 0.0; E = 0.0; F = 0.0;
        for (int j1 = 0; j1 < M; j1++)
        {
            A += massA * (pow(yA, 2) + pow(zA, 2));
            B += massA * (pow(zA, 2) + pow(xA, 2));
            C += massA * (pow(xA, 2) + pow(yA, 2));
            D -= massA * yA * zA;
            E -= massA * xA * zA;
            F -= massA * xA * yA;
        }
        Kx = A * OmegaX + F * OmegaY + E * OmegaZ;
        Ky = F * OmegaX + B * OmegaY + D * OmegaZ;
        Kz = E * OmegaX + D * OmegaY + C * OmegaZ;
    }
    __syncthreads();
    for (int t = 0; t < T; t++)
    {
        if (j == 0 && i == 0 && t % 1 == 0.0)  printf("%i\n", t);
        //Initiate
        if (j < M)
        {
            xA0 = xA;
            yA0 = yA;
            zA0 = zA;
            xC[i * M + j] = xA + xM;
            yC[i * M + j] = yA + yM;
            zC[i * M + j] = zA + zM;
            if (j == 0)
            {
                xM0 = xM;
                yM0 = yM;
                zM0 = zM;
                uM0 = uM;
                vM0 = vM;
                wM0 = wM;
                Kx0 = Kx;
                Ky0 = Ky;
                Kz0 = Kz;
            }
        }
        //1
        grid.sync();
        if (j < M)
        {
            double ro, FLJ;
            FLJx = 0.0; FLJy = 0.0; FLJz = 0.0;
            for (int i1 = 0; i1 < N; i1++)
            {
                if (i != i1)
                {
                    for (int j1 = 0; j1 < M; j1++)
                    {
                        ro = sqrt(pow(xC[i * M + j] - xC[i1 * M + j1], 2.0) + pow(yC[i * M + j] - yC[i1 * M + j1], 2.0) + pow(zC[i * M + j] - zC[i1 * M + j1], 2.0));
                        FLJ = 24.0 * eps / ro / ro * pow(sigma / ro, 6.0) * (2.0 * pow(sigma / ro, 6.0) - 1.0);
                        FLJx += FLJ * (xC[i * M + j] - xC[i1 * M + j1]);
                        FLJy += FLJ * (yC[i * M + j] - yC[i1 * M + j1]);
                        FLJz += FLJ * (zC[i * M + j] - zC[i1 * M + j1]);
                    }
                }
            }
        }
        __syncthreads();
        if (j == 0)
        {
            uM1 = 0.0;
            vM1 = 0.0;
            wM1 = 0.0;
            for (int j1 = 0; j1 < M; j1++)
            {
                uM1 += FLJx;
                vM1 += FLJy;
                wM1 += FLJz;
            }
            uM1 /= massM;
            vM1 /= massM;
            wM1 /= massM;
        }
        __syncthreads();
        if (j < M)
        {
            xA1 = OmegaY * zA - OmegaZ * yA;
            yA1 = OmegaZ * xA - OmegaX * zA;
            zA1 = OmegaX * yA - OmegaY * xA;
        }
        __syncthreads();
        if (j == 0)
        {
            Lx1 = Ly1 = Lz1 = 0.0;
            for (int j1 = 0; j1 < M; j1++)
            {
                Lx1 += yA * FLJz - zA * FLJy;
                Ly1 += zA * FLJx - xA * FLJz;
                Lz1 += xA * FLJy - yA * FLJx;
            }
            xM1 = uM;
            yM1 = vM;
            zM1 = wM;
            //
            xM = xM0 + dt / 2.0 * xM1;
            yM = yM0 + dt / 2.0 * yM1;
            zM = zM0 + dt / 2.0 * zM1;
            uM = uM0 + dt / 2.0 * uM1;
            vM = vM0 + dt / 2.0 * vM1;
            wM = wM0 + dt / 2.0 * wM1;
        }
        __syncthreads();
        if (j < M)
        {
            xA = xA0 + dt / 2.0 * xA1;
            yA = yA0 + dt / 2.0 * yA1;
            zA = zA0 + dt / 2.0 * zA1;
            xC[i * M + j] = xA + xM;
            yC[i * M + j] = yA + yM;
            zC[i * M + j] = zA + zM;
        }
        __syncthreads();
        if (j == 0)
        {
            Kx = Kx0 + dt / 2.0 * Lx1;
            Ky = Ky0 + dt / 2.0 * Ly1;
            Kz = Kz0 + dt / 2.0 * Lz1;
            A = 0.0; B = 0.0; C = 0.0; D = 0.0; E = 0.0; F = 0.0;
            for (int j1 = 0; j1 < M; j1++)
            {
                A += massA * (pow(yA, 2) + pow(zA, 2));
                B += massA * (pow(zA, 2) + pow(xA, 2));
                C += massA * (pow(xA, 2) + pow(yA, 2));
                D -= massA * yA * zA;
                E -= massA * xA * zA;
                F -= massA * xA * yA;
            }
            double delta0 = A * B * C + E * F * D + E * F * D - B * E * E - C * F * F - A * D * D;
            double delta1 = Kx * (B * C - D * D) + Ky * (E * D - F * C) + Kz * (F * D - B * E);
            double delta2 = Kx * (E * D - F * C) + Ky * (A * C - E * E) + Kz * (F * E - A * D);
            double delta3 = Kx * (F * D - E * B) + Ky * (E * F - A * D) + Kz * (A * B - F * F);
            if (delta0 != 0.0) { OmegaX = delta1 / delta0; OmegaY = delta2 / delta0; OmegaZ = delta3 / delta0; }
            else { OmegaX = 0.0; OmegaY = 0.0; OmegaZ = 0.0; }
        }
        //2
        grid.sync();
        if (j < M)
        {
            double ro, FLJ;
            FLJx = 0.0; FLJy = 0.0; FLJz = 0.0;
            for (int i1 = 0; i1 < N; i1++)
            {
                if (i != i1)
                {
                    for (int j1 = 0; j1 < M; j1++)
                    {
                        ro = sqrt(pow(xC[i * M + j] - xC[i1 * M + j1], 2.0) + pow(yC[i * M + j] - yC[i1 * M + j1], 2.0) + pow(zC[i * M + j] - zC[i1 * M + j1], 2.0));
                        FLJ = 24.0 * eps / ro / ro * pow(sigma / ro, 6.0) * (2.0 * pow(sigma / ro, 6.0) - 1.0);
                        FLJx += FLJ * (xC[i * M + j] - xC[i1 * M + j1]);
                        FLJy += FLJ * (yC[i * M + j] - yC[i1 * M + j1]);
                        FLJz += FLJ * (zC[i * M + j] - zC[i1 * M + j1]);
                    }
                }
            }
        }
        __syncthreads();
        if (j == 0)
        {
            uM2 = 0.0;
            vM2 = 0.0;
            wM2 = 0.0;
            for (int j1 = 0; j1 < M; j1++)
            {
                uM2 += FLJx;
                vM2 += FLJy;
                wM2 += FLJz;
            }
            uM2 /= massM;
            vM2 /= massM;
            wM2 /= massM;
        }
        __syncthreads();
        if (j < M)
        {
            xA2 = OmegaY * zA - OmegaZ * yA;
            yA2 = OmegaZ * xA - OmegaX * zA;
            zA2 = OmegaX * yA - OmegaY * xA;
        }
        __syncthreads();
        if (j == 0)
        {
            Lx2 = Ly2 = Lz2 = 0.0;
            for (int j1 = 0; j1 < M; j1++)
            {
                Lx2 += yA * FLJz - zA * FLJy;
                Ly2 += zA * FLJx - xA * FLJz;
                Lz2 += xA * FLJy - yA * FLJx;
            }
            xM2 = uM;
            yM2 = vM;
            zM2 = wM;
            //
            xM = xM0 + dt / 2.0 * xM2;
            yM = yM0 + dt / 2.0 * yM2;
            zM = zM0 + dt / 2.0 * zM2;
            uM = uM0 + dt / 2.0 * uM2;
            vM = vM0 + dt / 2.0 * vM2;
            wM = wM0 + dt / 2.0 * wM2;
        }
        __syncthreads();
        if (j < M)
        {
            xA = xA0 + dt / 2.0 * xA2;
            yA = yA0 + dt / 2.0 * yA2;
            zA = zA0 + dt / 2.0 * zA2;
            xC[i * M + j] = xA + xM;
            yC[i * M + j] = yA + yM;
            zC[i * M + j] = zA + zM;
        }
        __syncthreads();
        if (j == 0)
        {
            Kx = Kx0 + dt / 2.0 * Lx2;
            Ky = Ky0 + dt / 2.0 * Ly2;
            Kz = Kz0 + dt / 2.0 * Lz2;
            A = 0.0; B = 0.0; C = 0.0; D = 0.0; E = 0.0; F = 0.0;
            for (int j1 = 0; j1 < M; j1++)
            {
                A += massA * (pow(yA, 2) + pow(zA, 2));
                B += massA * (pow(zA, 2) + pow(xA, 2));
                C += massA * (pow(xA, 2) + pow(yA, 2));
                D -= massA * yA * zA;
                E -= massA * xA * zA;
                F -= massA * xA * yA;
            }
            double delta0 = A * B * C + E * F * D + E * F * D - B * E * E - C * F * F - A * D * D;
            double delta1 = Kx * (B * C - D * D) + Ky * (E * D - F * C) + Kz * (F * D - B * E);
            double delta2 = Kx * (E * D - F * C) + Ky * (A * C - E * E) + Kz * (F * E - A * D);
            double delta3 = Kx * (F * D - E * B) + Ky * (E * F - A * D) + Kz * (A * B - F * F);
            if (delta0 != 0.0) { OmegaX = delta1 / delta0; OmegaY = delta2 / delta0; OmegaZ = delta3 / delta0; }
            else { OmegaX = 0.0; OmegaY = 0.0; OmegaZ = 0.0; }
        }
        //3
        grid.sync();
        if (j < M)
        {
            double ro, FLJ;
            FLJx = 0.0; FLJy = 0.0; FLJz = 0.0;
            for (int i1 = 0; i1 < N; i1++)
            {
                if (i != i1)
                {
                    for (int j1 = 0; j1 < M; j1++)
                    {
                        ro = sqrt(pow(xC[i * M + j] - xC[i1 * M + j1], 2.0) + pow(yC[i * M + j] - yC[i1 * M + j1], 2.0) + pow(zC[i * M + j] - zC[i1 * M + j1], 2.0));
                        FLJ = 24.0 * eps / ro / ro * pow(sigma / ro, 6.0) * (2.0 * pow(sigma / ro, 6.0) - 1.0);
                        FLJx += FLJ * (xC[i * M + j] - xC[i1 * M + j1]);
                        FLJy += FLJ * (yC[i * M + j] - yC[i1 * M + j1]);
                        FLJz += FLJ * (zC[i * M + j] - zC[i1 * M + j1]);
                    }
                }
            }
        }
        __syncthreads();
        if (j == 0)
        {
            uM3 = 0.0;
            vM3 = 0.0;
            wM3 = 0.0;
            for (int j1 = 0; j1 < M; j1++)
            {
                uM3 += FLJx;
                vM3 += FLJy;
                wM3 += FLJz;
            }
            uM3 /= massM;
            vM3 /= massM;
            wM3 /= massM;
        }
        __syncthreads();
        if (j < M)
        {
            xA3 = OmegaY * zA - OmegaZ * yA;
            yA3 = OmegaZ * xA - OmegaX * zA;
            zA3 = OmegaX * yA - OmegaY * xA;
        }
        __syncthreads();
        if (j == 0)
        {
            Lx3 = Ly3 = Lz3 = 0.0;
            for (int j1 = 0; j1 < M; j1++)
            {
                Lx3 += yA * FLJz - zA * FLJy;
                Ly3 += zA * FLJx - xA * FLJz;
                Lz3 += xA * FLJy - yA * FLJx;
            }
            xM3 = uM;
            yM3 = vM;
            zM3 = wM;
            //
            xM = xM0 + dt * xM3;
            yM = yM0 + dt * yM3;
            zM = zM0 + dt * zM3;
            uM = uM0 + dt * uM3;
            vM = vM0 + dt * vM3;
            wM = wM0 + dt * wM3;
        }
        __syncthreads();
        if (j < M)
        {
            xA = xA0 + dt * xA3;
            yA = yA0 + dt * yA3;
            zA = zA0 + dt * zA3;
            xC[i * M + j] = xA + xM;
            yC[i * M + j] = yA + yM;
            zC[i * M + j] = zA + zM;
        }
        __syncthreads();
        if (j == 0)
        {
            Kx = Kx0 + dt * Lx3;
            Ky = Ky0 + dt * Ly3;
            Kz = Kz0 + dt * Lz3;
            A = 0.0; B = 0.0; C = 0.0; D = 0.0; E = 0.0; F = 0.0;
            for (int j1 = 0; j1 < M; j1++)
            {
                A += massA * (pow(yA, 2) + pow(zA, 2));
                B += massA * (pow(zA, 2) + pow(xA, 2));
                C += massA * (pow(xA, 2) + pow(yA, 2));
                D -= massA * yA * zA;
                E -= massA * xA * zA;
                F -= massA * xA * yA;
            }
            double delta0 = A * B * C + E * F * D + E * F * D - B * E * E - C * F * F - A * D * D;
            double delta1 = Kx * (B * C - D * D) + Ky * (E * D - F * C) + Kz * (F * D - B * E);
            double delta2 = Kx * (E * D - F * C) + Ky * (A * C - E * E) + Kz * (F * E - A * D);
            double delta3 = Kx * (F * D - E * B) + Ky * (E * F - A * D) + Kz * (A * B - F * F);
            if (delta0 != 0.0) { OmegaX = delta1 / delta0; OmegaY = delta2 / delta0; OmegaZ = delta3 / delta0; }
            else { OmegaX = 0.0; OmegaY = 0.0; OmegaZ = 0.0; }
        }
        //4
        grid.sync();
        if (j < M)
        {
            double ro, FLJ;
            FLJx = 0.0; FLJy = 0.0; FLJz = 0.0;
            for (int i1 = 0; i1 < N; i1++)
            {
                if (i != i1)
                {
                    for (int j1 = 0; j1 < M; j1++)
                    {
                        ro = sqrt(pow(xC[i * M + j] - xC[i1 * M + j1], 2.0) + pow(yC[i * M + j] - yC[i1 * M + j1], 2.0) + pow(zC[i * M + j] - zC[i1 * M + j1], 2.0));
                        FLJ = 24.0 * eps / ro / ro * pow(sigma / ro, 6.0) * (2.0 * pow(sigma / ro, 6.0) - 1.0);
                        FLJx += FLJ * (xC[i * M + j] - xC[i1 * M + j1]);
                        FLJy += FLJ * (yC[i * M + j] - yC[i1 * M + j1]);
                        FLJz += FLJ * (zC[i * M + j] - zC[i1 * M + j1]);
                    }
                }
            }
        }
        __syncthreads();
        if (j == 0)
        {
            uM4 = 0.0;
            vM4 = 0.0;
            wM4 = 0.0;
            for (int j1 = 0; j1 < M; j1++)
            {
                uM4 += FLJx;
                vM4 += FLJy;
                wM4 += FLJz;
            }
            uM4 /= massM;
            vM4 /= massM;
            wM4 /= massM;
        }
        __syncthreads();
        if (j < M)
        {
            xA4 = OmegaY * zA - OmegaZ * yA;
            yA4 = OmegaZ * xA - OmegaX * zA;
            zA4 = OmegaX * yA - OmegaY * xA;
        }
        __syncthreads();
        if (j == 0)
        {
            Lx4 = Ly4 = Lz4 = 0.0;
            for (int j1 = 0; j1 < M; j1++)
            {
                Lx4 += yA * FLJz - zA * FLJy;
                Ly4 += zA * FLJx - xA * FLJz;
                Lz4 += xA * FLJy - yA * FLJx;
            }
            xM4 = uM;
            yM4 = vM;
            zM4 = wM;
            //
            xM = xM0 + dt / 6.0 * (xM1 + 2.0 * xM2 + 2.0 * xM3 + xM4);
            yM = yM0 + dt / 6.0 * (yM1 + 2.0 * yM2 + 2.0 * yM3 + yM4);
            zM = zM0 + dt / 6.0 * (zM1 + 2.0 * zM2 + 2.0 * zM3 + zM4);
            uM = uM0 + dt / 6.0 * (uM1 + 2.0 * uM2 + 2.0 * uM3 + uM4);
            vM = vM0 + dt / 6.0 * (vM1 + 2.0 * vM2 + 2.0 * vM3 + vM4);
            wM = wM0 + dt / 6.0 * (wM1 + 2.0 * wM2 + 2.0 * wM3 + wM4);
        }
        __syncthreads();
        if (j < M)
        {
            xA = xA0 + dt / 6.0 * (xA1 + 2.0 * xA2 + 2.0 * xA3 + xA4);
            yA = yA0 + dt / 6.0 * (yA1 + 2.0 * yA2 + 2.0 * yA3 + yA4);
            zA = zA0 + dt / 6.0 * (zA1 + 2.0 * zA2 + 2.0 * zA3 + zA4);
        }
        __syncthreads();
        if (j == 0)
        {
            Kx = Kx0 + dt / 6.0 * (Lx1 + 2.0 * Lx2 + 2.0 * Lx3 + Lx4);
            Ky = Ky0 + dt / 6.0 * (Ly1 + 2.0 * Ly2 + 2.0 * Ly3 + Ly4);
            Kz = Kz0 + dt / 6.0 * (Lz1 + 2.0 * Lz2 + 2.0 * Lz3 + Lz4);
            A = 0.0; B = 0.0; C = 0.0; D = 0.0; E = 0.0; F = 0.0;
            for (int j1 = 0; j1 < M; j1++)
            {
                A += massA * (pow(yA, 2) + pow(zA, 2));
                B += massA * (pow(zA, 2) + pow(xA, 2));
                C += massA * (pow(xA, 2) + pow(yA, 2));
                D -= massA * yA * zA;
                E -= massA * xA * zA;
                F -= massA * xA * yA;
            }
            double delta0 = A * B * C + E * F * D + E * F * D - B * E * E - C * F * F - A * D * D;
            double delta1 = Kx * (B * C - D * D) + Ky * (E * D - F * C) + Kz * (F * D - B * E);
            double delta2 = Kx * (E * D - F * C) + Ky * (A * C - E * E) + Kz * (F * E - A * D);
            double delta3 = Kx * (F * D - E * B) + Ky * (E * F - A * D) + Kz * (A * B - F * F);
            if (delta0 != 0.0) { OmegaX = delta1 / delta0; OmegaY = delta2 / delta0; OmegaZ = delta3 / delta0; }
            else { OmegaX = 0.0; OmegaY = 0.0; OmegaZ = 0.0; }
        }
        grid.sync();
    }
}

int main()
{
    //initiate
    const int T = 10000000; const double dt = pow(10, -6.0);

    const double KB = 1.38 * pow(10.0, -23);
    const double massA = 1.9944733 * pow(10.0, -26); const double massM = massA * M, eps = 12.5 * KB, sigma = 0.34;
    //error
    cudaError_t err;
    //device allocate
    //Atom
    double* d_xA, * d_yA, * d_zA;
    //Molecule
    double* d_xM, * d_yM, * d_zM, * d_uM, * d_vM, * d_wM, * d_OmegaX, * d_OmegaY, * d_OmegaZ;

    //device allocate
    size_t sizeM = N * sizeof(double), sizeA = N * M * sizeof(double);
    //Atom
    cudaMalloc((void**)&d_xA, sizeA); cudaMalloc((void**)&d_yA, sizeA); cudaMalloc((void**)&d_zA, sizeA);
    //Molecule
    cudaMalloc((void**)&d_xM, sizeM); cudaMalloc((void**)&d_yM, sizeM); cudaMalloc((void**)&d_zM, sizeM);
    cudaMalloc((void**)&d_uM, sizeM); cudaMalloc((void**)&d_vM, sizeM); cudaMalloc((void**)&d_wM, sizeM);
    cudaMalloc((void**)&d_OmegaX, sizeM); cudaMalloc((void**)&d_OmegaY, sizeM); cudaMalloc((void**)&d_OmegaZ, sizeM);

    //host allocate
    //Molecule
    std::string h_name[N];
    double h_xM[N], h_yM[N], h_zM[N];
    double h_uM[N], h_vM[N], h_wM[N];
    double h_OmegaX[N], h_OmegaY[N], h_OmegaZ[N];
    h_name[0] = "First";
    h_xM[0] = -2.5; h_yM[0] = -1.0; h_zM[0] = 0.0; h_uM[0] = 12.0; h_vM[0] = 0.0; h_wM[0] = 0.0; h_OmegaX[0] = 0.0; h_OmegaY[0] = 0.0; h_OmegaZ[0] = 0.0;
    h_name[1] = "Second";
    h_xM[1] = 2.5; h_yM[1] = 1.0; h_zM[1] = 0.0;  h_uM[1] = -12.0; h_vM[1] = 0.0; h_wM[1] = 0.0; h_OmegaX[1] = 0.0; h_OmegaY[1] = 0.0; h_OmegaZ[1] = 0.0;

    //Atom
    double h_xA[N * M], h_yA[N * M], h_zA[N * M];                                                    
    readData("Nanotube.txt", h_xA, h_yA, h_zA);
    Average(h_xA, h_yA, h_zA);

    //host to device copy
    //Atom
    cudaMemcpy(d_xA, h_xA, sizeA, cudaMemcpyHostToDevice); cudaMemcpy(d_yA, h_yA, sizeA, cudaMemcpyHostToDevice); cudaMemcpy(d_zA, h_zA, sizeA, cudaMemcpyHostToDevice);
    //Molecule
    cudaMemcpy(d_xM, h_xM, sizeM, cudaMemcpyHostToDevice); cudaMemcpy(d_yM, h_yM, sizeM, cudaMemcpyHostToDevice); cudaMemcpy(d_zM, h_zM, sizeM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_uM, h_uM, sizeM, cudaMemcpyHostToDevice); cudaMemcpy(d_vM, h_vM, sizeM, cudaMemcpyHostToDevice); cudaMemcpy(d_wM, h_wM, sizeM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OmegaX, h_OmegaX, sizeM, cudaMemcpyHostToDevice); cudaMemcpy(d_OmegaY, h_OmegaY, sizeM, cudaMemcpyHostToDevice); cudaMemcpy(d_OmegaZ, h_OmegaZ, sizeM, cudaMemcpyHostToDevice);

    //Result Data Preparing
    double* h_xData, * h_yData, * h_zData;
    double* d_xData, * d_yData, * d_zData;
    size_t sizeD = (N * T / 100 + 1) * sizeof(double);
    h_xData = (double*)malloc(sizeD); h_yData = (double*)malloc(sizeD); h_zData = (double*)malloc(sizeD);
    cudaMalloc((void**)&d_xData, sizeD); cudaMalloc((void**)&d_yData, sizeD); cudaMalloc((void**)&d_zData, sizeD);

    //Solver
    int countBlock = N, countThread = (M / 32 + 1) * 32;
    Solver<<<countBlock, countThread>>>(T, dt, KB, massA, massM, sigma, eps,
                                        d_xA, d_yA, d_zA, d_xData, d_yData, d_zData,
                                        d_xM, d_yM, d_zM, d_uM, d_vM, d_wM,
                                        d_OmegaX, d_OmegaY, d_OmegaZ);
    cudaDeviceSynchronize();
    std::cout << "vse";

    //Results
    err = cudaMemcpy(h_xData, d_xData, sizeD, cudaMemcpyDeviceToHost); cudaMemcpy(h_yData, d_yData, sizeD, cudaMemcpyDeviceToHost); cudaMemcpy(h_zData, d_zData, sizeD, cudaMemcpyDeviceToHost);
    cudaFree(d_xA); cudaFree(d_yA); cudaFree(d_zA);
    cudaFree(d_xM); cudaFree(d_yM); cudaFree(d_zM);
    cudaFree(d_uM); cudaFree(d_vM); cudaFree(d_wM);
    cudaFree(d_OmegaX); cudaFree(d_OmegaY); cudaFree(d_OmegaZ);
    cudaFree(d_xData); cudaFree(d_yData); cudaFree(d_zData);
    if (cudaSuccess != err) {
        std::cout << "error";
    }
    return 0;
}