#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <math.h>
#include <vector>
#include <iostream>
#include "cuda_runtime.h"

class Molecule
{
private:
    std::string name, nameStructure;
    float xM = 0.f, yM = 0.f, zM = 0.f;
    float uM = 0.f, vM = 0.f, wM = 0.f;
    float omegaX = 0.f, omegaY = 0.f, omegaZ = 0.f;
    float Kx, Ky, Kz;
    float Kx0, Ky0, Kz0;
    float Kx1, Ky1, Kz1;
    float Lx, Ly, Lz;
    float xM0, yM0, zM0;
    float xM1, yM1, zM1;
    float uM0, vM0, wM0;
    float uM1, vM1, wM1;
    float ub, vb, wb;
    cudaStream_t CudaStream;      //CUDA Stream
    cudaError_t err;
    int M;                        //count of Atom
    int n;                        //number of Molecule
    int N;
    int countBlock, countThread;
    size_t sizeM;                 //size of Atoms arrays
    float* h_xA, * h_yA, * h_zA;  //host Atom pointer
    float* d_xA, * d_yA, * d_zA;  //device Atom pointer
    float* d_uA, * d_vA, * d_wA;
    float* d_xA0, * d_yA0, * d_zA0;
    float* d_xA1, * d_yA1, * d_zA1;
    float* d_xFLJ, * d_yFLJ, * d_zFLJ;
    float* h_ABCDEF, * d_ABCDEF;                //output Atom Data
    float* h_UVW, * d_UVW;
    float* h_Lxyz, * d_Lxyz;
    std::vector<float> xData = std::vector<float>(), yData = std::vector<float>(), zData = std::vector<float>();
    void coefTensor();
    void angleVelocity();
    void kineticMoment();
    void readData();
    void Average();
public:
    Molecule(std::string name, std::string nameStructure, int N, int n);
    ~Molecule();
    int countAtom();
    void initSpatial(float xM, float yM, float zM);
    void initVelocity(float uM, float vM, float wM);
    void initRotate(float omegaX, float omegaY, float omegaZ);
    void prepareStep(float* d_xC, float* d_yC, float* d_zC);
    void RungeKuttaStep(float dt);
    void moleculeStep(float dt, float* d_xC, float* d_yC, float* d_zC);
    void prologueStep();
    void epilogueStep();
    void UVW(float* d_xC, float* d_yC, float* d_zC); //+velocity Step
    void writeData();
    void writeDataToFile(std::string sData);
};
