#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <math.h>
#include <vector>
#include <iostream>
#include "cuda_runtime.h"

const double massM = 4.7867359e-24; //massA * M,

class Molecule
{
protected:
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
    int M;                        //count of Atom
    int n;                        //number of Molecule
    int N;                        //count of Molecules in Solver
    size_t sizeM;                 //size of Atoms arrays
    float* xA, * yA, * zA;  //host Atom pointer
    float* xC, * yC, * zC;
    float* ABCDEF, * UVW, * Lxyz;
    std::vector<float> xData = std::vector<float>(), yData = std::vector<float>(), zData = std::vector<float>();
    virtual void coefTensor() = 0;
    void angleVelocity();
    void kineticMoment();
    void readData();
    void Average();
public:
    Molecule(std::string name, std::string nameStructure, int N, int n);
    virtual ~Molecule();
    void getC(float* xC, float* yC, float* zC);
    int countAtom();
    void initSpatial(float xM, float yM, float zM);
    void initVelocity(float uM, float vM, float wM);
    void initRotate(float omegaX, float omegaY, float omegaZ);
    virtual void prepareStep();
    virtual void RungeKuttaStep(float dt);
    virtual void moleculeStep(float dt);
    void prologueStep();
    void epilogueStep();
    virtual void calculateFLJ() = 0; //UVW+velocity Step
    void writeData();
    void writeDataToFile(std::string sData);
    void Description();
};
class MoleculeGPU : public Molecule
{
private:
    int countThread = 256;
    float* d_xA, * d_yA, * d_zA;  //device Atom pointer
    float* d_uA, * d_vA, * d_wA;
    float* d_xA0, * d_yA0, * d_zA0;
    float* d_xA1, * d_yA1, * d_zA1;
    float* d_xFLJ, * d_yFLJ, * d_zFLJ;
    float* d_ABCDEF, * d_UVW, * d_Lxyz;
    cudaStream_t CudaStream;      //CUDA Stream
    cudaError_t err;
public:
    MoleculeGPU(std::string name, std::string nameStructure, int N, int n);
    ~MoleculeGPU() override;
    void coefTensor() override;
    void prepareStep() override;
    void RungeKuttaStep(float dt) override;
    void moleculeStep(float dt) override;
    void calculateFLJ() override; //+velocity Step
};
class MoleculeCPU : public Molecule
{
private:
    float* uA, * vA, * wA;
    float* xA0, * yA0, * zA0;
    float* xA1, * yA1, * zA1;
    float* xFLJ, * yFLJ, * zFLJ;
public:
    MoleculeCPU(std::string name, std::string nameStructure, int N, int n);
    ~MoleculeCPU() override;
    void coefTensor() override;
    void prepareStep() override;
    void RungeKuttaStep(float dt) override;
    void moleculeStep(float dt) override;
    void calculateFLJ() override; //+velocity Step
};
