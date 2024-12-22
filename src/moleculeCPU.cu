#include "molecule.cuh"

const float KB = 1.380649e-23, massA = 1.9944733e-26, sigma = 0.34;
const float eps = 1.7258113e-22; //h_KB * 12.5
void MoleculeCPU::coefTensor()
{
    ABCDEF[0] = 0.0f;
    ABCDEF[1] = 0.0f; 
    ABCDEF[2] = 0.0f; 
    ABCDEF[3] = 0.0f; 
    ABCDEF[4] = 0.0f; 
    ABCDEF[5] = 0.0f; 
    for (int j = 0; j < M; j++)
    {
        ABCDEF[0] += massA * (yA[j] * yA[j] + zA[j] * zA[j]);
        ABCDEF[1] += massA * (xA[j] * xA[j] + zA[j] * zA[j]);
        ABCDEF[2] += massA * (xA[j] * xA[j] + yA[j] * yA[j]);
        ABCDEF[3] -= massA * yA[j] * zA[j];
        ABCDEF[4] -= massA * xA[j] * zA[j];
        ABCDEF[5] -= massA * xA[j] * yA[j];
    }
    //std::cout <<"A: " << ABCDEF[0] << ", B: " << ABCDEF[1] << ", C: " << ABCDEF[2] << ", D: " << ABCDEF[3] << ", E: " << ABCDEF[4] << ", F: " << ABCDEF[5] << std::endl;
}
MoleculeCPU::MoleculeCPU(std::string name, std::string nameStructure, int N, int n) : Molecule(name, nameStructure, N, n)
{
    //Allocate
    uA = (float*)malloc(sizeM); vA = (float*)malloc(sizeM); wA = (float*)malloc(sizeM);
    xA0 = (float*)malloc(sizeM); yA0 = (float*)malloc(sizeM); zA0 = (float*)malloc(sizeM);
    xA1 = (float*)malloc(sizeM); yA1 = (float*)malloc(sizeM); zA1 = (float*)malloc(sizeM);
    xFLJ = (float*)malloc(sizeM); yFLJ = (float*)malloc(sizeM); zFLJ = (float*)malloc(sizeM);
    //OutputData
    ABCDEF = (float*)malloc(6 * sizeof(float));
    UVW = (float*)malloc(3 * sizeof(float));
    Lxyz = (float*)malloc(3 * sizeof(float));
    cudaDeviceSynchronize();
}
MoleculeCPU::~MoleculeCPU()
{
    free(uA); free(vA); free(wA);
    free(xA0); free(yA0); free(zA0);
    free(xA1); free(yA1); free(zA1);
    free(xFLJ); free(yFLJ); free(zFLJ);
    free(ABCDEF);
    free(UVW);
    free(Lxyz);
}
void MoleculeCPU::prepareStep()
{
    Molecule::prepareStep();
    for (int j = 0; j < M; j++)
    {
        xA1[j] = xA0[j] = xA[j];
        yA1[j] = yA0[j] = yA[j];
        zA1[j] = zA0[j] = zA[j];
        xC[n * M + j] = xA[j] + xM;
        yC[n * M + j] = yA[j] + yM;
        zC[n * M + j] = zA[j] + zM;
    }
    cudaDeviceSynchronize();
}
void MoleculeCPU::RungeKuttaStep(float dt)
{
    Molecule::RungeKuttaStep(dt);
    for (int j = 0; j < M; j++)
    {
        uA[j] = omegaY * zA[j] - omegaZ * yA[j];
        vA[j] = omegaZ * xA[j] - omegaX * zA[j];
        wA[j] = omegaX * yA[j] - omegaY * xA[j];
        xA1[j] += dt * uA[j];
        yA1[j] += dt * vA[j];
        zA1[j] += dt * wA[j];
    }
    cudaDeviceSynchronize();
}
void MoleculeCPU::moleculeStep(float dt)
{
    Molecule::moleculeStep(dt);
    for (int j = 0; j < M; j++)
    {
        xA[j] = xA0[j] + dt * uA[j];
        yA[j] = yA0[j] + dt * vA[j];
        zA[j] = zA0[j] + dt * wA[j];
        xC[n * M + j] = xA[j] + xM;
        yC[n * M + j] = yA[j] + yM;
        zC[n * M + j] = zA[j] + zM;
    }
    coefTensor();
    angleVelocity();
}
void MoleculeCPU::calculateFLJ() //+velocity Step
{
    float ro, c, FLJ;
    for (int j = 0; j < M; j++)
    {
        float x = xC[n * M + j], y = yC[n * M + j], z = zC[n * M + j];
        xFLJ[j] = 0.f; yFLJ[j] = 0.f; zFLJ[j] = 0.f;
        for (int i = 0; i < N; i++)
        {
            if (i != n)
            {
                for (int k = 0; k < M; k++)                              //Цикл потоков (нитей)
                {
                    //if ((k + j * M) != i)
                    {
                        ro = sqrtf(powf(x - xC[i * M + k], 2.f) + powf(y - yC[i * M + k], 2.f) + powf(z - zC[i * M + k], 2.f));
                        c = powf(sigma / ro, 6.f);
                        FLJ = 24.f * eps * c * (2.f * c - 1.f) / ro / ro;
                        xFLJ[j] += FLJ * (x - xC[i * M + k]);
                        yFLJ[j] += FLJ * (y - yC[i * M + k]);
                        zFLJ[j] += FLJ * (z - zC[i * M + k]);
                    }
                }
            }
        }
    }
    UVW[0] = 0.f; UVW[1] = 0.f; UVW[2] = 0.f;
    Lxyz[0] = 0.f; Lxyz[1] = 0.f; Lxyz[2] = 0.f;
    for (int j = 0; j < M; j++)
    {
        UVW[0] += xFLJ[j];
        UVW[1] += yFLJ[j];
        UVW[2] += zFLJ[j];
        Lxyz[0] += yC[n * M + j] * zFLJ[j] - zC[n * M + j] * yFLJ[j];
        Lxyz[1] += zC[n * M + j] * xFLJ[j] - xC[n * M + j] * zFLJ[j];
        Lxyz[2] += xC[n * M + j] * yFLJ[j] - yC[n * M + j] * xFLJ[j];
    }
    ub = UVW[0] / massM; vb = UVW[1] / massM; wb = UVW[2] / massM;
    //std::cout << "ub: " << ub << ", vb: " << vb << ", wb: " << wb << std::endl;
    Lx = Lxyz[0]; Ly = Lxyz[1]; Lz = Lxyz[2];
    //std::cout << "Lx: " << Lx << ", Ly: " << Ly << ", Lz: " << Lz << std::endl;
}