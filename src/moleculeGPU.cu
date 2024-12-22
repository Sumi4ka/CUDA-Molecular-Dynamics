#include "molecule.cuh"
#include "kernels.cuh"

void MoleculeGPU::coefTensor()
{
    atomCoofTensor << <1, countThread, 0, CudaStream >> > (d_ABCDEF, d_xA, d_yA, d_zA, M);
    cudaMemcpyAsync(ABCDEF, d_ABCDEF, 6 * sizeof(float), cudaMemcpyDeviceToHost, CudaStream);
    cudaDeviceSynchronize();
    //std::cout <<"A: " << h_ABCDEF[0] << ", B: " << h_ABCDEF[1] << ", C: " << h_ABCDEF[2] << ", D: " << h_ABCDEF[3] << ", E: " << h_ABCDEF[4] << ", F: " << h_ABCDEF[5] << std::endl;
}
MoleculeGPU::MoleculeGPU(std::string name, std::string nameStructure, int N, int n) : Molecule(name, nameStructure, N, n)
{
    err = cudaStreamCreate(&CudaStream);
    countThread = (M / 32 + 1) * 32;
    //Device allocate
    err = cudaMalloc((void**)&d_xA, sizeM); err = cudaMalloc((void**)&d_yA, sizeM); err = cudaMalloc((void**)&d_zA, sizeM);
    err = cudaMalloc((void**)&d_uA, sizeM); err = cudaMalloc((void**)&d_vA, sizeM); err = cudaMalloc((void**)&d_wA, sizeM);
    err = cudaMalloc((void**)&d_xA0, sizeM); err = cudaMalloc((void**)&d_yA0, sizeM); err = cudaMalloc((void**)&d_zA0, sizeM);
    err = cudaMalloc((void**)&d_xA1, sizeM); err = cudaMalloc((void**)&d_yA1, sizeM); err = cudaMalloc((void**)&d_zA1, sizeM);
    err = cudaMalloc((void**)&d_xFLJ, sizeM); err = cudaMalloc((void**)&d_yFLJ, sizeM); err = cudaMalloc((void**)&d_zFLJ, sizeM);

    //Host to device copy
    err = cudaMemcpyAsync(d_xA, xA, sizeM, cudaMemcpyHostToDevice, CudaStream);
    err = cudaMemcpyAsync(d_yA, yA, sizeM, cudaMemcpyHostToDevice, CudaStream);
    err = cudaMemcpyAsync(d_zA, zA, sizeM, cudaMemcpyHostToDevice, CudaStream);
    //OutputData
    err = cudaHostAlloc((void**)&ABCDEF, 6 * sizeof(float), cudaHostAllocDefault);
    err = cudaMalloc((void**)&d_ABCDEF, 6 * sizeof(float));
    err = cudaHostAlloc((void**)&UVW, 3 * sizeof(float), cudaHostAllocDefault);
    err = cudaMalloc((void**)&d_UVW, 3 * sizeof(float));
    err = cudaHostAlloc((void**)&Lxyz, 3 * sizeof(float), cudaHostAllocDefault);
    err = cudaMalloc((void**)&d_Lxyz, 3 * sizeof(float));
    cudaDeviceSynchronize();
}
MoleculeGPU::~MoleculeGPU()
{
    cudaStreamDestroy(CudaStream);
    cudaFree(d_xA); cudaFree(d_yA); cudaFree(d_zA);
    cudaFree(d_uA); cudaFree(d_vA); cudaFree(d_wA);
    cudaFree(d_xA0); cudaFree(d_yA0); cudaFree(d_zA0);
    cudaFree(d_xA1); cudaFree(d_yA1); cudaFree(d_zA1);
    cudaFree(d_xFLJ); cudaFree(d_yFLJ); cudaFree(d_zFLJ);
    cudaFreeHost(ABCDEF); cudaFree(d_ABCDEF);
    cudaFreeHost(UVW); cudaFree(d_UVW);
    cudaFreeHost(Lxyz); cudaFree(d_Lxyz);
}
void MoleculeGPU::prepareStep()
{
    Molecule::prepareStep();
    atomPrepareStepD << <1, countThread, 0, CudaStream >> > (d_xA, d_yA, d_zA, d_xA0, d_yA0, d_zA0, d_xA1, d_yA1, d_zA1, xC, yC, zC, xM, yM, zM, M, n);
    cudaDeviceSynchronize();
}
void MoleculeGPU::RungeKuttaStep(float dt)
{
    Molecule::RungeKuttaStep(dt);
    atomVelocity << <1, countThread, 0, CudaStream >> > (d_xA, d_yA, d_zA, d_xA1, d_yA1, d_zA1, d_uA, d_vA, d_wA, omegaX, omegaY, omegaZ, M, dt);
    cudaDeviceSynchronize();
}
void MoleculeGPU::moleculeStep(float dt)
{
    Molecule::moleculeStep(dt);
    atomStep << <1, countThread, 0, CudaStream >> > (d_xA, d_yA, d_zA, d_xA0, d_yA0, d_zA0, xC, yC, zC, d_uA, d_vA, d_wA, xM, yM, zM, n, M, dt);
    cudaDeviceSynchronize();
    coefTensor();
    angleVelocity();
}
void MoleculeGPU::calculateFLJ() //+velocity Step
{
    atomCalculateFLJ << <1, countThread, countThread * sizeof(float), CudaStream >> > (xC, yC, zC, d_UVW, d_Lxyz, N, M, n);  //ѕока что считаем что атомов в молекулах одинаково sm;
    err = cudaMemcpyAsync(UVW, d_UVW, 3 * sizeof(float), cudaMemcpyDeviceToHost, CudaStream);
    cudaDeviceSynchronize();
    ub = UVW[0] / massM; vb = UVW[1] / massM; wb = UVW[2] / massM;
    //std::cout << "ub: " << ub << ", vb: " << vb << ", wb: " << wb << std::endl;
    err = cudaMemcpyAsync(Lxyz, d_Lxyz, 3 * sizeof(float), cudaMemcpyDeviceToHost, CudaStream);
    cudaDeviceSynchronize();
    Lx = Lxyz[0]; Ly = Lxyz[1]; Lz = Lxyz[2];
    //std::cout << "Lx: " << Lx << ", Ly: " << Ly << ", Lz: " << Lz << std::endl;
}