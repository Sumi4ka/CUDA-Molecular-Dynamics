#include "Molecule.cuh"
#include "Kernels.cuh"
float h_massM = 4.7867359e-24; //h_massA * M,
void Molecule::coefTensor()
{
    atomCoofTensor << <1, countThread, 0, CudaStream >> > (d_ABCDEF, d_xA, d_yA, d_zA, M);
    cudaMemcpyAsync(h_ABCDEF, d_ABCDEF, 6 * sizeof(float), cudaMemcpyDeviceToHost, CudaStream);
    cudaDeviceSynchronize();
    //std::cout <<"A: " << h_ABCDEF[0] << ", B: " << h_ABCDEF[1] << ", C: " << h_ABCDEF[2] << ", D: " << h_ABCDEF[3] << ", E: " << h_ABCDEF[4] << ", F: " << h_ABCDEF[5] << std::endl;
}
void Molecule::angleVelocity()
{
    double A = (double)h_ABCDEF[0];
    double B = (double)h_ABCDEF[1];
    double C = (double)h_ABCDEF[2];
    double D = (double)h_ABCDEF[3];
    double E = (double)h_ABCDEF[4];
    double F = (double)h_ABCDEF[5];
    double OmegaX, OmegaY, OmegaZ;
    double delta0 = A * B * C + E * F * D + E * F * D - B * E * E - C * F * F - A * D * D;
    if (delta0 != 0.0)
    {
        double delta1 = Kx * (B * C - D * D) + Ky * (E * D - F * C) + Kz * (F * D - B * E);
        double delta2 = Kx * (E * D - F * C) + Ky * (A * C - E * E) + Kz * (F * E - A * D);
        double delta3 = Kx * (F * D - E * B) + Ky * (E * F - A * D) + Kz * (A * B - F * F);
        OmegaX = delta1 / delta0; OmegaY = delta2 / delta0; OmegaZ = delta3 / delta0;
    }
    else { OmegaX = 0.0; OmegaY = 0.0; OmegaZ = 0.0; }
    omegaX = OmegaX;
    omegaY = OmegaY;
    omegaZ = OmegaZ;
}
void Molecule::kineticMoment()
{
    Kx = h_ABCDEF[0] * omegaX + h_ABCDEF[5] * omegaY + h_ABCDEF[4] * omegaZ;
    Kz = h_ABCDEF[4] * omegaX + h_ABCDEF[3] * omegaY + h_ABCDEF[2] * omegaZ;
    Ky = h_ABCDEF[5] * omegaX + h_ABCDEF[1] * omegaY + h_ABCDEF[3] * omegaZ;
}
void Molecule::readData()
{
    std::string sData = "Data\\";
    std::ifstream in(sData + nameStructure + ".txt");
    std::string s1, xData, yData, zData;
    int is1;
    in >> s1;
    if (s1 == "Carbon")
    {
        in >> s1;
        M = stoi(s1);
        cudaError_t err;
        err = cudaHostAlloc((void**)&h_xA, M * sizeof(float), cudaHostAllocDefault);
        err = cudaHostAlloc((void**)&h_yA, M * sizeof(float), cudaHostAllocDefault);
        err = cudaHostAlloc((void**)&h_zA, M * sizeof(float), cudaHostAllocDefault);
        for (int j = 0; j < M; j++)
        {
            in >> xData >> yData >> zData;
            h_xA[j] = stof(xData);
            h_yA[j] = stof(yData);
            h_zA[j] = stof(zData);
        }
    }
    Average();
    in.close();
}
void Molecule::Average()
{
    float avgX, avgY, avgZ;
    avgX = avgY = avgZ = 0.0f;
    for (int j = 0; j < M; j++)
    {
        avgX += h_xA[j];
        avgY += h_yA[j];
        avgZ += h_zA[j];
    }
    avgX /= M; avgY /= M; avgZ /= M;
    for (int j = 0; j < M; j++)
    {
        h_xA[j] -= avgX;
        h_yA[j] -= avgY;
        h_zA[j] -= avgZ;
    }
}
Molecule::Molecule(std::string name, std::string nameStructure, int N, int n)
{
    err = cudaStreamCreate(&CudaStream);
    this->name = name; this->nameStructure = nameStructure;
    this->n = n;
    this->N = N;
    readData();              //prepare Data to calculate
    sizeM = M * sizeof(float);
    countBlock = N, countThread = (M / 32 + 1) * 32;
    //Device allocate
    err = cudaMalloc((void**)&d_xA, sizeM); err = cudaMalloc((void**)&d_yA, sizeM); err = cudaMalloc((void**)&d_zA, sizeM);
    err = cudaMalloc((void**)&d_uA, sizeM); err = cudaMalloc((void**)&d_vA, sizeM); err = cudaMalloc((void**)&d_wA, sizeM);
    err = cudaMalloc((void**)&d_xA0, sizeM); err = cudaMalloc((void**)&d_yA0, sizeM); err = cudaMalloc((void**)&d_zA0, sizeM);
    err = cudaMalloc((void**)&d_xA1, sizeM); err = cudaMalloc((void**)&d_yA1, sizeM); err = cudaMalloc((void**)&d_zA1, sizeM);
    err = cudaMalloc((void**)&d_xFLJ, sizeM); err = cudaMalloc((void**)&d_yFLJ, sizeM); err = cudaMalloc((void**)&d_zFLJ, sizeM);

    //Host to device copy
    err = cudaMemcpyAsync(d_xA, h_xA, sizeM, cudaMemcpyHostToDevice, CudaStream);
    err = cudaMemcpyAsync(d_yA, h_yA, sizeM, cudaMemcpyHostToDevice, CudaStream);
    err = cudaMemcpyAsync(d_zA, h_zA, sizeM, cudaMemcpyHostToDevice, CudaStream);
    //OutputData
    err = cudaHostAlloc((void**)&h_ABCDEF, 6 * sizeof(float), cudaHostAllocDefault);
    err = cudaMalloc((void**)&d_ABCDEF, 6 * sizeof(float));
    err = cudaHostAlloc((void**)&h_UVW, 3 * sizeof(float), cudaHostAllocDefault);
    err = cudaMalloc((void**)&d_UVW, 3 * sizeof(float));
    err = cudaHostAlloc((void**)&h_Lxyz, 3 * sizeof(float), cudaHostAllocDefault);
    err = cudaMalloc((void**)&d_Lxyz, 3 * sizeof(float));
    cudaDeviceSynchronize();
}
Molecule::~Molecule()
{
    cudaStreamDestroy(CudaStream);
    cudaFree(d_xA); cudaFree(d_yA); cudaFree(d_zA);
    cudaFree(d_uA); cudaFree(d_vA); cudaFree(d_wA);
    cudaFree(d_xA0); cudaFree(d_yA0); cudaFree(d_zA0);
    cudaFree(d_xA1); cudaFree(d_yA1); cudaFree(d_zA1);
    cudaFree(d_xFLJ); cudaFree(d_yFLJ); cudaFree(d_zFLJ);
    cudaFreeHost(h_ABCDEF); cudaFree(d_ABCDEF);
    cudaFreeHost(h_UVW); cudaFree(d_UVW);
    cudaFreeHost(h_Lxyz); cudaFree(d_Lxyz);
}
int Molecule::countAtom() { return M; }
void Molecule::initSpatial(float xM, float yM, float zM)
{
    this->xM = xM;
    this->yM = yM;
    this->zM = zM;
}
void Molecule::initVelocity(float uM, float vM, float wM)
{
    this->uM = uM;
    this->vM = vM;
    this->wM = wM;
}
void Molecule::initRotate(float omegaX, float omegaY, float omegaZ)
{
    this->omegaX = omegaX;
    this->omegaY = omegaY;
    this->omegaZ = omegaZ;
}
void Molecule::prepareStep(float* d_xC, float* d_yC, float* d_zC)
{
    xM1 = xM0 = xM;
    yM1 = yM0 = yM;
    zM1 = zM0 = zM;
    uM1 = uM0 = uM;
    vM1 = vM0 = vM;
    wM1 = wM0 = wM;
    Kx1 = Kx0 = Kx;
    Ky1 = Ky0 = Ky;
    Kz1 = Kz0 = Kz;
    atomPrepareStepD << <1, countThread, 0, CudaStream >> > (d_xA, d_yA, d_zA, d_xA0, d_yA0, d_zA0, d_xA1, d_yA1, d_zA1, d_xC, d_yC, d_zC, xM, yM, zM, M, n);
    cudaDeviceSynchronize();
}
void Molecule::RungeKuttaStep(float dt)
{
    xM1 += uM * dt;
    yM1 += vM * dt;
    zM1 += wM * dt;
    uM1 += ub * dt;
    vM1 += vb * dt;
    wM1 += wb * dt;
    Kx1 += Lx * dt;
    Ky1 += Ly * dt;
    Kz1 += Lz * dt;
    atomVelocity<<<1, countThread, 0, CudaStream>>>(d_xA, d_yA, d_zA, d_xA1, d_yA1, d_zA1, d_uA, d_vA, d_wA, omegaX, omegaY, omegaZ, M, dt);
    cudaDeviceSynchronize();
}
void Molecule::moleculeStep(float dt, float* d_xC, float* d_yC, float* d_zC)
{
    xM = xM0 + dt * uM;
    yM = yM0 + dt * vM;
    zM = zM0 + dt * wM;
    uM = uM0 + dt * ub;
    vM = vM0 + dt * vb;
    wM = wM0 + dt * wb;
    Kx = Kx0 + dt * Lx;
    Ky = Ky0 + dt * Ly;
    Kz = Kz0 + dt * Lz;
    atomStep << <1, countThread, 0, CudaStream >> > (d_xA, d_yA, d_zA, d_xA0, d_yA0, d_zA0, d_xC, d_yC, d_zC, d_uA, d_vA, d_wA, xM, yM, zM, n, M, dt);
    cudaDeviceSynchronize();
    coefTensor();
    angleVelocity();
}
void Molecule::prologueStep()
{
    coefTensor();
    kineticMoment();
}
void Molecule::epilogueStep()
{
    coefTensor();
    angleVelocity();
}
void Molecule::UVW(float* d_xC, float* d_yC, float* d_zC) //+velocity Step
{
    atomUVW << <1, countThread, countThread * sizeof(float), CudaStream >> > (d_xC, d_yC, d_zC, d_UVW, d_Lxyz, N, M, n);  //ѕока что считаем что атомов в молекулах одинаково sm;
    err = cudaMemcpyAsync(h_UVW, d_UVW, 3 * sizeof(float), cudaMemcpyDeviceToHost, CudaStream);
    cudaDeviceSynchronize();
    ub = h_UVW[0] / h_massM; vb = h_UVW[1] / h_massM; wb = h_UVW[2] / h_massM;
    //std::cout << "ub: " << ub << ", vb: " << vb << ", wb: " << wb << std::endl;
    err = cudaMemcpyAsync(h_Lxyz, d_Lxyz, 3 * sizeof(float), cudaMemcpyDeviceToHost, CudaStream);
    cudaDeviceSynchronize();
    Lx = h_Lxyz[0]; Ly = h_Lxyz[1]; Lz = h_Lxyz[2];
    //std::cout << "Lx: " << Lx << ", Ly: " << Ly << ", Lz: " << Lz << std::endl;
}
void Molecule::writeData()
{
    xData.push_back(xM);
    yData.push_back(yM);
    zData.push_back(zM);
}
void Molecule::writeDataToFile(std::string sData)
{
    std::ofstream out(sData + "\\" + name + ".txt");
    for (int i = 0; i < xData.size(); i++)
        out << xData[i] << ' ' << yData[i] << ' ' << zData[i] << std::endl;
    out.close();
}