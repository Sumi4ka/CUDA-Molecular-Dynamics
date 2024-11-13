#include "Solver.cuh"
Solver::Solver(std::string Description)
{
    //Description input
    molecules = std::vector<Molecule*>();
    std::ifstream is(Description);
    std::string inString, name, nameStructure, x, y, z;
    std::getline(is, inString);
    is >> inString;
    is >> inString;
    N = stoi(inString);
    is >> inString;
    is >> inString;
    NT = stoi(inString);
    is >> inString;
    is >> inString;
    dt = stof(inString);
    M = 0;
    std::getline(is, inString);
    //Molecules input
    for (int i = 0; i < N; i++)
    {
        std::getline(is, inString);
        is >> inString;
        is >> inString;
        name = inString;
        is >> inString;
        is >> inString;
        nameStructure = inString;
        molecules.push_back(new Molecule(name, nameStructure, N, i));
        is >> inString;
        is >> x >> y >> z;
        molecules[i]->initSpatial(stof(x), stof(y), stof(z));
        is >> inString;
        is >> x >> y >> z;
        molecules[i]->initVelocity(stof(x), stof(y), stof(z));
        is >> inString;
        is >> x >> y >> z;
        molecules[i]->initRotate(stof(x), stof(y), stof(z));
        M += molecules[i]->countAtom();
    }
    //Preparing
    cudaError_t err;
    size_t sizeM = M * sizeof(float);
    err = cudaMalloc((void**)&d_xC, sizeM);
    err = cudaMalloc((void**)&d_yC, sizeM);
    err = cudaMalloc((void**)&d_zC, sizeM);
    is.close();
}
Solver::~Solver()
{
    for (int i = 0; i < N; i++)
        delete molecules[i];
    cudaFree(d_xC);
    cudaFree(d_yC);
    cudaFree(d_zC);
}
void Solver::Solve()
{
    printf("Calculation begin...\n");
    //Solver
    for (int i = 0; i < N; i++)
    {
        molecules[i]->prologueStep();
    }
    auto start = std::chrono::high_resolution_clock::now(), stop = std::chrono::high_resolution_clock::now();
    auto startCalculation = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < NT; t++)
    {
        if (!(t % 1000)) start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++)
            molecules[i]->prepareStep(d_xC, d_yC, d_zC);//preparing step
        for (int i = 0; i < N; i++)
            molecules[i]->UVW(d_xC, d_yC, d_zC);
        for (int i = 0; i < N; i++)
        {
            molecules[i]->RungeKuttaStep(dt / 6.f);
            molecules[i]->moleculeStep(dt / 2.f, d_xC, d_yC, d_zC);
        }
        for (int i = 0; i < N; i++)
            molecules[i]->UVW(d_xC, d_yC, d_zC);
        for (int i = 0; i < N; i++)
        {
            molecules[i]->RungeKuttaStep(dt / 3.f);
            molecules[i]->moleculeStep(dt / 2.f, d_xC, d_yC, d_zC);
        }
        for (int i = 0; i < N; i++)
            molecules[i]->UVW(d_xC, d_yC, d_zC);
        for (int i = 0; i < N; i++)
        {
            molecules[i]->RungeKuttaStep(dt / 3.f);
            molecules[i]->moleculeStep(dt, d_xC, d_yC, d_zC);
        }
        for (int i = 0; i < N; i++)
            molecules[i]->UVW(d_xC, d_yC, d_zC);
        for (int i = 0; i < N; i++)
        {
            molecules[i]->RungeKuttaStep(dt / 6.f);
            molecules[i]->epilogueStep();
        }
        if (!(t % 100))
            for (int i = 0; i < N; i++) molecules[i]->writeData();
        if (!(t % 1000))
        {
            stop = std::chrono::high_resolution_clock::now();
            printf("Time of itteration %i is: %f ms;\n", t, std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / 1000000.0);
        }

    }
    stop = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) molecules[i]->writeDataToFile("Results");
    cudaDeviceSynchronize();
    printf("Calculation is over (time is %f ms)", std::chrono::duration_cast<std::chrono::nanoseconds>(stop - startCalculation).count() / 1000000.0);
}