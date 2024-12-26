#include "solver.cuh"
void Solver::Description()
{
    printf("\nDescription\n\n");
    printf("Count of Molecules: %i\n",N);
    printf("Count of Time Iterations: %i\n", NT);
    printf("Time Step: %f\n\n", dt);
    printf("Molecules:\n\n");
    for (int i = 0; i < N; i++)
    {
        molecules[i]->Description();
        printf("\n");
    }
    if (GPU) printf("Calculation with GPU\n");
    else printf("Calculation withOUT GPU\n\n");
    printf("\n");
}
Solver::Solver(std::string DataFile, bool GPU)
{
    //Description input
    this->GPU = GPU;
    molecules = std::vector<Molecule*>();
    std::ifstream is(DataFile);
    std::string inString, name, nameStructure, x, y, z;
    std::getline(is, inString);
    is >> inString;
    is >> inString;
    Folder = inString;
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
        nameStructure = inString;
        if (this->GPU)
            molecules.push_back(new MoleculeGPU(name, nameStructure, N, i));
        else
            molecules.push_back(new MoleculeCPU(name, nameStructure, N, i));
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
    size_t sizeM = M * sizeof(float);
    if (this->GPU) {
        Folder = "../results/" + Folder + '/' + "GPU_Results";
        cudaError_t err;
        err = cudaMalloc((void**)&xC, sizeM);
        err = cudaMalloc((void**)&yC, sizeM);
        err = cudaMalloc((void**)&zC, sizeM);
    }
    else {
        Folder = "../results/" + Folder + '/' + "CPU_Results";
        xC = (float*)malloc(sizeM);
        yC = (float*)malloc(sizeM);
        zC = (float*)malloc(sizeM);
    }
    try {
        if (std::filesystem::create_directories(Folder)) std::cout << "Folder was created correctly: " << Folder << std::endl;
        else {
            if (std::filesystem::exists(Folder)) std::cout << "The folder has already been created : " << Folder << std::endl;
            else std::cout << "The folder could not be created for an unknown reason!!!" << std::endl;
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    for (int i = 0; i < N; i++) molecules[i]->getC(xC, yC, zC);
    is.close();
    this->Description();
}
Solver::~Solver()
{
    for (int i = 0; i < N; i++)
        delete molecules[i];
    if (this->GPU)
    {
        cudaFree(xC);
        cudaFree(yC);
        cudaFree(zC);
    }
    else
    {
        free(xC);
        free(yC);
        free(zC);
    }
}
void Solver::Solve()
{
    printf("Start the calculation?\n");
    std::cin.get();
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
        if (!(t % 100)) start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++)
            molecules[i]->prepareStep();//preparing step
        for (int i = 0; i < N; i++)
            molecules[i]->calculateFLJ();
        for (int i = 0; i < N; i++)
        {
            molecules[i]->RungeKuttaStep(dt / 6.f);
            molecules[i]->moleculeStep(dt / 2.f);
        }
        for (int i = 0; i < N; i++)
            molecules[i]->calculateFLJ();
        for (int i = 0; i < N; i++)
        {
            molecules[i]->RungeKuttaStep(dt / 3.f);
            molecules[i]->moleculeStep(dt / 2.f);
        }
        for (int i = 0; i < N; i++)
            molecules[i]->calculateFLJ();
        for (int i = 0; i < N; i++)
        {
            molecules[i]->RungeKuttaStep(dt / 3.f);
            molecules[i]->moleculeStep(dt);
        }
        for (int i = 0; i < N; i++)
            molecules[i]->calculateFLJ();
        for (int i = 0; i < N; i++)
        {
            molecules[i]->RungeKuttaStep(dt / 6.f);
            molecules[i]->epilogueStep();
        }
        if (!(t % 1))
            for (int i = 0; i < N; i++) molecules[i]->writeData();
        if (!(t % 100))
        {
            stop = std::chrono::high_resolution_clock::now();
            printf("Time of itteration %i is: %f ms;\n", t, std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / 1000000.0);
        }

    }
    stop = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) molecules[i]->writeDataToFile(Folder+ '/');
    cudaDeviceSynchronize();
    printf("Calculation is over (time is %f ms)", std::chrono::duration_cast<std::chrono::nanoseconds>(stop - startCalculation).count() / 1000000.0);
}