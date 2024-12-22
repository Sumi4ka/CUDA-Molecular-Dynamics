#pragma once
#include "molecule.cuh"
#include <chrono>
#include <filesystem>
class Solver
{
private:
    bool GPU = false;
    void Description();
    std::string Folder;
public:
    int N, M, NT;
    float dt;
    std::vector<Molecule*> molecules;
    float* xC, * yC, * zC;
    Solver(std::string DataFile, bool GPU);
    ~Solver();
    void Solve();
};
