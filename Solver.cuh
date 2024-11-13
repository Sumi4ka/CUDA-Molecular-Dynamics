#pragma once
#include "Molecule.cuh"
#include <chrono>
class Solver
{
private:
public:
    int N, M, NT;
    float dt;
    std::vector<Molecule*> molecules;
    float* d_xC, * d_yC, * d_zC;
    Solver(std::string Description);
    ~Solver();
    void Solve();
};
