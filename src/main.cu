#include "solver.cuh"
#include <sstream>
int main(int argc, char* argv[])
{
    Solver* S;
    std::string DataFile;
    if (argc > 1)
    {
        std::stringstream ss;
        ss << argv[1];
        ss >> DataFile;
    }
    else DataFile ="Description.txt";
    printf("Parsing Data from %s\n", DataFile.c_str());
    S = new Solver("../"+DataFile, true);
    S->Solve();
    delete S;
    std::cin.get();
    return 0;
}