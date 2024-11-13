#include "Solver.cuh"
#include <sstream>
int main(int argc, char* argv[])
{
    Solver* S;
    if (argc > 1)
    {
        std::stringstream ss;
        std::string Description;
        ss << argv[1];
        ss >> Description;
        S = new Solver(Description);
    }
    else S = new Solver("Description.txt");
    S->Solve();
    delete S;
    return 0;
}