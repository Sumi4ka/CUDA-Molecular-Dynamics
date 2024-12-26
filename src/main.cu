#include "solver.cuh"
int main(int argc, char* argv[])
{
    Solver* S;
    bool withGPU = true;
    std::string DataFile, withGPUstring;
    if (argc > 1)
    {
        DataFile = argv[1];
	    if (argc > 2)
            withGPU = (bool)std::atoi(argv[2]);
    }
    else DataFile ="Description.txt";
    if (argc<=2)
    {   
        printf("Calculation with CUDA?\n");
        while (true)
        {
            printf("true or false\n");
	        std::cin>>withGPUstring;
            if (withGPUstring=="true") {withGPU = true; break;}
            if (withGPUstring=="false") {withGPU = false; break;}
	    }
    }
    printf("Parsing Data from %s\n", DataFile.c_str());
    S = new Solver("../" + DataFile, withGPU);
    S->Solve();
    delete S;
    std::cin.get();
    return 0;
}