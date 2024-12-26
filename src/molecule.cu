#include "molecule.cuh"
void Molecule::Description()
{
    printf("Molecule name: %s\n", name.c_str());
    printf("Structure name: %s\n", nameStructure.c_str());
    printf("x: %f, y: %f, z: %f\n", xM, yM, zM);
    printf("u: %f, v: %f, w: %f\n", uM, vM, wM);
    printf("omegaX: %f, omegaY: %f, omegaZ: %f\n", omegaX, omegaY, omegaZ);
}
void Molecule::getC(float* xC, float* yC, float* zC)
{
    this->xC = xC; this->yC = yC; this->zC = zC;
}
void Molecule::angleVelocity()
{
    double A = (double)ABCDEF[0];
    double B = (double)ABCDEF[1];
    double C = (double)ABCDEF[2];
    double D = (double)ABCDEF[3];
    double E = (double)ABCDEF[4];
    double F = (double)ABCDEF[5];
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
    omegaX = (float)OmegaX;
    omegaY = (float)OmegaY;
    omegaZ = (float)OmegaZ;
}
void Molecule::kineticMoment()
{
    Kx = ABCDEF[0] * omegaX + ABCDEF[5] * omegaY + ABCDEF[4] * omegaZ;
    Kz = ABCDEF[4] * omegaX + ABCDEF[3] * omegaY + ABCDEF[2] * omegaZ;
    Ky = ABCDEF[5] * omegaX + ABCDEF[1] * omegaY + ABCDEF[3] * omegaZ;
}
void Molecule::readData()
{
    std::string sData = "../data/";
    std::ifstream in(sData + nameStructure + ".txt");
    std::string s1, xData, yData, zData;
    in >> s1;
    if (s1 == "Carbon")
    {
        in >> s1;
        M = stoi(s1);
        sizeM = M * sizeof(float);
        xA = (float*)malloc(sizeM);
        yA = (float*)malloc(sizeM);
        zA = (float*)malloc(sizeM);
        for (int j = 0; j < M; j++)
        {
            in >> xData >> yData >> zData;
            xA[j] = stof(xData);
            yA[j] = stof(yData);
            zA[j] = stof(zData);
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
        avgX += xA[j];
        avgY += yA[j];
        avgZ += zA[j];
    }
    avgX /= M; avgY /= M; avgZ /= M;
    for (int j = 0; j < M; j++)
    {
        xA[j] -= avgX;
        yA[j] -= avgY;
        zA[j] -= avgZ;
    }
}
Molecule::Molecule(std::string name, std::string nameStructure, int N, int n)
{
    this->name = std::to_string(n+1) + "Molecule"; this->nameStructure = nameStructure;
    this->n = n;
    this->N = N;
    readData();              //prepare Data to calculate
    sizeM = M * sizeof(float);
}
Molecule::~Molecule()
{
    free(xA);
    free(yA);
    free(zA);
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
void Molecule::prepareStep()
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
}
void Molecule::moleculeStep(float dt)
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
void Molecule::writeData()
{
    xData.push_back(xM);
    yData.push_back(yM);
    zData.push_back(zM);
}
void Molecule::writeDataToFile(std::string sData)
{
    std::ofstream out(sData + '/' + name + ".txt");
    for (int i = 0; i < xData.size(); i++)
        out << xData[i] << ' ' << yData[i] << ' ' << zData[i] << std::endl;
    out.close();
}

