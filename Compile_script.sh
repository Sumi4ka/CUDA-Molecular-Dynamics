#!/bin/bash

# ������ ������ ��� ��������
files=("Molecule.cu" "Molecule.cuh" "Solver.cu" "Solver.cuh" "Kernels.cu" "Kernels.cuh" "main.cu")

# �������� ������� ���� ������
for file in "${files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "������: ���� $file �� ������."
        exit 1
    fi
done

echo "��� ����� �������. ���������� ����������..."

# ���������� main.cu
nvcc -o main_program main.cu Molecule.cu Solver.cu Kernels.cu

if [[ $? -eq 0 ]]; then
    echo "���������� ��������� �������."
	./main_program
else
    echo "������ ��� ����������."
    exit 1
fi