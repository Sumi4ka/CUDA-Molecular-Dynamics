#!/bin/bash

# Список файлов для проверки
files=("Molecule.cu" "Molecule.cuh" "Solver.cu" "Solver.cuh" "Kernels.cu" "Kernels.cuh" "main.cu")

# Проверка наличия всех файлов
for file in "${files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "Ошибка: файл $file не найден."
        exit 1
    fi
done

echo "Все файлы найдены. Начинается компиляция..."

# Компиляция main.cu
nvcc -o main_program main.cu Molecule.cu Solver.cu Kernels.cu

if [[ $? -eq 0 ]]; then
    echo "Компиляция завершена успешно."
	./main_program
else
    echo "Ошибка при компиляции."
    exit 1
fi