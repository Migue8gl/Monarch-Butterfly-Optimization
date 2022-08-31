#!/bin/bash

# Para opción concurrente pasar argumento 1 (tarda menos, pero los tiempos de cada alg no son precisos)
# Para opción secuencial no pasar nada (tarda más en general, pero los tiempos son precisos)

Seed=218728273;

if [[ $1 -eq 1 ]]
then
	./main $Seed 1 > heart_results.txt &
	./main $Seed 2 > parkinsons_results.txt &
	./main $Seed 3 > ionosphere_results.txt
else
	./main $Seed 1 > heart_results.txt 
	./main $Seed 2 > parkinsons_results.txt 
	./main $Seed 3 > ionosphere_results.txt
fi
