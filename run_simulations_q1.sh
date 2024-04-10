#!/bin/bash

for i in {1..20}
do
mpirun -n $i python3 mpi.py >> mpi.out
done

