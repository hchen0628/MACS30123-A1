#!/bin/bash

for i in {1..20}
do
mpirun -n $i 123_1b_test.py >> mpi.out
done

