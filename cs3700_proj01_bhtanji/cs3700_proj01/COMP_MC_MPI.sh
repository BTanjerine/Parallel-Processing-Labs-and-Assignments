#! /bin/bash

#SBATCH --job-name=COMP_PI_MPI
#SBATCH --output=COMP_PI_MPI_%j.log
#SBATCH --partition=compute
#SBATCH --mem=1gb
#SBATCH --nodes=1
#SBATCH --time=00:02:00

. /etc/profile.d/modules.sh

module load openmpi/2.1.2

/opt/openmpi-2.1.2/bin/mpirun g++ -o mc monteCarlo_MPI.cpp -I /opt/openmpi-2.1.2/include -L /opt/openmpi-2.1.2/lib -l mpi
