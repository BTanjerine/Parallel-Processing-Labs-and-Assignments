#! /bin/bash

#SBATCH --job-name=RUN_PI_MPI
#SBATCH --output=RUN_PI_MPI_%j.log
#SBATCH --output=RUN_PI_MPI_%j.log

#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --mem=1gb
#SBATCH --time=00:02:00

. /etc/profile.d/modules.sh

module load openmpi/2.1.2

/opt/openmpi-2.1.2/bin/mpirun ./mc

