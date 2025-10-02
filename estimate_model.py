import os
import argparse

import subprocess

from dsge import read_yaml
from dsge.codegen import create_fortran_smc
from dsge.translate import write_prior_file



# parse model, nruns, npart, nblocks, and observable
k_choice = [0,1,2,3,4,8,10,15,30,40]
model_choice = [f'FHP[k={k}]' for k in k_choice] + ['SI']

parser = argparse.ArgumentParser(description='Compile and Run a model')
parser.add_argument('--model', type=str, help='Model to run', choices=model_choice, default='FHP[k=4]')
parser.add_argument('--nruns', type=int, help='Number of SMC runs', default=10)
parser.add_argument('--npart', type=int, help='Number of particles', default=16000)
parser.add_argument('--nblocks', type=int, help='Number of blocks', default=3)
parser.add_argument('--observable', type=str, help='Observable to use', default='macro', choices=['macro','macro+expectations'])

args = parser.parse_args()


cmds = '\n'.join([f'srun --mpi=pmi2 ./smc --npart {args.npart} --nblocks {args.nblocks} -pe 1000 --output-file output-{i+1:02d}.json --seed {i+1:02d}' 
                  for i in range(args.nruns)])


slurm=f"""#!/bin/bash
#SBATCH --tasks=50
#SBATCH --mem-per-cpu=500M
#SBATCH --constraint=infiniband
#SBATCH --partition=general-bionic
#SBATCH -o .cache/compiled_models/{args.observable}/{args.model}/estimation.log
#SBATCH -J {args.model}[{args.observable}]
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export STAN_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd .cache/compiled_models/{args.observable}/{args.model}
{cmds}
"""

if 'FHP' in args.model:
    FHP = read_yaml('models/finite_horizon.yaml')
    FHP = FHP.fix_parameters(ρ_η=0,σ_η=0)
    cFHP = FHP.compile_model()
    #yy, prior = cFHP.yy, cFHP.prior.fortran_prior()
    k = int(args.model[6:-1])
    template_file = FHP.smc(k, expectations=0)
    create_fortran_smc(template_file,
                       output_directory=f'.cache/compiled_models/{args.observable}/{args.model}',
                       fortress_cmake_dir='/home/eherbst/Dropbox/code/fortress/build')

    #write_prior_file(cFHP.prior, f'.cache/compiled_models/{args.observable}/{args.model}')
else:
    SI = read_yaml('models/sticky_information.yaml').fix_parameters(h=4,spread=1.0)
    SI.create_fortran_model(f'.cache/compiled_models/{args.observable}/{args.model}', env=env)


with open(f'.cache/compiled_models/{args.observable}/{args.model}/slurm.sh','w') as f:
    f.write(slurm)

subprocess.run([f'sbatch .cache/compiled_models/{args.observable}/{args.model}/slurm.sh'], shell=True)


