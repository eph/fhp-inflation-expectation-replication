import os
import argparse
import subprocess
import sys

from dsge import read_yaml
from dsge.codegen import create_fortran_smc

# Define dataset paths
dataset_map = {
    'nan': '.cache/observables/fullsample_with_nan_inflation_expectations.txt',
    '1q': '.cache/observables/fullsample_with_1q_inflation_expectations.txt',
    '4q': '.cache/observables/fullsample_with_4q_inflation_expectations.txt'
}

# parse model, nruns, npart, nblocks, and dataset
k_choice = [0, 1, 2, 3, 4, 8, 10, 15, 30, 40]
model_choice = [f'FHP[k={k}]' for k in k_choice] + ['SI']

parser = argparse.ArgumentParser(description='Compile and Run a model')
parser.add_argument('--model', type=str, help='Model to run', choices=model_choice, default='FHP[k=4]')
parser.add_argument('--nruns', type=int, help='Number of SMC runs', default=10)
parser.add_argument('--npart', type=int, help='Number of particles', default=16000)
parser.add_argument('--nblocks', type=int, help='Number of blocks', default=3)
parser.add_argument('--dataset', type=str, help='Dataset to use', default='nan', choices=['nan', '1q', '4q'])
parser.add_argument('--no-slurm', action='store_true', help='Run estimation directly without SLURM')
parser.add_argument('--seed', type=int, help='Random seed for SMC (only used with --no-slurm)', default=1)
parser.add_argument('--pe', type=int, help='SMC tempering schedule parameter', default=1000)
parser.add_argument('--ntasks', type=int, help='Number of SLURM tasks (only used with SLURM)', default=50)
parser.add_argument('--mem-per-cpu', type=str, help='Memory per CPU for SLURM (only used with SLURM)', default='500M')
parser.add_argument('--partition', type=str, help='SLURM partition (only used with SLURM)', default='general-bionic')
parser.add_argument('--constraint', type=str, help='SLURM constraint (only used with SLURM)', default='infiniband')

args = parser.parse_args()

# Check for disallowed combination
if args.model == 'SI' and args.dataset == '1q':
    sys.exit("Error: The SI model is not compatible with the '1q' dataset.")

# Get the data file path
data_file = dataset_map[args.dataset]
output_dir = f'.cache/compiled_models/{args.dataset}/{args.model}'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

if not args.no_slurm:
    cmds = '\n'.join([f'srun --mpi=pmi2 ./smc_driver --npart {args.npart} --nblocks {args.nblocks} -pe {args.pe} --output-file output-{i+1:02d}.json --seed {i+1:02d}'
                      for i in range(args.nruns)])

    slurm = f"""#!/bin/bash
#SBATCH --tasks={args.ntasks}
#SBATCH --mem-per-cpu={args.mem_per_cpu}
#SBATCH --constraint={args.constraint}
#SBATCH --partition={args.partition}
#SBATCH -o {output_dir}/estimation.log
#SBATCH -J {args.model}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export STAN_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd {output_dir}
{cmds}
"""

if 'FHP' in args.model:
    FHP = read_yaml('models/finite_horizon.yaml')
    FHP.update_data_file(data_file, start_date='1966Q1')
    if args.dataset == 'nan':
        FHP = FHP.fix_parameters(ρ_η=0, σ_η=0)
    cFHP = FHP.compile_model()
    k = int(args.model[6:-1])
    
    expectations_map = {'nan': 0, '1q': 1, '4q': 4}
    expectations_val = expectations_map[args.dataset]
    
    template_file = FHP.smc(k, expectations=expectations_val)
    create_fortran_smc(template_file,
                       output_directory=output_dir)
#                       fortress_cmake_dir='/home/eherbst/Dropbox/code/fortress/build')

else:
    SI = read_yaml('models/sticky_information.yaml')
    SI.update_data_file(data_file, start_date='1966Q1')
    if args.dataset == 'nan':
        SI = SI.fix_parameters(h=4, spread=1.0)
    SI.create_fortran_model(output_dir, env=env)

if args.no_slurm:
    # Run estimation directly
    print(f"Running SMC estimation directly in {output_dir}/build...")
    smc_cmd = f'./smc_driver --npart {args.npart} --nblocks {args.nblocks} -pe {args.pe} --output-file output.json --seed {args.seed}'
    subprocess.run(smc_cmd, shell=True, cwd=f'{output_dir}/build', check=True)
    print(f"Estimation complete. Results saved to {output_dir}/build/output.json")
else:
    # Submit SLURM job
    with open(f'{output_dir}/slurm.sh', 'w') as f:
        f.write(slurm)

    subprocess.run([f'sbatch {output_dir}/slurm.sh'], shell=True)
