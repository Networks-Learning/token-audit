#!/bin/bash

echo "Starting the script..."

# File containing the list of string arguments
TEXT_FILE="data/LMSYS.txt"

# Number of lines per job (adjustable)
LINES_PER_JOB=300

# Total number of jobs to submit (adjustable)
TOTAL_JOBS=15

START_OFFSET=1  # Start reading from

# Temperature and p values to sweep
TEMPERATURES=( 1.0 1.15 1.3)
PS=( 6 )

# Activate the virtual environment once for efficiency
source env/bin/activate

# Loop through the text file in chunks based on the specified parameters
for ((job_index=0; job_index<TOTAL_JOBS; job_index++)); do
    # Calculate the starting and ending lines for this job
    start_line=$((START_OFFSET + job_index * LINES_PER_JOB))
    end_line=$((start_line + LINES_PER_JOB - 1))
    
    # Extract the lines for this job and format them as space-separated quoted strings
    prompts=$(sed -n "${start_line},${end_line}p" "$TEXT_FILE" | sed 's/"/\\"/g' | awk '{printf("\"%s\" ", $0)}')
    
    # Loop over all temperature and p combinations
    for temp in "${TEMPERATURES[@]}"; do
        for p in "${PS[@]}"; do
            # Submit the job to SLURM
            sbatch <<EOF
#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 
#SBATCH --partition=a100
#SBATCH --gres gpu:1            # set 1 GPUs per job
#SBATCH --mem=40G
#SBATCH -o outputs/slurm_logs/simulation_%j.out
#SBATCH -e outputs/slurm_logs/simulation_%j.err

source env/bin/activate
cd src

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


python -u audit_faithful.py --prompts ${prompts} --temperature $temp --model "G1B" --poisson $p --job_id $start_line 
EOF
        done
    done
done

# Deactivate the virtual environment
deactivate