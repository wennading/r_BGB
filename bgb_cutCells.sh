#!/bin/bash
#SBATCH -A node                   # Node account
#SBATCH -p node                   # Node partition
#SBATCH -n 6                      # Number of cores
#SBATCH --qos normal              # Priority level
#SBATCH --job-name=run_R_script   # A more descriptive job name
#SBATCH --output=/home/dingwenn/BGB/are%j.out
#SBATCH --error=/home/dingwenn/BGB/are%j.err
#SBATCH --time=696:24:15
#SBATCH --mem=30G
#SBATCH --mail-user=wenna.ding@wsl.ch
#SBATCH --mail-type=SUBMIT,END,FAIL

# --- Load Singularity module if needed ---
# module load singularity

# --- Move into your project directory ---
cd /home/dingwenn/BGB

# --- Run the R script inside the r_geo container ---
singularity exec \
--scratch /run,/var/lib/rstudio-server \
--workdir /home/dingwenn/BGB \
/home/dingwenn/singularity/r_geo.sif \
Rscript cutCells_com.R
