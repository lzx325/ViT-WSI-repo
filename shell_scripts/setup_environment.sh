source /sw/csi/anaconda3/4.4.0/binary/anaconda3/etc/profile.d/conda.sh
module purge
conda deactivate
module load cuda/11.1.1
conda activate clam_pt1.8
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.conda/envs/clam_pt1.8/lib/"
unset DISPLAY

