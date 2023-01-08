
module load anaconda/3
module load cuda/11.0/cudnn/8.0


conda create --prefix ~/scratch/nemo2 python==3.8
conda activate /home/mila/c/chris.emezue/scratch/nemo2
conda install pytorch==1.7.0 cudatoolkit=11.0 -c pytorch
pip install Cython
pip install nemo_toolkit['all']