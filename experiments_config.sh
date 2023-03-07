#!/bin/bash

# accent + domain
#sbatch run_experiments.sh src/config/config_xlsr_group_lengths_multi_task_accent_domain.ini

# accent (prepend)
#sbatch run_experiments.sh src/config/config_xlsr_group_lengths_multi_task_accent.ini

# domain
#sbatch run_experiments.sh src/config/config_xlsr_group_lengths_multi_task_domain.ini


# accent (append)
#sbatch run_experiments.sh src/config/config_xlsr_group_lengths_multi_task_accent_append.ini

# accent + domain + append
sbatch run_experiments.sh src/config/config_xlsr_group_lengths_multi_task_accent_domain_append.ini

# domain + append
sbatch run_experiments.sh src/config/config_xlsr_group_lengths_multi_task_domain_append.ini


#Submitted batch job 2905943
#Submitted batch job 2905944
#Submitted batch job 2905945
#Submitted batch job 2905946

#Submitted batch job 2914945
#Submitted batch job 2914946