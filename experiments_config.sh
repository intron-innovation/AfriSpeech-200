#!/bin/bash

# accent + domain
sbatch run_experiments.sh src/config/config_xlsr_group_lengths_multi_task_accent_domain.ini

# accent (prepend)
sbatch run_experiments.sh src/config/config_xlsr_group_lengths_multi_task_accent.ini

# domain
sbatch run_experiments.sh src/config/config_xlsr_group_lengths_multi_task_domain.ini

# accent (append)
sbatch run_experiments.sh src/config/config_xlsr_group_lengths_multi_task_accent_append.ini

# accent + domain + append
sbatch run_experiments.sh src/config/config_xlsr_group_lengths_multi_task_accent_domain_append.ini

# domain + append
#sbatch run_experiments.sh src/config/config_xlsr_group_lengths_multi_task_domain_append.ini


#Submitted batch job 2930580
#Submitted batch job 2930581
#Submitted batch job 2930582
#Submitted batch job 2930583
#Submitted batch job 2930584
#Submitted batch job 2928885


# UNFROZEN
#Submitted batch job 2935178
#Submitted batch job 2935179
#Submitted batch job 2935180
#Submitted batch job 2935181
#Submitted batch job 2935182