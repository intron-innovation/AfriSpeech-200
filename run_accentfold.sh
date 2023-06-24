#!/bin/bash

# normal mode (1 and 2)
# CHEKPOINT_PATH='None' # default
# APPEND_NAME="_" # default
# EPOCH=10
# for B in "bini" "angas" "agatu"
# #for B in "angas"
# do
#     for neighbor in 10 20 35 10000 # 10000 corresponds to finetuning on the whole training dataset 
#     #for neighbor in 35
#     do
#         sbatch train_accent.sh $B $neighbor $APPEND_NAME $CHEKPOINT_PATH $EPOCH
#         #bash train_accent.sh $B $neighbor ${APPEND_NAME} ${CHEKPOINT_PATH} ${EPOCH}
#     done
# done



# # #continuation of finetuning
# CHEKPOINT_PATH="decipher" 
# APPEND_NAME="_cont_test_6" # tells it to continue from 6 
# EPOCH=6
# #for B in "bini" "angas" "agatu"
# for B in "agatu"
# do
#     #for neighbor in 10 20 35
#     for neighbor in 10
#     do
#         #sbatch train_accent.sh $B $neighbor $APPEND_NAME $CHEKPOINT_PATH $EPOCH
#         bash train_accent.sh $B $neighbor ${APPEND_NAME} ${CHEKPOINT_PATH} ${EPOCH}
#     done
# done


# using random subset of accents
CHEKPOINT_PATH="None" 
APPEND_NAME="_random" # tells it to continue from 6 
EPOCH=10
#for B in "bini" "angas" "agatu"
#for B in 'afemai' 'angas' 'afo' 'afrikaans' 'akan' 'akan (fante)' 'alago' 'anaang' 'bagi' 'bajju' 'bassa' 'bassa-nge/nupe' 'bekwarra' 'benin' 'berom' 'bette' 'brass' 'delta' 'ebiobo' 'ebira' 'edo' 'efik' 'eggon' 'ekene' 'eket' 'ekpeye' 'eleme' 'english' 'epie' 'esan' 'estako' 'etche' 'etsako' 'fulani' 'gbagyi' 'gerawa' 'hausa' 'hausa/fulani' 'ibani' 'ibibio' 'idah' 'idoma' 'igala' 'igarra' 'igbo' 'igbo and yoruba' 'ijaw' 'ijaw(nembe)' 'ika' 'ikulu' 'ikwere' 'ishan' 'isindebele' 'isixhosa' 'isizulu' 'isoko' 'itsekiri' 'izon' 'jaba' 'jukun' 'kalabari' 'kanuri' 'khana' 'kikuyu' 'kinyarwanda' 'kiswahili' 'kubi' 'luganda' 'luhya' 'luo' 'mada' 'meru' 'mwaghavul' 'nembe' 'ngas' 'nupe' 'nyandang' 'obolo' 'ogbia' 'ogoni' 'okirika' 'oklo' 'okrika' 'pidgin' 'sepedi' 'sesotho' 'setswana' 'shona' 'siswati' 'sotho' 'south african english' 'swahili' 'tiv' 'tshivenda' 'tswana' 'twi' 'ukwuani' 'urhobo' 'urobo' 'venda' 'venda and xitsonga' 'xhosa' 'yala mbembe' 'yoruba' 'yoruba hausa' 'zulu'
for B in "angas"
do
    #for neighbor in 10 20 35
    for neighbor in 20 #35
    do
        sbatch train_accent.sh $B $neighbor $APPEND_NAME $CHEKPOINT_PATH $EPOCH
        #bash train_accent.sh $B $neighbor ${APPEND_NAME} ${CHEKPOINT_PATH} ${EPOCH}
    done
done




# using random subset of accents
# CHEKPOINT_PATH="None" 
# #APPEND_NAME="_random"
# EPOCH=10
# B='test_accent_not_exists'
# #for B in "bini" "angas" "agatu"
# #for B in 'afemai' 'angas' 'afo' 'afrikaans' 'akan' 'akan (fante)' 'alago' 'anaang' 'bagi' 'bajju' 'bassa' 'bassa-nge/nupe' 'bekwarra' 'benin' 'berom' 'bette' 'brass' 'delta' 'ebiobo' 'ebira' 'edo' 'efik' 'eggon' 'ekene' 'eket' 'ekpeye' 'eleme' 'english' 'epie' 'esan' 'estako' 'etche' 'etsako' 'fulani' 'gbagyi' 'gerawa' 'hausa' 'hausa/fulani' 'ibani' 'ibibio' 'idah' 'idoma' 'igala' 'igarra' 'igbo' 'igbo and yoruba' 'ijaw' 'ijaw(nembe)' 'ika' 'ikulu' 'ikwere' 'ishan' 'isindebele' 'isixhosa' 'isizulu' 'isoko' 'itsekiri' 'izon' 'jaba' 'jukun' 'kalabari' 'kanuri' 'khana' 'kikuyu' 'kinyarwanda' 'kiswahili' 'kubi' 'luganda' 'luhya' 'luo' 'mada' 'meru' 'mwaghavul' 'nembe' 'ngas' 'nupe' 'nyandang' 'obolo' 'ogbia' 'ogoni' 'okirika' 'oklo' 'okrika' 'pidgin' 'sepedi' 'sesotho' 'setswana' 'shona' 'siswati' 'sotho' 'south african english' 'swahili' 'tiv' 'tshivenda' 'tswana' 'twi' 'ukwuani' 'urhobo' 'urobo' 'venda' 'venda and xitsonga' 'xhosa' 'yala mbembe' 'yoruba' 'yoruba hausa' 'zulu'
# #for neighbor in 10 20 35

# for APPEND_NAME in '_random' '_'
# do
#     for neighbor in 20
#     do
#         sbatch train_accent_array.sh $B $neighbor $APPEND_NAME $CHEKPOINT_PATH $EPOCH
#         #bash train_accent_array.sh $B $neighbor ${APPEND_NAME} ${CHEKPOINT_PATH} ${EPOCH}
#     done
# done



# # using random subset of accents - SEPARATE RUNS
# CHEKPOINT_PATH="None" 
# #APPEND_NAME="_random" # 62 77
# APPEND_NAME="_" # 69 78 102

# EPOCH=10
# #for B in "bini" "angas" "agatu"
# #for B in 'afemai' 'angas' 'afo' 'afrikaans' 'akan' 'akan (fante)' 'alago' 'anaang' 'bagi' 'bajju' 'bassa' 'bassa-nge/nupe' 'bekwarra' 'benin' 'berom' 'bette' 'brass' 'delta' 'ebiobo' 'ebira' 'edo' 'efik' 'eggon' 'ekene' 'eket' 'ekpeye' 'eleme' 'english' 'epie' 'esan' 'estako' 'etche' 'etsako' 'fulani' 'gbagyi' 'gerawa' 'hausa' 'hausa/fulani' 'ibani' 'ibibio' 'idah' 'idoma' 'igala' 'igarra' 'igbo' 'igbo and yoruba' 'ijaw' 'ijaw(nembe)' 'ika' 'ikulu' 'ikwere' 'ishan' 'isindebele' 'isixhosa' 'isizulu' 'isoko' 'itsekiri' 'izon' 'jaba' 'jukun' 'kalabari' 'kanuri' 'khana' 'kikuyu' 'kinyarwanda' 'kiswahili' 'kubi' 'luganda' 'luhya' 'luo' 'mada' 'meru' 'mwaghavul' 'nembe' 'ngas' 'nupe' 'nyandang' 'obolo' 'ogbia' 'ogoni' 'okirika' 'oklo' 'okrika' 'pidgin' 'sepedi' 'sesotho' 'setswana' 'shona' 'siswati' 'sotho' 'south african english' 'swahili' 'tiv' 'tshivenda' 'tswana' 'twi' 'ukwuani' 'urhobo' 'urobo' 'venda' 'venda and xitsonga' 'xhosa' 'yala mbembe' 'yoruba' 'yoruba hausa' 'zulu'
# #for neighbor in 10 20 35

# for B in 69 78 102
# do
#     for neighbor in 20
#     do
#         sbatch train_accent_array2.sh $B $neighbor $APPEND_NAME $CHEKPOINT_PATH $EPOCH
#         #bash train_accent_array2.sh $B $neighbor ${APPEND_NAME} ${CHEKPOINT_PATH} ${EPOCH}
#     done
# done
