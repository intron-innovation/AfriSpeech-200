# AfriSpeech-100

#### Pan-African accented speech dataset for clinical and general domain ASR

> 100+ African accents totalling  196+ hrs of audio

[By Intron Innovation](https://www.intron.io)

Contributor List: []

#### Progress
- [x] Select & Publish and splits
- [ ] Upload audio
- [ ] Start experiments


#### Abstract [draft]

Africa has a very low doctor:patient ratio. At very buys clinics, doctors could  see 30+ patients per day
 (a very heavy patient burden compared with developed countries), but productivity tools are lacking for these
  overworked clinicians. 
However, clinical ASR is mature in developed nations, even ubiquitous in the US, UK, and Canada (e.g. Dragon, Fluency)
. Clinician-reported performance of commercial clinical ASR systems is mostly satisfactory. Furthermore, recent
 performance of general domain ASR is approaching human accuracy. However, several gaps exist. Several papers have
  highlighted racial bias with speech-to-text algorithms and performance on minority accents lags significantly. 
To our knowledge there is no publicly available research or benchmark on accented African clinical ASR.
We release AfriSpeech, 196 hrs of Pan-African speech across 120 indigenous accents for clinical and general domain ASR
, a benchmark test set and publicly available pretrained models.


#### Data Stats

- Total Number of Unique Speakers: 2,463
- Female/Male/Other Ratio: 51.96/47.38/0.66
- Data was first split on speakers. Speakers in Train/Dev/Test do not cross partitions

|  | Train | Dev | Test |
| ----------- | ----------- | ----------- | ----------- |
| # Speakers | 1466 | 247 | 750 |
| # Seconds | 624239.41 | 31461.05 | 52818.46 |
| # Hours | 173.4 | 8.74 | 14.67 |
| # Accents | 71 | 45 | 108 |
| Avg secs/speaker | 425.81 | 127.37 | 70.42 |
| Avg num clips/speaker | 39.56 | 13.09 | 6.77 |
| Avg num speakers/accent | 20.65 | 5.49 | 6.94 |
| Avg secs/accent | 8792.1 | 699.13 | 489.06 |
| # clips general domain | 21682 | 1407 | 2209 |
| # clips clinical domain | 36319 | 1825 | 2868 |


#### Country Stats

| Country | Clips | Speakers | Duration (seconds) | Duration (hrs) |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| NG | 44722 | 1979 | 499016.25 | 138.62 |
| KE | 8254 | 137 | 74713.57 | 20.75 |
| ZA | 7837 | 223 | 81387.36 | 22.61 |
| GH | 2008 | 37 | 18475.60 | 5.13 |
| BW | 1384 | 38 | 14194.02 | 3.94 |
| UG | 1081 | 26 | 10318.94 | 2.87 |
| RW | 466 | 9 | 5260.11 | 1.46 |
| US | 219 | 5 | 1900.98 | 0.53 |
| TR | 66 | 1 | 664.01 | 0.18 |
| ZW | 63 | 3 | 635.11 | 0.18 |
| MW | 60 | 1 | 554.61 | 0.15 |
| TZ | 51 | 2 | 645.51 | 0.18 |
| LS | 7 | 1 | 78.40 | 0.02 |

#### Accent Stats

|  Accent | Clips | Speakers | Duration (s) | Country | Splits | 
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| yoruba | 15241 | 683 | 159835.76 | US,NG | train,test,dev |
| igbo | 8598 | 374 | 92197.0 | US,NG,ZA | train,test,dev |
| swahili | 6274 | 119 | 55533.56 | KE,ZA,UG,TZ | train,test,dev |
| hausa | 5713 | 248 | 69879.79 | NG | train,test,dev |
| ijaw | 2471 | 105 | 32993.6 | NG | train,test,dev |
| afrikaans | 2047 | 33 | 20575.27 | ZA | train,test,dev |
| idoma | 1860 | 72 | 20314.98 | NG | train,test,dev |
| zulu | 1779 | 52 | 18072.16 | ZA,TR,LS | train,test,dev |
| setswana | 1581 | 39 | 16503.03 | BW,ZA | train,test,dev |
| twi | 1561 | 22 | 14286.57 | GH | train,test,dev |
| isizulu | 1043 | 48 | 10328.73 | ZA | train,test,dev |
| igala | 917 | 31 | 9837.73 | NG | train,test |
| izon | 834 | 47 | 9553.19 | NG | train,test,dev |
| kiswahili | 827 | 6 | 8988.26 | KE | train,test |
| ebira | 750 | 42 | 7669.66 | NG | train,test,dev |
| luganda | 719 | 22 | 6741.09 | UG,BW,KE | train,test,dev |
| urhobo | 642 | 32 | 6628.91 | NG | train,test,dev |
| nembe | 577 | 16 | 6633.1 | NG | train,test,dev |
| ibibio | 552 | 39 | 6295.46 | NG | train,test,dev |
| pidgin | 512 | 20 | 5854.71 | NG | train,test,dev |
| luhya | 499 | 4 | 4372.13 | KE | train,test |
| kinyarwanda | 466 | 9 | 5260.11 | RW | train,test,dev |
| xhosa | 391 | 12 | 4602.02 | ZA | train,test,dev |
| tswana | 383 | 18 | 4118.79 | ZA,BW | train,test,dev |
| esan | 377 | 13 | 4135.55 | NG | train,test,dev |
| alago | 359 | 8 | 3844.58 | NG | train,test |
| tshivenda | 352 | 5 | 3252.63 | ZA | train,test |
| fulani | 309 | 18 | 5047.73 | NG | train,test |
| isoko | 294 | 16 | 4203.16 | NG | train,test,dev |
| akan (fante) | 292 | 9 | 2813.66 | GH | train,test,dev |
| ikwere | 289 | 14 | 3417.86 | NG | train,test,dev |
| sepedi | 273 | 10 | 2736.84 | ZA | train,test,dev |
| efik | 267 | 11 | 2552.32 | NG | train,test,dev |
| edo | 235 | 12 | 1815.63 | NG | train,test,dev |
| luo | 233 | 4 | 2044.66 | KE,UG | train,test,dev |
| kikuyu | 226 | 4 | 1921.25 | KE | train,test,dev |
| isixhosa | 210 | 9 | 2100.28 | ZA | train,test,dev |
| epie | 201 | 6 | 2315.21 | NG | train,test |
| isindebele | 198 | 2 | 1759.49 | ZA | train,test |
| hausa/fulani | 198 | 3 | 2188.53 | NG | train,test |
| bekwarra | 197 | 3 | 1856.74 | NG | train,test |
| venda and xitsonga | 187 | 2 | 2590.48 | ZA | train,test |
| sotho | 182 | 4 | 2082.21 | ZA | train,test,dev |
| nupe | 156 | 9 | 1608.24 | NG | train,test,dev |
| akan | 155 | 6 | 1375.38 | GH | train,test |
| afemai | 142 | 2 | 1877.04 | NG | train,test |
| shona | 138 | 8 | 1419.98 | ZA,ZW | train,test,dev |
| luganda and kiswahili | 134 | 1 | 1356.93 | UG | train |
| sesotho | 131 | 10 | 1387.49 | ZA | train,test,dev |
| kagoma | 123 | 1 | 1781.04 | NG | train |
| nasarawa eggon | 120 | 1 | 1039.99 | NG | train |
| south african english | 119 | 2 | 1643.82 | ZA | train,test |
| tiv | 118 | 14 | 1057.84 | NG | train,test,dev |
| anaang | 118 | 8 | 1148.89 | NG | test,dev |
| benin | 118 | 4 | 1380.35 | NG | train,test |
| english | 115 | 11 | 1725.15 | NG | test,dev |
| borana | 112 | 1 | 1090.71 | KE | train |
| swahili ,luganda ,arabic | 109 | 1 | 929.46 | UG | train |
| ogoni | 109 | 4 | 1629.7 | NG | train,test |
| eggon | 106 | 5 | 1430.9 | NG | test |
| bette | 104 | 4 | 909.16 | NG | train,test |
| ngas | 100 | 3 | 1195.67 | NG | train,test |
| venda | 99 | 2 | 938.14 | ZA | train,test |
| ukwuani | 97 | 7 | 917.24 | NG | test |
| okrika | 96 | 3 | 1861.47 | NG | train,test |
| siswati | 95 | 5 | 1351.7 | ZA | train,test,dev |
| etsako | 94 | 4 | 1014.53 | NG | train,test |
| damara | 92 | 1 | 674.43 | NG | train |
| berom | 91 | 4 | 1133.09 | NG | test,dev |
| southern sotho | 89 | 1 | 889.73 | ZA | train |
| bini | 83 | 4 | 1139.91 | NG | test |
| itsekiri | 82 | 3 | 778.47 | NG | test,dev |
| mada | 80 | 2 | 1399.91 | NG | test |
| luo, swahili | 71 | 1 | 616.57 | KE | train |
| kanuri | 70 | 7 | 1308.81 | NG | test,dev |
| dholuo | 70 | 1 | 669.07 | KE | train |
| ateso | 63 | 1 | 624.28 | UG | train |
| ekpeye | 61 | 2 | 633.0 | NG | test |
| chichewa | 60 | 1 | 554.61 | MW | train |
| mwaghavul | 58 | 2 | 561.02 | NG | test |
| meru | 58 | 2 | 865.07 | KE | train,test |
| bajju | 55 | 2 | 529.78 | NG | test |
| ika | 55 | 4 | 485.62 | NG | test,dev |
| yoruba, hausa | 55 | 5 | 588.73 | NG | test |
| oklo | 54 | 1 | 799.52 | NG | test |
| ekene | 54 | 1 | 659.25 | NG | test |
| jaba | 52 | 2 | 441.4 | NG | test |
| portuguese | 50 | 1 | 525.02 | ZA | train |
| brass | 46 | 2 | 680.92 | NG | test |
| angas | 45 | 1 | 412.19 | NG | test |
| ikulu | 44 | 1 | 250.2 | NG | test |
| eleme | 42 | 2 | 872.92 | NG | test |
| ijaw(nembe) | 41 | 2 | 410.26 | NG | test |
| igarra | 40 | 1 | 454.26 | NG | test |
| delta | 39 | 2 | 340.05 | NG | test |
| gbagyi | 39 | 4 | 398.21 | NG | test |
| ogbia | 39 | 4 | 363.15 | NG | test,dev |
| okirika | 37 | 1 | 533.3 | NG | test |
| khana | 36 | 2 | 354.93 | NG | test |
| etche | 36 | 1 | 444.91 | NG | test |
| bassa | 35 | 1 | 458.13 | NG | test |
| agatu | 34 | 1 | 214.11 | NG | test |
| igbo and yoruba | 33 | 2 | 370.27 | NG | test |
| kalabari | 33 | 5 | 216.49 | NG | test |
| jukun | 32 | 2 | 277.82 | NG | test |
| kubi | 32 | 1 | 357.1 | NG | test |
| urobo | 31 | 3 | 408.78 | NG | test |
| ibani | 30 | 1 | 259.42 | NG | test |
| bassa-nge/nupe | 28 | 3 | 233.42 | NG | test,dev |
| obolo | 27 | 1 | 145.79 | NG | test |
| idah | 25 | 1 | 461.5 | NG | test |
| yala mbembe | 22 | 1 | 179.82 | NG | test |
| eket | 20 | 1 | 190.85 | NG | test |
| ebiobo | 20 | 1 | 172.42 | NG | test |
| nyandang | 19 | 1 | 158.27 | NG | test |
| ishan | 18 | 1 | 147.12 | NG | test |
| estako | 18 | 1 | 448.78 | NG | test |
| afo | 17 | 1 | 120.41 | NG | test |
| bagi | 15 | 1 | 235.54 | NG | test |
| gerawa | 10 | 1 | 245.01 | NG | test |


--------
 
#### License

(c) 2022. This work is licensed under a CC BY-NC-SA 4.0 license. See official instructions [here](https://creativecommons.org/about/cclicenses/).