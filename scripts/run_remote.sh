for patient_name in P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12 P13 P14 P15 P16 P17 P18 P19
do
    sbatch scripts/remote_script.sh "$patient_name"
    sbatch scripts/remote_script.sh "$patient_name"
done