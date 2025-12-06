for patient_name in P01 P02 P03 P04 P05
do
    sbatch scripts/remote_script.sh "$patient_name"
done