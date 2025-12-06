for patient_name in P01 P02
do
    sbatch scripts/remote_script.sh "$patient_name"
done