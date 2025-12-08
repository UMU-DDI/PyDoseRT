for patient_name in P0
do
    sbatch scripts/remote_script.sh "$patient_name"
done