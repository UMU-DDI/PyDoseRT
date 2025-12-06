for patient_name in P03
do
    sbatch scripts/remote_script.sh "$patient_name"
done