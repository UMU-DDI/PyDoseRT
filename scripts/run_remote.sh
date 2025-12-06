for patient_name in P01 P02
do
    sbatch run_remote.sh "$patient_name"
done