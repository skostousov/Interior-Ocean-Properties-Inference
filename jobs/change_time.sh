for jobid in $(squeue -u $USER -h -o %i); do
    scontrol update jobid=$jobid TimeLimit=10:30:00
done