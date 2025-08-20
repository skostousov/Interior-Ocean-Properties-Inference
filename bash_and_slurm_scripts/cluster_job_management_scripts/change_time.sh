for jobid in $(squeue -u $USER -h -o %i); do
    scontrol update jobid=$jobid TimeLimit=18:00:00
done
