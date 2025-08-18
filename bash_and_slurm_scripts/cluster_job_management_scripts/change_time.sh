for jobid in $(squeue -u $USER -h -o %i); do
    scontrol update jobid=$jobid TimeLimit=04:00:00
done
