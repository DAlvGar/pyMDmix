docker build -t pymdmix:latest -f Dockerfile .
singularity build pymdmix.sif docker-daemon://pymdmix:latest
singularity exec -B /run/shm:/run/shm pymdmix.sif mdmix info
