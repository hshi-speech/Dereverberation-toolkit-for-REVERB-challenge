#!/bin/bash
#$ -S /bin/bash

#here you'd best to change testjob as username
#$ -N matlab-evaluation-dnn-256-mapping-0005

#cwd define the work environment,files(username.o) will generate here
#$ -cwd

# merge stdo and stde to one file
#$ -j y

echo "job start time: `date`"
# start whatever your job below, e.g., python, matlab, etc.
#ADD YOUR COMMAND HERE,LIKE python3 main.py
#chmod a+x run.sh
/opt18/MATLAB/R2017b/bin/matlab -r Main_pesq_reverb2014_et_enhanced

# resource requesting, e.g. for gpu use
#$ -l h=node02

hostname
sleep 10
echo "job end time:`date`"
