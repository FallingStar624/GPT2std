#! /bin/bash

jid1=$(sbatch --partition=3090 train1.sh)
echo "${jid1}"

jid2=$(sbatch --dependency=afternotok:${jid1##* } --partition=3090 train1.sh --idx 1 --resume)
echo "${jid2}"

jid3=$(sbatch --dependency=afternotok:${jid2##* } --partition=3090 train1.sh --idx 1 --resume)
echo "${jid3}"

jid4=$(sbatch --dependency=afternotok:${jid3##* } --partition=3090 train1.sh --idx 1 --resume)
echo "${jid4}"
