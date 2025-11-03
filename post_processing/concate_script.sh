#!/bin/sh

for d in */
do
    ( cd "$d" && awk '(NR ==1) || (FNR > 1)' *.csv > Risk_Results.csv )
done