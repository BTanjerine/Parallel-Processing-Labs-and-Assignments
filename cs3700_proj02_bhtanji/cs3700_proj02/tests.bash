#! /bin/bash

echo "starting tests"

# data set for running tests
T_SET="2 4 8 10"
D_SET="100 1000 10000 100000 1000000"

# run tests with double for loop
for i in $T_SET
do	
	for j in $D_SET:
	do
		echo "threads: $i, dataSize: $j"
		./tMC $i $j	
		echo "	"
	done
done

