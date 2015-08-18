#!/bin/bash

for i in a99sb.*; do cp $i bkup_${i}; done

sed -i "s/THETAS/ANGLES/g" a99sb.*
sed -i "s/IMPHI/IMPROPER/g" a99sb.*
sed -i "s/PHI/DIHEDRALS/g" a99sb.*
