#!/bin/bash

for i in fb15.*; do cp $i bkup_${i}; done

sed -i "s/THETAS/ANGLES/g" fb15.*
sed -i "s/IMPHI/IMPROPER/g" fb15.*
sed -i "s/PHI/DIHEDRALS/g" fb15.*
