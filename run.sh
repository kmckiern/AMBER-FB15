#!/bin/bash

python oxml-crtfprm.py --rtf Params/CHARMM/klauda_amber_at/a99sb.rtf --prm Params/CHARMM/klauda_amber_at/a99sb.prm
cd Params/CHARMM/
mv klauda_amber_at/a99sb.rtf.new fb15.rtf
mv klauda_amber_at/a99sb.prm.new fb15.prm
bash /home/kmckiern/ff/AMBER-FB15/case_sensitivity.sh

cp fb15.* /home/kmckiern/ff/AMBER-FB15/test/bkup/
cd /home/kmckiern/ff/AMBER-FB15/test
rm *.*
cp bkup/* .
. ~/.bash_profile.lp
python ../gmx-charmm-compare.py protein.pdb --repo /home/kmckiern/ff/AMBER-FB15/ --ff fb15
