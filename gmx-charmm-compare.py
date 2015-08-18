#!/usr/bin/env python

import os, sys, re
import numpy as np
import copy
from molecule import Molecule
from nifty import _exec, printcool_dictionary
import parmed
from collections import OrderedDict
from atest import Calculate_GMX, Calculate_AMBER, interpret_mdp
from ctest import Calculate_CHARMM
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import argparse
from parmed.charmm import CharmmPsfFile, CharmmCrdFile, CharmmParameterSet
import IPython

parser = argparse.ArgumentParser()
parser.add_argument('--ff', type=str, default='fb15', choices=['fb15', 'fb15ni', 'ildn'], help='Select a force field to compare between Gromacs and AMBER.')
parser.add_argument('--repo', type=str, help='path to AMBER-FB15 repo root')
args, sys.argv = parser.parse_known_args(sys.argv)

# path to root of repository
reporoot = args.repo

# Disable Gromacs backup files
os.environ["GMX_MAXBACKUP"] = "-1"

# Sections in the .rtp file not corresponding to a residue
rtpknown = ['atoms', 'bonds', 'bondedtypes', 'dihedrals', 'impropers']

# Mapping of amino acid atom names to new atom classes.  Mostly new
# atom classes for beta carbons but a few new gamma carbons are
# defined.  They are named using the number "6" (for carbon) and the
# one-letter amino acid code.  A few exceptions in the case of alternate
# protonation states.
NewAC = {"SER":{"CB":"6S"}, "THR":{"CB":"6T", "CG2":"6t"}, "LEU":{"CB":"6L"},
         "VAL":{"CB":"6V"}, "ILE":{"CB":"6I", "CG2":"6i"}, "ASN":{"CB":"6N"},
         "GLN":{"CB":"6Q", "CG":"6q"}, "ARG":{"CB":"6R"}, "HID":{"CB":"6H"},
         "HIE":{"CB":"6h"}, "HIP":{"CB":"6+"}, "TRP":{"CB":"6W"},
         "TYR":{"CB":"6Y"}, "PHE":{"CB":"6F"}, "GLU":{"CB":"6E", "CG":"6e"},
         "ASP":{"CB":"6D"}, "LYS":{"CB":"6K"}, "LYN":{"CB":"6k"},
         "PRO":{"CB":"6P"}, "CYS":{"CB":"6C"}, "CYM":{"CB":"6c"},
         "MET":{"CB":"6M"}, "ASH":{"CB":"6d"}, "GLH":{"CB":"6J", "CG":"6j"}}
for k, v in NewAC.items():
    NewAC["C"+k] = v
    NewAC["N"+k] = v

def RTPAtomNames(rtpfnm):
    answer = OrderedDict()
    for line in open(rtpfnm).readlines():
        s = line.split()
        # Do nothing for an empty line or a comment line.
        if len(s) == 0 or re.match('^ *;',line): 
            section = 'None'
            continue
        line = line.split(';')[0]
        s = line.split()
        if re.match('^ *\[.*\]',line):
            # Makes a word like "atoms", "bonds" etc.
            section = re.sub('[\[\] \n]','',line.strip())
            if section not in rtpknown:
                resname = section
        elif section == 'atoms':
            answer.setdefault(resname, []).append((s[0], s[1], float(s[2])))
    return answer
# Gromacs .rtp file with residue definitions
gmxrtp = reporoot + '/Params/GMX/amberfb15.ff/aminoacids.rtp'
# Gromacs atom names in residues
gmx_resatoms = RTPAtomNames(gmxrtp)

# load CHARMM residue information
def RTFAtomNames(rtf):
    answer = OrderedDict()
    # read in original rft
    lines = open(rtf).readlines()
    for line in lines:
        l = line.split('!')[0].strip()
        s = l.split()
        if len(s) == 0: continue
        specifier = s[0]
        # profile atom name to atom type on a per residue basis
        if specifier.startswith('RESI') or specifier.startswith('PRES'):
            spec, AA, AACharge = s
        # populate residue indexed map from atom names to atom types
        elif specifier == 'ATOM':
            spec, An, At, ACharge = s
            answer.setdefault(AA, []).append((An, At, float(ACharge)))
    return answer
# CHARMM rtf with residue definitions
chrm_rtf = reporoot + '/test/fb15.rtf'
chrm_resatoms = RTFAtomNames(chrm_rtf)

gmx_resnames = set(gmx_resatoms.keys())
chrm_resnames = set(chrm_resatoms.keys())
print "GMX resnames not in CHARMM:", ' '.join(sorted(list(gmx_resnames.difference(chrm_resnames))))
print "CHARMM resnames not in GMX:", ' '.join(sorted(list(chrm_resnames.difference(gmx_resnames))))


chrm_star = {'C*': 'CS', 'N*': 'NS'}

gmx_chrm_amap = OrderedDict()
chrm_gmx_amap = OrderedDict()
for resname in sorted(list(gmx_resnames.intersection(chrm_resnames))):
    for gmx_atom, chrm_atom in zip(gmx_resatoms[resname], chrm_resatoms[resname]):
        gmx_aname, gmx_atype, gmx_acharge = gmx_atom
        # Gromacs doesn't use new AMBER-FB15 atom types, so figure it out from the NewAC dictionary
        gmx_atype = NewAC.get(resname, {}).get(gmx_aname, gmx_atype)
        chrm_aname, chrm_atype, chrm_acharge = chrm_atom
        # The following code for mapping CHARMM -> AMBER atom names
        # assumes that the atoms in the CHARMM .rtp and AMBER .off
        # dictionaries come in the same order.  This is a sanity check
        # that checks to see if the atom type and atomic charge are
        # indeed the same.
        # this is because charmm is not case sensitive and needed to rename somethings.
        if chrm_atype[0] == '5':
            if chrm_atype[-1] == 'P':
                chrm_atype = '6+'
            else:
                chrm_atype = str(6) + chrm_atype[1:]
        if gmx_atype != chrm_atype:
            if gmx_atype not in chrm_star:
                print "Atom types don't match for", resname, gmx_atom, chrm_atom
                # raise RuntimeError
        if gmx_acharge != chrm_acharge:
            print "Atomic charges don't match for", resname, gmx_atom, chrm_atom
            # raise RuntimeError
        # Print the atoms that are renamed
        # if gmx_aname != chrm_aname:
        #     print "%s (AMBER) %s <-> %s (CHARMM)" % (resname, chrm_aname, gmx_aname)
        gmx_chrm_amap.setdefault(resname, OrderedDict())[gmx_aname] = chrm_aname
        chrm_gmx_amap.setdefault(resname, OrderedDict())[chrm_aname] = gmx_aname

chrm_atomnames = OrderedDict([(k, set(v.keys())) for k, v in chrm_gmx_amap.items()])
max_reslen = max([len(v) for v in chrm_atomnames.values()])

pdb_in = sys.argv[1]
# rewrite the gmx pdb using mdtraj
_exec('python ~/scripts/manip_proteins/mdt_rewrite_pdb.py --p0 %s --pf mdt_%s' % (pdb_in, pdb_in))

# run pdb thru/chrm charmming
_exec('python ~/local/charmming/parser_v3.py mdt_%s' % (pdb_in))

# Begin with an GMX-compatible PDB file
# Please ensure by hand :)
pdb = Molecule(pdb_in, build_topology=False)
gmx_pdb = copy.deepcopy(pdb)
del gmx_pdb.Data['elem']

# Convert to a GROMACS-compatible PDB file
# This mainly involves renaming atoms and residues,
# notably hydrogen names.

# List of atoms in the current residue
anameInResidue = []
anumInResidue = []

anirs = []

for i in range(gmx_pdb.na):
    # Rename the ions residue names to be compatible with Gromacs
    if gmx_pdb.resname[i] == 'Na+':
        gmx_pdb.resname[i] = 'NA'
    elif gmx_pdb.resname[i] == 'Cl-':
        gmx_pdb.resname[i] = 'CL'
    elif gmx_pdb.resname[i] in ['HOH','WAT']:
        gmx_pdb.resname[i] = 'SOL'
        if gmx_pdb.atomname[i] == 'O':
            gmx_pdb.atomname[i] = 'OW'
        elif gmx_pdb.atomname[i] == 'H1':
            gmx_pdb.atomname[i] = 'HW1'
        elif gmx_pdb.atomname[i] == 'H2':
            gmx_pdb.atomname[i] = 'HW2'
        else:
            print "Atom number %i unexpected atom name in water molecule: %s" % gmx_pdb.atomname[i]
            raise RuntimeError
    # Rename the ion atom names
    if gmx_pdb.atomname[i] == 'Na+':
        gmx_pdb.atomname[i] = 'Na'
    elif gmx_pdb.atomname[i] == 'Cl-':
        gmx_pdb.atomname[i] = 'Cl'
    # A peculiar thing sometimes happens with a hydrogen being named "3HG1" where in fact it shold be "HG13"
    if gmx_pdb.atomname[i][0] in ['1','2','3'] and gmx_pdb.atomname[i][1] == 'H':
        gmx_pdb.atomname[i] = gmx_pdb.atomname[i][1:] + gmx_pdb.atomname[i][0]
    anumInResidue.append(i)
    anameInResidue.append(gmx_pdb.atomname[i])
    if (i == (gmx_pdb.na - 1)) or ((gmx_pdb.chain[i+1], gmx_pdb.resid[i+1]) != (gmx_pdb.chain[i], gmx_pdb.resid[i])):
        # Try to look up the residue in the chrm_atomnames template
        mapped = False
        for k, v in chrm_atomnames.items():
            # charmm demands C-terminus written in particular way
            if 'OXT' in anameInResidue:
                anameInResidue[anameInResidue.index('O')] = 'OT1'
                anameInResidue[anameInResidue.index('OXT')] = 'OT2'
            if set(anameInResidue) == v:
                mapped = True
                break
        if mapped:
            for anum, aname in zip(anumInResidue, anameInResidue):
                gmx_pdb.atomname[anum] = chrm_gmx_amap[k][aname]
        anirs.append(anameInResidue)
        anameInResidue = []
        anumInResidue = []

# Write the Gromacs-compatible PDB file
gmx_pdbfnm = os.path.splitext(sys.argv[1])[0]+"-gmx.pdb"
gmx_pdb.write(gmx_pdbfnm)

gmx_ffnames = {'fb15':'amberfb15', 'fb15ni':'amberfb15ni', 'ildn':'amber99sb-ildn'}
amb_ffnames = {'fb15':'fb15', 'fb15ni':'fF15ni', 'ildn':'ff99SBildn'}
chrm_ffnames = {'fb15':'fb15', 'fb15ni':'fF15ni', 'ildn':'ff99SBildn'}

# Set up the system in Gromacs
_exec("pdb2gmx -ff %s -f %s > pdb2gmx.out" % (gmx_ffnames[args.ff], gmx_pdbfnm), stdin="1\n")

# need prmtop for openmm
# Set up the system in AMBER
amb_outpdbfnm = os.path.splitext(sys.argv[1])[0]+"-amb.pdb"

with open("stage.leap", 'w') as f:
    print >> f, """source leaprc.{choice}
pdb = loadpdb {pdbin}
savepdb pdb {pdbout}
saveamberparm pdb prmtop inpcrd
quit
""".format(choice = amb_ffnames[args.ff],
           pdbin = sys.argv[1], pdbout = amb_outpdbfnm)

_exec("tleap -f stage.leap")

# Set up the system in CHARMM
chrm_outpdbfnm = os.path.splitext(pdb_in)[0]+"-chrm.pdb"

pdb_pref = pdb_in.split('.')[0]
rele = ['mdt', pdb_pref + '-', '.pdb']
pdbs = [i for i in os.listdir('.') if all(x in i for x in rele)]
# ensure protein pdb is first in list
# via: http://stackoverflow.com/questions/2170900/get-first-list-index-containing-sub-string-in-python
def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1
protein_ndx = index_containing_substring(pdbs, 'pro.')
pdbs.insert(0, pdbs.pop(protein_ndx))

# for each pdb segment, generate crd and psf
for i in pdbs:
    i = '-' + i.split('protein-')[-1].split('.')[0]
    pdb_pref = pdb_in.split('.')[0] + i

    choice = chrm_ffnames[args.ff]
    sys = 'new_mdt_' + pdb_pref

    # run charmming protein pdb thru charmm
    with open(pdb_pref + '.inp', 'w') as f:
        print >> f, """* Run Segment Through CHARMM
*

! read topology and parameter files

read rtf card name "{choice}.rtf"
read param card name "{choice}.prm"

! Read sequence from the PDB coordinate file
open unit 1 card read name {sys}.pdb
read sequ pdb unit 1

! now generate the PSF and also the IC table (SETU keyword)
generate pro setup

rewind unit 1

! set bomlev to -1 to avois sying on lack of hydrogen coordinates
bomlev -1
read coor pdb unit 1
! them put bomlev back up to 0
bomlev 0

close unit 1

! prints out number of atoms that still have undefined coordinates.
define test select segid a-pro .and. ( .not. hydrogen ) .and. ( .not. init ) show end

ic para
ic fill preserve
ic build
! hbuild sele all end

energy

! write out the protein structure file (psf) and
! the coordinate file in pdb and crd format.

write psf card name {sys}-final.psf
* PSF
*

write coor card name {sys}-final.crd
* Coords
*

stop

    """.format(choice = choice, sys = sys)
    _exec('~/src/c37b2/exec/gnu/charmm < ' + pdb_pref + '.inp > ' + pdb_pref + '.out')

# merge psfs and crds
with open('join.inp', 'w') as f:
    print >> f, """* Append the PDBs
*

! read topology and parameter files

read rtf card name {choice}.rtf
read param card name {choice}.prm

! Read PSF from file
OPEN UNIT 1 FORM READ NAME {protein}-final.psf
READ PSF CARD UNIT 1
CLOSe UNIT 1

OPEN UNIT 1 FORM READ NAME {hetatms}-final.psf
READ PSF CARD APPEnd UNIT 1
CLOSe UNIT 1

! Read protein coord from the coordinate file
OPEN UNIT 1 CARD READ NAME {protein}-final.crd
READ COOR CARD UNIT 1 RESID
CLOSE UNIT 1

OPEN UNIT 1 CARD READ NAME {hetatms}-final.crd
READ COOR CARD APPEnd UNIT 1
CLOSE UNIT 1

! redo hydrogen coordinates for the complete structure
! COOR INIT SELE HYDRogen END
! HBUILd

ENERgy

OPEN UNIT 1 CARD WRITe NAME sys-full.psf
WRITe PSF CARD UNIT 1
* PSF
*

OPEN UNIT 1 CARD WRITe NAME {pdbout}
WRITe COOR PDB UNIT 1
* Coords
*

OPEN UNIT 1 CARD WRITe NAME sys-full.crd
WRITe COOR CARD UNIT 1
* Coords
*

stop

""".format(choice = chrm_ffnames[args.ff], 
           protein = pdbs[0].split('.pdb')[0], 
           hetatms = pdbs[1].split('.pdb')[0], 
           pdbout = chrm_outpdbfnm)
_exec('~/src/c37b2/exec/gnu/charmm < join.inp > join.out')

# Load the GROMACS output coordinate file
gmx_gro = Molecule("conf.gro", build_topology=False)

# Build a mapping of (atom numbers in original PDB -> atom numbers in Gromacs .gro)
anumInResidue = []
anameInResidue_pdb = []
anameInResidue_gro = []
# For an atom number in the original .pdb file, this is the corresponding atom number in the .gro file
anumMap_gro = []
for i in range(gmx_pdb.na):
    anumInResidue.append(i)
    anameInResidue_pdb.append(gmx_pdb.atomname[i])
    anameInResidue_gro.append(gmx_gro.atomname[i])
    if (i == (gmx_pdb.na - 1)) or ((gmx_pdb.chain[i+1], gmx_pdb.resid[i+1]) != (gmx_pdb.chain[i], gmx_pdb.resid[i])):
        if set(anameInResidue_pdb) != set(anameInResidue_gro):
            print set(anameInResidue_pdb).symmetric_difference(set(anameInResidue_gro))
            print "GROMACS PDB: Atom names do not match for residue %i (%s)" % (gmx_pdb.resid[i], gmx_pdb.resname[i])
            raise RuntimeError
        for anum, aname in zip(anumInResidue, anameInResidue_pdb):
            anumMap_gro.append(anumInResidue[anameInResidue_gro.index(aname)])
        anumInResidue = []
        anameInResidue_pdb = []
        anameInResidue_gro = []


# run pdb thru/chrm charmming
_exec('sed -i "s/NA+/Na+/g" *pdb')

chrm_pdb = Molecule(chrm_outpdbfnm, build_topology=False)
anumInResidue = []
anameInResidue_pdb = []
anameInResidue_chrm = []
# For an atom number in the original .pdb file, this is the corresponding atom number in the .gro file
anumMap_chrm = []
for i in range(pdb.na):
    # A peculiar thing sometimes happens with a hydrogen being named "3HG1" where in fact it shold be "HG13"
    if pdb.atomname[i][0] in ['1','2','3'] and pdb.atomname[i][1] == 'H':
        pdb.atomname[i] = pdb.atomname[i][1:] + pdb.atomname[i][0]
    anumInResidue.append(i)
    anameInResidue_pdb.append(pdb.atomname[i])
    anameInResidue_chrm.append(chrm_pdb.atomname[i])
    if (i == (pdb.na - 1)) or ((pdb.chain[i+1], pdb.resid[i+1]) != (pdb.chain[i], pdb.resid[i])):
        if set(anameInResidue_pdb) != set(anameInResidue_chrm):
            unmatched = set(anameInResidue_pdb).symmetric_difference(set(anameInResidue_chrm))
            print unmatched
            print "AMBER PDB: Atom names do not match for residue %i (%s)" % (pdb.resid[i], pdb.resname[i])
            terminal_o = ['OXT', 'O', 'OT1', 'OT2']
            if len(unmatched - set(terminal_o)) > 0:
                raise RuntimeError
            else:
                anameInResidue_pdb[anameInResidue_pdb.index('O')] = 'OT1'
                anameInResidue_pdb[anameInResidue_pdb.index('OXT')] = 'OT2'
        for anum, aname in zip(anumInResidue, anameInResidue_pdb):
            anumMap_chrm.append(anumInResidue[anameInResidue_chrm.index(aname)])
        anumInResidue = []
        anameInResidue_pdb = []
        anameInResidue_chrm = []


anumMap_chrm_gro = [None for i in range(pdb.na)]
for i, (anum_chrm, anum_gro) in enumerate(zip(anumMap_chrm, anumMap_gro)):
    # if i != anum_chrm:
    #     print i, anum_chrm, anum_gro
    if anum_chrm != anum_gro:
        print "Gromacs and CHARMM atoms don't have the same order:", "\x1b[91m", i, anum_chrm, anum_gro, "\x1b[0m"
        # raise RuntimeError
    # anumMap_chrm_gro[anum_chrm] = anum_gro

# Run a quick MD simulation in Gromacs 
_exec("grompp_d -f eq.mdp -o eq.tpr")
# You can set print_to_screen=True to see how fast it's going
_exec("mdrun_d -nt 1 -v -stepout 10 -deffnm eq", print_to_screen=False)
_exec("trjconv_d -s eq.tpr -f eq.trr -o eq.gro -ndec 9 -pbc mol -dump 1", stdin="0\n")

# Confirm that constraints are satisfied
OMM_eqgro = app.GromacsGroFile('eq.gro')
OMM_prmtop = app.AmberPrmtopFile('prmtop')
system = OMM_prmtop.createSystem(nonbondedMethod=app.NoCutoff)
integ = mm.VerletIntegrator(1.0*u.femtosecond)
plat = mm.Platform.getPlatformByName('Reference')
simul = app.Simulation(OMM_prmtop.topology, system, integ)
simul.context.setPositions(OMM_eqgro.positions)
simul.context.applyConstraints(1e-12)
state = simul.context.getState(getPositions=True)
pos = np.array(state.getPositions().value_in_unit(u.angstrom)).reshape(-1,3)
M = Molecule('eq.gro')
M.xyzs[0] = pos
M.write('constrained.gro')

# Gromacs calculation
GMX_Energy, GMX_Force, Ecomps_GMX = Calculate_GMX('constrained.gro', 'topol.top', 'shot.mdp')
GMX_Force = GMX_Force.reshape(-1,3)

# Print Gromacs energy components
printcool_dictionary(Ecomps_GMX, title="GROMACS energy components")

# Parse the .mdp file to inform ParmEd
defines, sysargs, mdp_opts = interpret_mdp('shot.mdp')

GmxGro = parmed.gromacs.GromacsGroFile.parse('constrained.gro')
# CHARMM calculation
parm = CharmmParameterSet('fb15.prm', 'fb15.rtf')
psf = CharmmPsfFile('sys-full.psf')
crd = CharmmCrdFile('sys-full.crd')
parm.box = GmxGro.box
parm.positions = GmxGro.positions

IPython.embed()
CHARMM_Energy, CHARMM_Force, Ecomps_CHARMM = Calculate_CHARMM(parm, psf, crd, sysargs, defines)
CHARMM_Force = CHARMM_Force.reshape(-1,3)
# Print CHARMM energy components
printcool_dictionary(Ecomps_CHARMM, title="CHARMM energy components")

def compare_ecomps(in1, in2):
    groups = [([['Bond'], ['BOND']]),
              ([['Angle'], ['ANGLE']]),
              ([['Proper-Dih.', 'Improper-Dih.'], ['DIHED']]),
              ([['LJ-14'], ['1-4 NB']]),
              ([['Coulomb-14'], ['1-4 EEL']]),
              ([['LJ-(SR)', 'Disper.-corr.'], ['VDWAALS']]),
              ([['Coulomb-(SR)', 'Coul.-recip.'], ['EELEC']]),
              ([['Potential'], ['EPTOT']])]
    t1 = []
    t2 = []
    v1 = []
    v2 = []
    for g1, g2 in groups:
        t1.append('+'.join(g1))
        v1.append(sum([v for k, v in in1.items() if k in g1]))
        t2.append('+'.join(g2))
        v2.append(sum([v for k, v in in2.items() if k in g2]))

    for i in range(len(t1)):
        print "%%%is" % (max([len(t) for t in t1]) + 3) % t1[i], "% 18.6f" % v1[i], "  --vs--",
        print "%%%is" % (max([len(t) for t in t2]) + 3) % t2[i], "% 18.6f" % v2[i],
        print "Diff: % 12.6f" % (v2[i]-v1[i])

compare_ecomps(Ecomps_GMX, Ecomps_CHARMM)

D_Force = GMX_Force - CHARMM_Force
D_FrcRMS = np.sqrt(np.mean([sum(k**2) for k in D_Force]))
D_FrcMax = np.sqrt(np.max(np.array([sum(k**2) for k in D_Force])))

print "RMS / Max Force Difference (kJ/mol/nm): % .6e % .6e" % (D_FrcRMS, D_FrcMax)
