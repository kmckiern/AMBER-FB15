#!/usr/bin/env python

import os, sys
from collections import OrderedDict
import numpy as np
import parmed as pmd
from parmed.amber import parameters
import xml.etree.ElementTree as ET
from nifty import printcool_dictionary
import argparse
import IPython

parser = argparse.ArgumentParser(description='convert A99SB xml to A99SB-FB15 CHARMM format')
parser.add_argument('--a99sbxml', type=str, help='path to OpemMM A99SB xml', 
    default='/home/leeping/src/OpenMM/wrappers/python/simtk/openmm/app/data/amber99sb.xml')
parser.add_argument('--crtf0', type=str, help='path to CHARMM A99SB RTF file')
parser.add_argument('--cprm0', type=str, help='path to CHARMM A99SB PRM file')
args = parser.parse_args()

"""
STEP 0: read in AMBER99SB OpenMM xml file and create maps between residues and components
"""
# Atom types from parm99.dat
Type99 = ["C", "CA", "CB", "CC", "CD", "CK", "CM", "CN", "CQ", "CR", "CT", "CV", 
          "CW", "C*", "CY", "CZ", "C0", "H", "HC", "H1", "H2", "H3", "HA", "H4", 
          "H5", "HO", "HS", "HW", "HP", "HZ", "F", "Cl", "Br", "I", "IM", "IB", 
          "MG", "N", "NA", "NB", "NC", "N2", "N3", "NT", "N*", "NY", "O", "O2", 
          "OW", "OH", "OS", "P", "S", "SH", "CU", "FE", "Li", "IP", "Na", "K", 
          "Rb", "Cs", "Zn", "LP"]

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

# Parse the original AMBER99SB XML file.
A99SB = ET.parse(args.a99sbxml)
root = A99SB.getroot()
A99SB_AtAc = {}
A99SB_AnAc = {}
for force in root:
    if force.tag == 'AtomTypes':
        for atype in force:
            A99SB_AtAc[atype.attrib["name"]] = atype.attrib["class"]

# Build a mapping of Amino Acid / Atom Names -> Atom Types
for force in root:
    if force.tag == 'Residues':
        for res in force:
            A99SB_AnAc[res.attrib["name"]] = {}
            for atom in res:
                if atom.tag == 'Atom':
                    A99SB_AnAc[res.attrib["name"]][atom.attrib["name"]] = A99SB_AtAc[atom.attrib["type"]]

# For each amino acid, create the mapping of the new atom type (in FB15) back to the old atom type (in AMBER99SB).
RevMap = {}
for k1, v1 in NewAC.items():
    for k2, v2 in v1.items():
        if v2 in RevMap.keys():
            print "Atom type already in reverse map"
            raise RuntimeError
        RevMap[v2] = A99SB_AnAc[k1][k2]

# Mappings of AMBER99(SB) atom types to elements and hybridization
A99_Hyb = OrderedDict([("H",  ("H", "sp3")), ("HO", ("H", "sp3")), ("HS", ("H", "sp3")), 
                       ("H1", ("H", "sp3")), ("H2", ("H", "sp3")), ("H3", ("H", "sp3")),
                       ("H4", ("H", "sp3")), ("H5", ("H", "sp3")), ("HW", ("H", "sp3")), 
                       ("HC", ("H", "sp3")), ("HA", ("H", "sp3")), ("HP", ("H", "sp3")),
                       ("OH", ("O", "sp3")), ("OS", ("O", "sp3")), ("O",  ("O", "sp2")), 
                       ("O2", ("O", "sp2")), ("OW", ("O", "sp3")), ("CT", ("C", "sp3")),
                       ("CH", ("C", "sp3")), ("C2", ("C", "sp3")), ("C3", ("C", "sp3")),
                       ("C",  ("C", "sp2")), ("C*", ("C", "sp2")), ("CA", ("C", "sp2")),
                       ("CB", ("C", "sp2")), ("CC", ("C", "sp2")), ("CN", ("C", "sp2")), 
                       ("CM", ("C", "sp2")), ("CK", ("C", "sp2")), ("CQ", ("C", "sp2")),
                       ("CD", ("C", "sp2")), ("CE", ("C", "sp2")), ("CF", ("C", "sp2")), 
                       ("CP", ("C", "sp2")), ("CI", ("C", "sp2")), ("CJ", ("C", "sp2")),
                       ("CW", ("C", "sp2")), ("CV", ("C", "sp2")), ("CR", ("C", "sp2")), 
                       ("CA", ("C", "sp2")), ("CY", ("C", "sp2")), ("C0", ("Ca", "sp3")),
                       ("MG", ("Mg", "sp3")), ("N",  ("N", "sp2")), ("NA", ("N", "sp2")), 
                       ("N2", ("N", "sp2")), ("N*", ("N", "sp2")), ("NP", ("N", "sp2")),
                       ("NQ", ("N", "sp2")), ("NB", ("N", "sp2")), ("NC", ("N", "sp2")), 
                       ("NT", ("N", "sp3")), ("N3", ("N", "sp3")), ("S",  ("S", "sp3")),
                       ("SH", ("S", "sp3")), ("P",  ("P", "sp3")), ("LP", ("",  "sp3")), 
                       ("F",  ("F", "sp3")), ("CL", ("Cl", "sp3")), ("BR", ("Br", "sp3")),
                       ("I",  ("I",  "sp3")), ("FE", ("Fe", "sp3")), ("EP", ("",  "sp3")), 
                       ("OG", ("O", "sp3")), ("OL", ("O", "sp3")), ("AC", ("C", "sp3")),
                       ("EC", ("C", "sp3"))])

# Note components unique to CHARMM (need to be mapped uniquely later)
# Type99 atom types 
CType99 = {"C*": "CS", "N*": "NS"}
# Symmetric difference between A99SB xml and CHRM RTF residue names
AResOnly = ['LYN', 'ASH']
CResOnly = ['GUA', 'URA', 'THY', 'CYT', 'CYX', 'ALA', 'GLY', 'ACE', 'ADE', 'NME', 'TIP3', 'CIP']

new_at = set(RevMap.keys())
old_at = set(RevMap.values())

"""
STEP 1: rewrite CHARMM rtf file
- parse in file
    - map atom types to masses
    - map atom names to atom types on a per residue basis
- write out file
    - add new masses for each new atom type
    - replace atom type with new atom classes
"""
CRTF = args.crtf0
# atom types to atom masses
CAtAm = OrderedDict()
# atom name to atom type, per residue
CAnAt = OrderedDict()
def ParseRTF(rtf):
    # read in original rft
    lines = open(rtf).readlines()
    for line in lines:
        l = line.split('!')[0].strip()
        s = l.split()
        if len(s) == 0: continue
        specifier = s[0]
        # build map between atom types and masses
        if specifier == 'MASS':
            spec, Anum, At, mass = s
            CAtAm[At] = mass
        # profile atom name to atom type on a per residue basis
        elif specifier.startswith("RESI"):
            spec, AA, AACharge = s
            CAnAt[AA] = dict()
        # populate residue indexed map from atom names to atom types
        elif specifier == 'ATOM':
            spec, An, At, ACharge = s
            CAnAt[AA][An] = At
    return lines
l_rtf = ParseRTF(CRTF)
def RewriteRTF(rtf, lines):
    # open new rtf for writing
    of = open(rtf + '.new', 'w+')
    # get number of atom types
    n_at = len(CAtAm.keys())
    # log how many mass terms have been witnessed
    n_mass = 0
    # references for parsing
    at_lines = {}
    RID = None
    for ln, line in enumerate(lines):
        if n_mass < n_at:
            if line.startswith('MASS'):
                l = line.split()
                at = l[2]
                # take note of line number if atom type is relevant
                if at in old_at:
                    at_lines[at] = ln
                n_mass += 1
        # once original masses have been passed, write new entries
        elif n_mass == n_at:
            n_mass += 1
            for na, new_at in enumerate(RevMap):
                sub_line = lines[at_lines[RevMap[new_at]]]
                updated = sub_line.replace(RevMap[new_at], new_at)
                orig_anum = sub_line.split()[1]
                updated = updated.replace(orig_anum, str(n_mass)).replace('!', '! FB')
                of.write(updated)
                n_mass += 1
        elif line.startswith('RESI'):
            RID = line.split()[1]
        elif RID not in CResOnly and line.startswith('ATOM'):
            an = line.split()[1]
            at = line.split()[2]
            if an in NewAC[RID].keys():
                line = line.replace(at, NewAC[RID][an])
        of.write(line)
RewriteRTF(CRTF, l_rtf)

"""
STEP 2: rewrite CHARMM prm file
- parse in file
    - map atom types to masses
    - map atom names to atom types on a per residue basis
- write out file
    - add new masses for each new atom type
    - replace atom type with new atom classes
"""
CPRM = args.cprm0
# Parse PRM
# determine which atom class quartets receive
# the sidechain-specific dihedral parameters.
def RewritePRM(prm):
    # open new prm for writing
    of = open(prm + '.new', 'w+')
    section = None
    sections = ['BONDS', 'THETAS', 'PHI', 'IMPHI', 'NONBONDED']
    for ln, line in enumerate(open(prm).readlines()):
        l = line.split('!')[0].strip()
        s = l.split()
        if section != None:
            # append new AC params to end of section
            if len(s) == 0:
                for newAC in new_at:
                    # substitute old AT with new
                    for p_line in old_lines:
                        line = lines[p_line]
                        l = line.split()
                        at_replace = [x for x in enumerate(l) if x in old_at]
                        for swap in at_replace:
                            line = line.replace(swap, newAC)
                        of.write(line)
                section = None
                continue
            # store line numbers of param specs involving old ATs
            else:
                if bool(set(s) & set(old_at)):
                    old_lines.append(ln)
        if len(s) == 0: continue
        elif s[0] in sections:
            section = s[0]
            old_lines = []
        of.write(line)
RewritePRM(CPRM)
IPython.embed()
