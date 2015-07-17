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
# Symmetric difference between A99SB xml and CHRM
AResOnly = ['LYN', 'ASH']
CResOnly = ['GUA', 'URA', 'THY', 'CYT', 'CYX', 'ALA', 'GLY', 'ACE', 'ADE', 'NME']

"""
STEP 1: parse in CHARMM AMBER99SB rtf file
"""
CRTF = args.crtf0
