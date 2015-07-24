#!/usr/bin/env python

import os, sys
from collections import OrderedDict
import numpy as np
import parmed as pmd
from parmed.amber import parameters
import xml.etree.ElementTree as ET
from nifty import printcool_dictionary
import argparse
import re
import IPython

parser = argparse.ArgumentParser(description='convert A99SB-FB15 xml to A99SB-FB15 CHARMM format')
parser.add_argument('--ambxml', type=str, help='path to OpemMM A99SB xml', 
    default='/home/leeping/src/OpenMM/wrappers/python/simtk/openmm/app/data/amber99sb.xml')
parser.add_argument('--ambfbxml', type=str, help='path to OpemMM A99SB xml', 
    default='/home/kmckiern/ff/AMBER-FB15/Params/OpenMM/amber-fb15.xml')
parser.add_argument('--rtf', type=str, help='path to CHARMM A99SB RTF file')
parser.add_argument('--prm', type=str, help='path to CHARMM A99SB PRM file')
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
A99SB = ET.parse(args.ambxml)
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

new_at = set(RevMap.keys())
old_at = set(RevMap.values())

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

def xml_torsions(xml_parsed):
    root = xml_parsed.getroot()
    # maps
    XDihPrm = OrderedDict()
    XImpPrm = OrderedDict()
    Params = parameters.ParameterSet()
    # populate maps
    for force in root:
        # Periodic torsion parameters.
        if force.tag == 'PeriodicTorsionForce':
            for elem in force:
                att = elem.attrib
                def fillx(strin):
                    if strin == "" : return "X"
                    else: return strin
                c1 = fillx(att['class1'])
                c2 = fillx(att['class2'])
                c3 = fillx(att['class3'])
                c4 = fillx(att['class4'])
                DC = (c1, c2, c3, c4)
                DCr = (c4, c3, c2, c1)
                if c1 > c4:
                    # Reverse ordering if class4 is alphabetically before class1
                    acijkl = DCr
                elif c1 < c4:
                    # Forward ordering if class1 is alphabetically before class4
                    acijkl = DC
                else:
                    # If class1 and class4 are the same, order by class2/class3
                    if c2 > c3:
                        acijkl = DCr
                    elif c3 > c2:
                        acijkl = DC
                    else:
                        acijkl = DC
                keylist = sorted([i for i in att.keys() if 'class' not in i])
                dprms = OrderedDict()
                for p in range(1, 7):
                    pkey = "periodicity%i" % p
                    fkey = "phase%i" % p
                    kkey = "k%i" % p
                    if pkey in keylist:
                        dprms[int(att[pkey])] = (float(att[fkey])*180.0/np.pi, float(att[kkey])/4.184)
                dprms = OrderedDict([(p, dprms[p]) for p in sorted(dprms.keys())])
                # ParmEd dihedral list
                dihedral_list = []
                for p in dprms.keys():
                    f, k = dprms[p]
                    dihedral_list.append(pmd.DihedralType(k, p, f, 1.2, 2.0, list=dihedral_list))
                    # Pass information to ParmEd
                    dtyp = 'normal' if (elem.tag == 'Proper') else 'improper'
                    Params._add_dihedral(acijkl[0], acijkl[1], acijkl[2], acijkl[3],
                                         pk=k, phase=f, periodicity=p, dihtype=dtyp)
                if elem.tag == 'Proper':
                    if acijkl in XDihPrm:
                        print acijkl, "already defined in XDihPrm"
                        raise RuntimeError
                    # alter key so it has the CHARMM standard
                    if len(set(acijkl).intersection(CType99.keys())) > 0:
                        chrm_acijkl = []
                        for AC in acijkl:
                            if AC in CType99.keys():
                                chrm_acijkl.append(CType99[AC])
                            else:
                                chrm_acijkl.append(AC)
                        acijkl = tuple(chrm_acijkl)
                    XDihPrm[acijkl] = dprms
                    # New Params object can't write frcmod files.
                    # Params.dihedral_types[acijkl] = dihedral_list
                elif elem.tag == 'Improper':
                    if acijkl in XImpPrm:
                        print acijkl, "already defined in XImpPrm"
                        raise RuntimeError
                    XImpPrm[acijkl] = dprms
                    if len(dihedral_list) > 1:
                        print acijkl, "more than one interaction"
                        raise RuntimeError
                    # New Params object can't write frcmod files.
                    # Params.improper_periodic_types[acijkl] = dihedral_list[0]
                else:
                    raise RuntimeError
        else:
            continue
    return XDihPrm, XImpPrm, Params

A99SB_DihPrm, A99SB_ImpPrm, A99SB_Params = xml_torsions(A99SB)
A99SBFB = ET.parse(args.ambfbxml)
A99SBFB_DihPrm, A99SBFB_ImpPrm, A99SBFB_Params = xml_torsions(A99SBFB)

# get elements in A99SBFB_DihPrm not present in A99SB_DihPrm
DihPrm_new = list(set(A99SBFB_DihPrm.keys()) - set(A99SB_DihPrm.keys()))
# also get overlapping keys
DihPrm_overlap = list(set(A99SBFB_DihPrm.keys()) - set(DihPrm_new))

"""
STEP 1: rewrite CHARMM rtf file
- parse in file
    - map atom types to masses
    - map atom names to atom types on a per residue basis
- write out file
    - add new masses for each new atom type
    - replace atom type with new atom classes
"""
CRTF = args.rtf
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
- read in lines with old atom types
- append each section with new atom classes
- add in new torsional quartet parameters
"""
# for formatting dihedral params
def format_param(param, c_len):
    f = str(param)
    if len(f) < c_len:
        return f.ljust(c_len, '0')[:c_len]
    if len(f) == c_len:
        return f
    else:
        return str(round(param, c_len))[:c_len]
# for finding string indicies for a list of keys
def find_all_indices(str_l, keys):
    indices = []
    for i, ele in enumerate(str_l):
        if ele in keys:
            indices.append(i)
    return indices
# for substituting dihedral parameters
def get_dquart(s, param_keys):
    dih_quart = tuple(s[:4])
    if dih_quart not in param_keys:
        # sometimes the ordering is backwards
        dih_quart_r = tuple(list(dih_quart)[::-1])
        if dih_quart_r not in param_keys:
            return None
        else:
            dih_quart = dih_quart_r
    return dih_quart
def sub_existing(s, line, stage):
    if stage == 'existing':
        param_old = A99SB_DihPrm
        param_new = A99SBFB_DihPrm
    if stage == 'new':
        param_old = A99SBFB_DihPrm
        param_new = A99SBFB_DihPrm
    dih_quart = get_dquart(s, param_old.keys())
    if dih_quart == None:
        return line
    dih_old = param_old[dih_quart]
    dih_new = param_new[dih_quart]
    for mult in dih_old.keys():
        # TODO: handle extra multiplicities
        if s[5] != str(mult):
            continue
        else:
            geo_old, phi_old = dih_old[mult]
            geo_new, phi_new = dih_new[mult]
            line = line.replace(format_param(geo_old, 5), format_param(geo_new, 5))
            line = line.replace(format_param(phi_old, 10), format_param(phi_new, 10))
    return line

CPRM = args.prm
# Parse PRM
# determine which atom class quartets receive
# the sidechain-specific dihedral parameters.
def most_similar(tuple_query, )
def RewritePRM(prm):
    # open new prm for writing
    of = open(prm + '.new', 'w+')
    section = None
    sections = ['BONDS', 'THETAS', 'PHI', 'IMPHI', 'NONBONDED']
    lines = open(prm).readlines()
    for ln, line in enumerate(lines):
        if line.startswith('!'):
            of.write(line)
            continue
        l = line.split('!')[0].strip()
        s = l.split()
        if section != None:
            # append new AC params to end of section
            if len(s) == 0:
                if section == 'PHI':
                    for new_dih in DihPrm_new:
                        reference = closest(new_dih, relevant_lines)
                        s = new_line.split()
                        new_line = sub_dihed(s, line, 'new')
                else:
                    for newAC in new_at:
                        # substitute old AT with new
                        for p_line in relevant_lines:
                            new_line = lines[p_line]
                            # add FB comment
                            new_line = new_line.replace('!', '!  FB')
                            # preserve white space
                            l = re.split(r'(\s+)', new_line)
                            # get indices of what needs to be replaced
                            at_replace = find_all_indices(l, old_at)
                            # TODO: use AA graphs to remove superfluous dihedral specifications
                            for swap in at_replace:
                                # TODO: handle if old and new AC are of variable length
                                l[swap] = newAC
                                new_line_sub = ''.join(l)
                                of.write(''.join(new_line_sub))
                                l = re.split(r'(\s+)', new_line)
                section = None
                of.write(line)
            # store line numbers of lines with old ACs
            else:
                if bool(set(s) & set(old_at)):
                    relevant_lines.append(ln)
            # if dihedral params, look up new param values
            if section == 'PHI':
                sub_line = sub_dihed(s, line, 'existing')
                line = sub_line
                # global substitution (useful for adding new entries)
                lines[ln] = sub_line
        elif len(s) > 0:
            if s[0] in sections:
                section = s[0]
                relevant_lines = []
        of.write(line)
RewritePRM(CPRM)

# IPython.embed()
