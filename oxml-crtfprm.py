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

def sub_key(key, reference_dict):
    ref_keys = reference_dict.keys()
    # alter key so it has the CHARMM standard
    if len(set(key).intersection(ref_keys)) > 0:
        new_keys = []
        for AC in key:
            if AC in ref_keys:
                new_keys.append(reference_dict[AC])
            else:
                new_keys.append(AC)
        return tuple(new_keys)
    else:
        return key
def xml_parameters(xml_parsed):
    root = xml_parsed.getroot()
    # maps
    XBondPrm = OrderedDict()
    XAnglePrm = OrderedDict()
    XDihPrm = OrderedDict()
    XImpPrm = OrderedDict()
    Params = parameters.ParameterSet()
    # populate maps
    for force in root:
        # Harmonic bond parameters
        if force.tag == 'HarmonicBondForce':
            for elem in force:
                att = elem.attrib
                BC = (att['class1'], att['class2'])
                BCr = (att['class2'], att['class1'])
                acij = tuple(sorted(BC))
                acij = sub_key(acij, CType99)
                if acij in XBondPrm:
                    print acij, "already defined in XBondPrm"
                    raise RuntimeError
                b = float(att['length'])*10
                k = float(att['k'])/10/10/2/4.184
                XBondPrm[acij] = (b, k)
                # Pass information to ParmEd
                Params._add_bond(acij[0], acij[1], rk=k, req=b)
        # Harmonic angle parameters.  Same as for harmonic bonds.
        if force.tag == 'HarmonicAngleForce':
            for elem in force:
                att = elem.attrib
                AC = (att['class1'], att['class2'], att['class3'])
                ACr = (att['class3'], att['class2'], att['class1'])
                if AC[2] >= AC[0]:
                    acijk = tuple(AC)
                else:
                    acijk = tuple(ACr)
                acijk = sub_key(acijk, CType99)
                if acijk in XAnglePrm:
                    print acijk, "already defined in XAnglePrm"
                    raise RuntimeError
                t = float(att['angle'])*180/np.pi
                k = float(att['k'])/2/4.184
                XAnglePrm[acijk] = (t, k)
                # Pass information to ParmEd
                Params._add_angle(acijk[0], acijk[1], acijk[2], thetk=k, theteq=t)
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
                    acijkl = sub_key(acijkl, CType99)
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
    return XBondPrm, XAnglePrm, XDihPrm

A99SB_BondPrm, A99SB_AnglePrm, A99SB_DihPrm = xml_parameters(A99SB)
A99SBFB = ET.parse(args.ambfbxml)
A99SBFB_BondPrm, A99SBFB_AnglePrm, A99SBFB_DihPrm = xml_parameters(A99SBFB)

# get elements in A99SBFB_*Prm not present in A99SB_*Prm
def prm_diff(prm_dict_new, prm_dict_old):
    new_params = list(set(prm_dict_new.keys()) - set(prm_dict_old.keys()))
    # run thru symmetric difference to make sure it's truly new and not a substitution
    for np in new_params:
        if sub_key(np, RevMap) in prm_dict_old.keys():
            new_params.remove(np)
    return new_params

DihPrm_new = prm_diff(A99SBFB_DihPrm, A99SB_DihPrm)

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
                updated = updated.replace(orig_anum, str(n_mass)).replace('!', '! FB15')
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
# RewriteRTF(CRTF, l_rtf)

"""
STEP 2: rewrite CHARMM prm file
- read in lines with old atom types
- append each section with new atom classes
- add in new torsional quartet parameters
"""
# stolen from LP
def almostequal(i, j, tol):
    return np.abs(i-j) < tol
# for finding string indicies for a list of keys
def find_all_indices(str_l, keys):
    indices = []
    for i, ele in enumerate(str_l):
        if ele in keys:
            indices.append(i)
    return indices
# for formatting params
def format_param(param, c_len, padding):
    param = float(param)
    f = str(param)
    if len(f) < c_len:
        if padding == '0':
            return f.ljust(c_len, padding)[:c_len]
        elif padding == ' ':
            return f.rjust(c_len, padding)[:c_len]
    if len(f) == c_len:
        return f[:c_len]
    else:
        return str(round(param, c_len))[:c_len]
# for when ACs have different lengths
def format_ac(ac):
    if len(ac) == 1:
        return ac + ' '
    else:
        return ac
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
def format_quart(quart):
    formatted = []
    for i in quart:
        formatted.append(format_ac(i) + '  ')
    return ''.join(formatted)
# if AC quartet is new, sub terms from a similar reference line
def add_dihed(new_quart, ref_line, append_comment=False, preserve_quart=False):
    nq = format_quart(new_quart)
    old_quart = tuple(ref_line.split()[:4])
    oq = format_quart(old_quart)
    if preserve_quart:
        ref_line = ref_line.replace(oq, nq)
    if append_comment:
        ref_line = ref_line.replace('!', '!  FB15 NEW')
    for mult in A99SBFB_DihPrm[new_quart]:
        min_geo, k_phi = A99SBFB_DihPrm[new_quart][mult]
        min_geo = format_param(min_geo, 5, '0')
        k_phi = format_param(k_phi, 10, '0')
        sub_line = ref_line.split()[:7]
        for ndx, val in enumerate(sub_line):
            # already replaced ACs above
            if ndx < 4:
                continue
            # replace force constant
            elif ndx == 4: 
                sub = k_phi
            elif ndx == 5:
                sub = str(mult)
            elif ndx == 6:
                sub = min_geo
            # don't replace if unnecessary
            if not almostequal(float(val), float(sub), 1e-8):
                if len(val) < len(sub):
                    val = format_param(val, len(sub), ' ')
                val = ' ' + val + ' '
                sub = ' ' + sub + ' '
                ref_line = ref_line.replace(val, sub)
        return ref_line, nq
# substitute new torsional parameters for existing ACs
def sub_existing_dihed(s, line):
    dih_quart = get_dquart(s, A99SB_DihPrm.keys())
    if dih_quart == None:
        return line
    return add_dihed(dih_quart, line, preserve_quart=True)

CPRM = args.prm
# Parse PRM
# determine which atom class quartets receive
# the sidechain-specific dihedral parameters.
def RewritePRM(prm):
    # open new prm for writing
    of = open(prm + '.new', 'w+')
    section = None
    np = {'BONDS': 2, 'THETAS': 3, 'PHI': 4, 'IMPHI': 4, 'NONBONDED': 1}
    sections = np.keys()
    written = None
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
                # substitute old AC with new
                for newAC in new_at:
                    for p_line in relevant_lines:
                        new_line = lines[p_line]
                        # add FB comment
                        new_line = new_line.replace('!', '!  FB15')
                        # preserve whitespace
                        l = re.split(r'(\s+)', new_line)
                        # get indices of what needs to be replaced
                        at_replace = find_all_indices(l, old_at)
                        # TODO: use AA graphs to remove superfluous dihedral specifications
                        for swap in at_replace:
                            l[swap] = format_ac(newAC)
                            new_line_sub = ''.join(l)
                            quart = tuple(new_line_sub.split()[:4])
                            if quart not in written:
                                written.append(quart)
                                of.write(new_line_sub)
                            l = re.split(r'(\s+)', new_line)
                # add completely new dihedral parms
                if section == 'PHI':
                    # pick arbitrary line for substitution
                    reference = lines[relevant_lines[-1]]
                    for new_dih in DihPrm_new:
                        sub_line, quart = add_dihed(new_dih, reference, True, True)
                        if quart not in written:
                            written.append(quart)
                            of.write(sub_line)
                section = None
                # spacing between sections
                of.write(line)
            # store line numbers of lines with old ACs
            else:
                if bool(set(s) & set(old_at)):
                    relevant_lines.append(ln)
                if written != None:
                    ACs = tuple(line.split()[:np[section]])
                    if ACs not in written:
                        written.append(ACs)
                    else:
                        continue
            # if dihedral params, look up new param values
            if section == 'PHI':
                sub_line, ACs = sub_existing_dihed(s, line)
                line = sub_line
                # global substitution (useful for adding new entries)
                lines[ln] = sub_line
        elif len(s) > 0:
            if s[0] in sections:
                section = s[0]
                # record lines with old ACs
                relevant_lines = []
                # record written params to avoid duplicates
                written = []
        of.write(line)
RewritePRM(CPRM)

IPython.embed()
