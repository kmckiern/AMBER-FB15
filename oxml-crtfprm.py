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
    
    # OpenMM Atom Types to Atom Class
    OAtAc = OrderedDict()
    
    # OpenMM Atom Classes to Masses
    OAcMass = OrderedDict()
    
    # OpenMM Residue-Atom Names to Atom Class
    AA_OAc = OrderedDict()
    
    # OpenMM Atom Class to Parameter Mapping
    # (vdW sigma and epsilon in AKMA)
    ONbPrm = OrderedDict()
    OBondPrm = OrderedDict()
    OAnglePrm = OrderedDict()
    ODihPrm = OrderedDict()
    OImpPrm = OrderedDict()
    
    Params = parameters.ParameterSet()
    
    # Stage 1 processing: Read in force field parameters
    for force in root:
        # Top-level tags in force field XML file are:
        # Forces
        # Atom types
        # Residues
        if force.tag == 'AtomTypes':
            for elem in force:
                OAtAc[elem.attrib['name']] = elem.attrib['class']
                mass = float(elem.attrib['mass'])
                if elem.attrib['class'] in OAcMass and mass != OAcMass[elem.attrib['class']]:
                    print "Atom class mass not consistent"
                    raise RuntimeError
                OAcMass[elem.attrib['class']] = mass
            # printcool_dictionary(OAtAc)
        # Harmonic bond parameters
        if force.tag == 'HarmonicBondForce':
            for elem in force:
                att = elem.attrib
                BC = (att['class1'], att['class2'])
                BCr = (att['class2'], att['class1'])
                acij = tuple(sorted(BC))
                if acij in OBondPrm:
                    print acij, "already defined in OBndPrm"
                    raise RuntimeError
                b = float(att['length'])*10
                k = float(att['k'])/10/10/2/4.184
                # sub C* and N* for CS and NS
                acij = sub_key(acij, CType99)
                OBondPrm[acij] = (k, b)
                # Pass information to ParmEd
                Params._add_bond(acij[0], acij[1], rk=k, req=b)
                # New Params object can't write frcmod files.
                # Params.bond_types[acij] = pmd.BondType(k, b)
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
                if acijk in OAnglePrm:
                    print acijk, "already defined in OAnglePrm"
                    raise RuntimeError
                t = float(att['angle'])*180/np.pi
                k = float(att['k'])/2/4.184
                acijk = sub_key(acijk, CType99)
                OAnglePrm[acijk] = (k, t)
                # Pass information to ParmEd
                Params._add_angle(acijk[0], acijk[1], acijk[2], thetk=k, theteq=t)
                # New Params object can't write frcmod files.
                # Params.angle_types[acijk] = pmd.AngleType(k, t)
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
                    if acijkl in ODihPrm:
                        print acijkl, "already defined in ODihPrm"
                        raise RuntimeError
                    # sub C* and N* for CS and NS
                    acijkl_0 = acijkl
                    acijkl = sub_key(acijkl, CType99)
                    ODihPrm[acijkl] = dprms
                    acijkl = acijkl_0
                elif elem.tag == 'Improper':
                    if acijkl in OImpPrm:
                        print acijkl, "already defined in OImpPrm"
                        raise RuntimeError
                    acijkl_0 = acijkl
                    acijkl = sub_key(acijkl, CType99)
                    OImpPrm[acijkl] = dprms
                    acijkl = acijkl_0
                    if len(dihedral_list) > 1:
                        print acijkl, "more than one interaction"
                        raise RuntimeError
                else:
                    raise RuntimeError
        # Nonbonded parameters
        if force.tag == 'NonbondedForce':
            for elem in force:
                sigma = float(elem.attrib['sigma'])*10
                epsilon = float(elem.attrib['epsilon'])/4.184
                atype = elem.attrib['type']
                aclass = OAtAc[atype]
                amass = OAcMass[aclass]
                msigeps = (amass, sigma, epsilon)
                if aclass in ONbPrm:
                    if ONbPrm[aclass] != msigeps:
                        print 'mass, sigma and epsilon for atom class %s not uniquely defined:' % aclass
                        print msigeps, ONbPrm[aclass]
                        raise RuntimeError
                else:
                    aclass = tuple([aclass])
                    ONbPrm[aclass] = (amass, sigma, epsilon)
    
        # Residue definitions
        if force.tag == 'Residues':
            resnode = force

    return OBondPrm, OAnglePrm, ODihPrm, OImpPrm, ONbPrm

# Create maps from parameter atom class tuples to parameter values
A99SB_BondPrm, A99SB_AnglePrm, A99SB_DihPrm, A99SB_ImpPrm, A99SB_NbPrm = xml_parameters(A99SB)

A99SBFB = ET.parse(args.ambfbxml)
A99SBFB_BondPrm, A99SBFB_AnglePrm, A99SBFB_DihPrm, A99SBFB_ImpPrm, A99SBFB_NbPrm = xml_parameters(A99SBFB)

# get elements in A99SBFB_*Prm not present in A99SB_*Prm
def prm_diff(prm_dict_new, prm_dict_old):
    new_params = list(set(prm_dict_new.keys()) - set(prm_dict_old.keys()))
    return new_params

BondPrm_new = prm_diff(A99SBFB_BondPrm, A99SB_BondPrm)
AnglePrm_new = prm_diff(A99SBFB_AnglePrm, A99SB_AnglePrm)
DihPrm_new = prm_diff(A99SBFB_DihPrm, A99SB_DihPrm)
ImpPrm_new = prm_diff(A99SBFB_ImpPrm, A99SB_ImpPrm)
NbPrm_new = prm_diff(A99SBFB_NbPrm, A99SB_NbPrm)

# circuitous formating precaution
# for i, nb in enumerate(NbPrm_new):
#     reformat = tuple([nb])
#     A99SBFB_NbPrm[reformat] = A99SBFB_NbPrm[nb]
#     NbPrm_new[i] = reformat

"""
STEP 1: rewrite CHARMM rtf file
- parse in file
    - map atom types to masses
    - map atom names to atom types on a per residue basis
- write out file
    - add new masses for each new atom type
    - replace atom type with new atom classes

CAtAm: map from atom types to atom masses
CAnAt: map from Atom names to atom types, indexed my residue
"""

CRTF = args.rtf
CAtAm = OrderedDict()
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
                updated = updated.replace(orig_anum, str(n_mass)).replace('!', '! FB15 ')
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
## handy functions ##
# for finding string indices for a list of keys
def find_all_indices(str_l, keys):
    indices = []
    for i, ele in enumerate(str_l):
        if ele in keys:
            indices.append(i)
    return indices
# stolen from LP
def almostequal(i, j, tol):
    return np.abs(i-j) < tol

## formatting functions ##
# format parameter to a desired precision
def format_param(param, c_len, padding):
    param = float(param)
    f = str(param) # '%f' % param
    if len(f) < c_len:
        if padding == '0':
            return f.ljust(c_len, padding)[:c_len]
        elif padding == ' ':
            return f.rjust(c_len, padding)[:c_len]
    if len(f) == c_len:
        return f[:c_len]
    else:
        return str(round(param, c_len))[:c_len]
# for when ACs have variable lengths
def format_AC(AC):
    if len(AC) == 1:
        return AC + ' '
    else:
        return AC
# given set of ACs, find ordering in parameter dict
def order_ACs(param_tuple, param_dict):
    pd_keys = param_dict.keys()
    if param_tuple not in pd_keys:
        pt_reverse = tuple(list(param_tuple)[::-1])
        if pt_reverse not in pd_keys:
            # discretionarily ignore some parameter tuples
            # print """Parameter tuple not found in parameter dictionary:  
            #     %s""" % ' '.join(param_tuple)
            return None
        else:
            param_tuple = pt_reverse
    return param_tuple
# format tuple of ACs into properly formatted string
def AC_string(param_tuple, spacing):
    n_param = len(param_tuple)
    AC_string = []
    for p, param in enumerate(param_tuple):
        param = format_AC(param)
        if p < n_param-1:
            for space in range(spacing):
                param += ' '
        AC_string.append(param)
    return ''.join(AC_string)

## Parameter replacement functions ##
# format force constant and equilibrium length/angle parameters
def fc_eq(section, new_p, pad):
    k_i, eq_i = new_p
    k_p, eq_p = section_data[section][-1]
    k_i = format_param(k_i, k_p, pad)
    eq_i = format_param(eq_i, eq_p, pad)
    return k_i, eq_i

# given a line, substitute a value for a substitute
def sub_val(line, val, sub):
    # don't replace if unnecessary
    if not almostequal(float(val), float(sub), 1e-8):
        if 'FB15' not in line:
            # update comment
            line = line.replace('!', '!  FB15 ')
        if len(val) < len(sub):
            val = format_param(val, len(sub), ' ')
        # pad so that replace doesn't sub any substrings
        val = ' ' + val + ' '
        sub = ' ' + sub + ' '
        line = line.replace(val, sub)
    return line

# updates bond and angle parameters
def update_ba(line, section, new_p, num_AC):
    k, eq = fc_eq(section, new_p, '0')
    line_iter = line.split()[:len(new_p) + 3]
    for ndx, val in enumerate(line_iter):
        if ndx < num_AC:
            continue
        # force constant
        elif ndx == num_AC:
            sub = k
        elif ndx == num_AC + 1:
            sub = eq
        else:
            continue
        line = sub_val(line, val, sub)
    return line

# updates dihedral and improper parameters
def update_di(line, section, new_p, mult, num_AC):
    new_p = new_p[mult]
    k, eq = fc_eq(section, new_p, '0')
    line_iter = line.split()[:7]
    for ndx, val in enumerate(line_iter):
        if ndx < num_AC:
            continue
        # force constant
        elif ndx == num_AC: 
            sub = k
        # multiplicity
        elif ndx == num_AC + 1:
            sub = str(mult)
        # minimum geometry
        elif ndx == num_AC + 2:
            sub = eq
        line = sub_val(line, val, sub)
    return line

# check nonbonded parameters
def update_nb(line, section, new_p, num_AC):
    l = line.split()
    # new params
    mass, sig, eps = new_p
    # convert to CHARMM convention
    sig = sig*(2.0**(1.0/6.0))/2.0
    # old params
    eps_0 = l[2]
    sig_0 = l[6]
    o_eps = -1.0 * float(eps_0)
    o_sig = float(sig_0)
    if almostequal(sig, o_sig, 1e-8) and almostequal(eps, o_eps, 1e-8):
        return line
    else:
        # hopefully only needed for params not included in original prm file
        print """Nonbonded parameters do not match for line
            %s""" % line
        print "Old parameters: %s, %s " % (o_sig, o_eps)
        print "New parameters: %s, %s " % (sig, eps)
        s, e = fc_eq(section, [sig, eps], '0')
        e = '-' + e
        for ndx, val in enumerate(l):
            if ndx < num_AC:
                continue
            elif ndx == 2:
                sub = e
            elif ndx == 6:
                sub = s
            else:
                continue
            line = sub_val(line, val, sub)
        return line

# wrapper function
def update_parameters(line, section, new_p, ptup, match_mult=True):
    l_ptup = len(ptup)
    if section == 'BONDS' or section == 'THETAS':
        l = [update_ba(line, section, new_p, l_ptup)]
    elif section == 'PHI' or section == 'IMPHI':
        l = []
        for mult in new_p:
            if match_mult:
                if line.split()[5] != str(mult):
                    continue
            l.append(update_di(line, section, new_p, mult, l_ptup))
    elif section == 'NONBONDED':
        l = [update_nb(line, section, new_p, l_ptup)]
    return l

# parameter type: (number of parameters, spacing between parameters,
#      (equilibrium *, force constant))
section_data = {'BONDS': (2, 3, (5, 5)), 'THETAS': (3, 3, (4, 6)), 'PHI': (4, 2, (10, 5)), 
    'IMPHI': (4, 2, (3, 5)), 'NONBONDED': (1, 0, (8, 8))}
# 'NONBONDED': (1, 0, (3, 9, 8, 3, 10, 8))
sections = section_data.keys()
# parameters with modified values
orig_params = {'BONDS': A99SB_BondPrm, 'THETAS': A99SB_AnglePrm, 
    'PHI': A99SB_DihPrm, 'IMPHI': A99SB_ImpPrm, 'NONBONDED': A99SB_NbPrm}
modified = orig_params.keys()
# map old bonded parameter dicts to new param dicts
new_params = {'BONDS': A99SBFB_BondPrm, 'THETAS': A99SBFB_AnglePrm, 
    'PHI': A99SBFB_DihPrm, 'IMPHI': A99SBFB_ImpPrm, 'NONBONDED': A99SBFB_NbPrm}
# map from section to exclusively unique new parameter values
new_AC_params = {'BONDS': BondPrm_new, 'THETAS': AnglePrm_new, 
    'PHI': DihPrm_new, 'IMPHI': ImpPrm_new, 'NONBONDED': NbPrm_new}

# Parse PRM
# determine which atom class quartets receive
# the sidechain-specific dihedral parameters.
def RewritePRM(prm):
    # open new prm for writing
    of = open(prm + '.new', 'w+')

    # track current section, initialize parameter values, and write record
    section = None
    p_0, p_f = (None, None)
    written = None

    lines = open(prm).readlines()
    for ln, line in enumerate(lines):
        ref_line = line

        if line.startswith('!'):
            of.write(line)
            continue

        l = line.split('!')[0].strip()
        s = l.split()
        if section != None:
            # read through params, replace if needed
            if len(s) > 0:
                param_tuple = tuple(s[:num_params])
                # if old AC is witnessed, save line number
                if bool(set(s) & set(old_at)):
                    relevant_lines.append(ln)
                # get correct order for substitution
                param_tuple = order_ACs(param_tuple, p_0)
                # update parameter values
                if param_tuple == None:
                    of.write(line)
                else:
                    new_p = p_f[param_tuple]
                    ls = update_parameters(line, section, new_p, param_tuple)
                    for l in ls:
                        data = AC_string(tuple(l.split()[:num_params]), param_spacing)
                        data_r = AC_string(tuple(reversed(l.split()[:num_params])), param_spacing)
                        l_r = l.replace(data, data_r)
                        # prevent duplicates
                        if l not in written:
                            of.write(l)
                            written.append(l)
                            written.append(l_r)

            elif len(s) == 0:
                # append new AC params to end of section
                AC_append = new_AC_params[section]
                for ptup in AC_append:
                    # default line is arbitrary
                    line = lines[relevant_lines[-1]]
                    # pick convenient line for substitution, if possible
                    forward = AC_string(ptup, param_spacing)
                    backward = AC_string(ptup[::-1], param_spacing)
                    for param in ptup:
                        if len(set([param]) & set(RevMap.keys())) > 0:
                            forward = forward.replace(param, RevMap[param])
                            backward = backward.replace(param, RevMap[param])
                    rls = []
                    for rl in relevant_lines:
                        check = lines[rl]
                        if forward in check or backward in check:
                            rls.append(check)
                            line = check
                    # remove comment
                    line = line.split('!')[0] + '!  FB15 \r\n'
                    # replace all of the ACs
                    line_ls = line.split()
                    param_tuple = tuple(line_ls[:num_params])
                    AC_old = AC_string(param_tuple, param_spacing)
                    AC_new = AC_string(ptup, param_spacing)
                    line = line.replace(AC_old, AC_new)
                    new_p = p_f[ptup]
                    # replace params
                    ls = update_parameters(line, section, new_p, ptup)
                    if section == 'PHI' and '6H' in ptup and 'X' in ptup:
                        IPython.embed()
                    for i, l in enumerate(ls):
                        data = AC_string(tuple(l.split()[:num_params]), param_spacing)
                        data_r = AC_string(tuple(reversed(l.split()[:num_params])), param_spacing)
                        l_r = l.replace(data, data_r)
                        # prevent duplicates
                        if l not in written:
                            if section == 'PHI':
                                l_ls = l.split()
                                l_mult = l_ls[5]
                                mult_match = False
                                for rl in rls:
                                    rl_mult = rl.split()[5]
                                    if l_mult == rl_mult:
                                        mult_match = True
                            of.write(l)
                            written.append(l)
                            written.append(l_r)

                # reinitialize tracker variables for next section
                section = None
                p_0, p_f = (None, None)
                written = None
                # write spacing between sections
                of.write(ref_line)

        elif len(s) > 0:
            of.write(line)
            if s[0] in sections:
                section = s[0]
                # record lines where we see old ACs
                relevant_lines = []
                # record written params to avoid duplicates
                written = []
                # record section information
                num_params, param_spacing, param_precision = section_data[section]
                p_0 = orig_params[section]
                p_f = new_params[section]

        else:
            of.write(line)

CPRM = args.prm
RewritePRM(CPRM)
