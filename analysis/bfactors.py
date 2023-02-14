from pymol import cmd
import pandas as pd
import numpy as np
	
# Variable site tomfoolery
def map_data_to_sites(data, variable_sites_file):
    variable_sites = np.loadtxt(variable_sites_file, dtype='int32')
    all_sites = np.arange(0,variable_sites[-1]+1, dtype='int32')
    site_data = pd.DataFrame(index=all_sites)
    site_data.loc[:, 'values'] = 0
    site_data.loc[variable_sites, 'values'] = data
    return site_data['values'].values

def reassign_b_factors(prot, chain, b_factors):
    # Reassign the B-factors in PyMol
    for i, b_fact in enumerate(b_factors):
        cmd.alter("{0} and chain {1} and resi {2} and n. CA".format(prot, chain, i), "b={0}".format(b_fact))


def loadBfacts (mol,startaa=1,source="newBfactors.txt", visual="Y"):
	"""
	Replaces B-factors with a list of values contained in a plain txt file
	
	usage: loadBfacts mol, [startaa, [source, [visual]]]
 
	mol = any object selection (within one single object though)
	startaa = number of first amino acid in 'new B-factors' file (default=1)
	source = name of the file containing new B-factor values (default=newBfactors.txt)
	visual = redraws structure as cartoon_putty and displays bar with min/max values (default=Y)
 
	example: loadBfacts 1LVM and chain A
	"""
	obj=cmd.get_object_list(mol)[0]
	cmd.alter(mol,"b=-1.0")
	inFile = open(source, 'r')
	counter=int(startaa)
	bfacts=[]
	for line in inFile.readlines():	
		bfact=float(line)
		bfacts.append(bfact)
		cmd.alter("%s and resi %s and n. CA"%(mol,counter), "b=%s"%bfact)
		counter=counter+1
	if visual=="Y":
		cmd.show_as("cartoon",mol)
		cmd.cartoon("putty", mol)
		cmd.set("cartoon_putty_scale_min", min(bfacts),obj)
		cmd.set("cartoon_putty_scale_max", max(bfacts),obj)
		cmd.set("cartoon_putty_transform", 0,obj)
		cmd.set("cartoon_putty_radius", 0.2,obj)
		cmd.spectrum("b","rainbow", "%s and n. CA " %mol)
		cmd.ramp_new("count", obj, [min(bfacts), max(bfacts)], "rainbow")
		cmd.recolor()

cmd.extend("reassign_b_factors", reassign_b_factors)
