"""
 splitROOT.py  -  description
 ---------------------------------------------------------------------------------
 split ROOT file tree into subset trees
 ---------------------------------------------------------------------------------
 copyright            : (C) 2019 Valentina Fioretti (INAF/OAS Bologna)
 email                : valentina.fioretti@inaf.it
 ----------------------------------------------
 Usage:
 python splitROOT.py old_filename new_filename tree_name N_subfiles
 ---------------------------------------------------------------------------------
 Parameters:
 - old_filename = name of the file (+path)
 - new_filename = name of the new file WITHOUT extension
 - tree_name = name of the tree to split
 - N_subfiles = number of files to create
 --------------------------------------------------------------------------------
 Usage example:
 python splitROOT.py old_file new_file sim 100
 ---------------------------------------------------------------------------------
 Caveats:
 None
 ---------------------------------------------------------------------------------
 Modification history:
 - 2019/04/19: creation date
"""

from ROOT import TFile, gDirectory, TTreeReader, TTree
import numpy as np

import sys

# Import the input parameters
arg_list = sys.argv
old_filename = arg_list[1]
new_filename = arg_list[2]
tree_name = arg_list[3]
N_subfiles = np.int(arg_list[4])

print '##########################################################################'
print '#                           splitROOT.py '
print '# ------------------------------------------------------------------------'
print '# copyright: (C) 2019 Valentina Fioretti (INAF/OAS Bologna)'
print '# email: valentina.fioretti@inaf.it'
print '# ------------------------------------------------------------------------'
print '# Usage:'
print '# python splitROOT.py old_filename new_filename tree_name N_subfiles'
print '# ------------------------------------------------------------------------'
print '# Parameters:'
print '# - old_filename = name of the file (+path)'
print '# - new_filename = name of the new file WITHOUT extension'
print '# - tree_name = name of the tree to split'
print '# - N_subfiles = number of files to create'
print '###########################################################################'
print '# Input filename: ', old_filename
print '# New filenames: ', new_filename+'_<i>.root'
print '# Selected tree to split: ', tree_name
print '# Number of subfiles to create: ', N_subfiles
print '###########################################################################'

input_file = TFile( old_filename )
input_tree = input_file.Get(tree_name)

NEntries = input_tree.GetEntriesFast()

n_infile = np.float(NEntries)/np.float(N_subfiles)
n_infile = np.int(n_infile)
n_lastfile = np.int(NEntries - n_infile*(N_subfiles-1))


file_counter = 0
start_counter = 0
for jf in xrange(N_subfiles):
  newfile = TFile(new_filename+'_'+str(file_counter)+'.root', "recreate")
  newtree = input_tree.CloneTree(0)
  if (jf < (N_subfiles - 1)):
    for je in xrange( n_infile ):
      input_tree.GetEntry( start_counter + je )
      newtree.Fill()
      if (je == n_infile-1): 
        newfile.Write()
    file_counter+=1
    start_counter+=n_infile
  else:
    for je in xrange( n_lastfile ):
      input_tree.GetEntry( start_counter + je )
      newtree.Fill()
      if (je == n_lastfile-1):  
        newfile.Write()

   
   
   
   
