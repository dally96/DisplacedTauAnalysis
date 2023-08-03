import ROOT as rt
import numpy as np 
import os, sys 
#Make sure you have the tdrstyle.py file downloaded in this directory before you import. 
import tdrstyle_diff

#This sets the histogram style for you
tdrstyle_diff.setTDRStyle()


##Everything else here is how you would fill, draw, and save a histogram
# Here, put the file you want to open
file = "Stau_M_100_100mm_Summer22EE_NanoAOD.root"
# Open the file
file_open = rt.TFile.Open(file)
# Define the tree you want to use
Events = file_open.Get("Events")

#You can define the branch you want to look at, but in order to get the values contained in the branches you need to define the leaf (at least for NanoAOD's)
GenVisTau_pt = Events.GetBranch("GenVisTau_pt").GetLeaf("GenVisTau_pt")
#Number of events in the file 
nevt = Events.GetEntriesFast()
#Define your histogram
h_GVT_pt = rt.TH1F("h_pt", "p_{T} of Gen Vis Taus; p_{T} [GeV]; Number of taus", 100, 0, 500)

##Fill your histogram
for evt in range(nevt):
#Loop through your events, then the "GetEntry" function gets the values for a single event.
  Events.GetEntry(evt)
#Then loop through your the entries in each of your branch.
#For a branch, when you use GetEntry, it tells you the number of bytes stored for that event. Divide it by 4 to get how many entries this event has
  for val in range(int(Events.GetBranch("GenVisTau_pt").GetEntry(evt)/4)):
#Get the value of the branch with GetValue and fill your histogram
    h_GVT_pt.Fill(GenVisTau_pt.GetValue(val))
#Define the canvas on which your histogram will be drawn
can = rt.TCanvas("can", "can", 1000, 600)
#Draw the histogram. The HISTE option tells it to draw a histogram (HIST) with error bars (E)
h_GVT_pt.Draw("HISTE")
#Then, save your plot. 
can.SaveAs("with_cmsstyle.pdf")

