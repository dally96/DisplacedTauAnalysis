import uproot
import scipy
import matplotlib as mpl
import awkward as ak
import numpy as np
import math
import ROOT
import array
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
can = ROOT.TCanvas("can", "can")

def makeEffPlot(lepton, plot_type, dict_entries, xvar, bins, xmin, xmax, xbinsize, xunit, tot_arr, pass_arr, log_set, file):
  h_eff_dict = {}
  h_eff_num_dict = {}
  h_eff_den_dict = {}
  
  
  for ent in range(len(dict_entries)):
    h_eff_dict[dict_entries[ent]] = ROOT.TEfficiency("h_eff_"+xvar+"_"+dict_entries[ent], file.split(".")[0]+";"+xvar+" "+xunit+" ; Fraction of gen "+lepton+" which are recon'd", bins, xmin, xmax)
    h_eff_num_dict[dict_entries[ent]] = ROOT.TH1F("h_eff_num"+xvar+"_"+dict_entries[ent], file.split(".")[0]+";"+xvar+" "+xunit+" ; Number of gen "+lepton+" which are recon'd", bins, xmin, xmax)
    h_eff_den_dict[dict_entries[ent]] = ROOT.TH1F("h_eff_den"+xvar+"_"+dict_entries[ent], file.split(".")[0]+";"+xvar+" "+xunit+" ; Number of gen "+lepton, bins, xmin, xmax)

  for ent in range(len(dict_entries)):
    for i in range(bins):
      h_eff_dict[dict_entries[ent]].SetTotalEvents(i + 1, len(ak.flatten(tot_arr[(tot_arr > (xmin + (i * xbinsize))) & (tot_arr < (xmin + ((i + 1) * xbinsize)))])))
      h_eff_den_dict[dict_entries[ent]].SetBinContent(i + 1, len(ak.flatten(tot_arr[(tot_arr > (xmin + (i * xbinsize))) & (tot_arr < (xmin + ((i + 1) * xbinsize)))])))
      h_eff_dict[dict_entries[ent]].SetPassedEvents(i + 1, len(ak.flatten(pass_arr[ent][(pass_arr[ent] > (xmin + (i * xbinsize))) & (pass_arr[ent] < (xmin + ((i + 1) * xbinsize)))])))
      h_eff_num_dict[dict_entries[ent]].SetBinContent(i + 1, len(ak.flatten(pass_arr[ent][(pass_arr[ent] > (xmin + (i * xbinsize))) & (pass_arr[ent] < (xmin + ((i + 1) * xbinsize)))])))      

  can.SetLogy(log_set)
  l_eff = ROOT.TLegend()
  l_eff.SetBorderSize(0)
  for ent in range(len(dict_entries)):
    h_eff_dict[dict_entries[ent]].SetLineColor(ent + 1)
    l_eff.AddEntry(h_eff_dict[dict_entries[ent]], dict_entries[ent])
    if ent == 0:
      h_eff_dict[dict_entries[ent]].Draw()
      ROOT.gPad.Update()
      h_eff_dict[dict_entries[ent]].GetPaintedGraph().GetYaxis().SetRangeUser(0, 1.1)
    else:
      h_eff_dict[dict_entries[ent]].Draw("same")
  if len(dict_entries) > 1:
    l_eff.Draw()
  can.SaveAs("Stauto"+lepton+"_eff_"+plot_type+"_"+xvar+".pdf")

  l_eff_num = ROOT.TLegend()
  for ent in range(len(dict_entries)):
    h_eff_num_dict[dict_entries[ent]].SetLineColor(ent + 1)
    l_eff_num.AddEntry(h_eff_num_dict[dict_entries[ent]], dict_entries[ent])
    if ent == 0:
      h_eff_num_dict[dict_entries[ent]].Draw("hist")
    else:
      h_eff_num_dict[dict_entries[ent]].Draw("samehist")
  if len(dict_entries) > 1:
    l_eff_num.Draw()
  can.SaveAs("Stauto"+lepton+"_eff_"+plot_type+xvar+"_num.pdf")

  l_eff_den = ROOT.TLegend()
  for ent in range(len(dict_entries)):
    h_eff_den_dict[dict_entries[ent]].SetLineColor(ent + 1)
    l_eff_den.AddEntry(h_eff_den_dict[dict_entries[ent]], dict_entries[ent])
    if ent == 0:
      h_eff_den_dict[dict_entries[ent]].Draw("hist")
    else:
      h_eff_den_dict[dict_entries[ent]].Draw("samehist")
  if len(dict_entries) > 1:
    l_eff_den.Draw()
  can.SaveAs("Stauto"+lepton+"_eff_"+plot_type+xvar+"_den.pdf")

def makeResPlot(lepton, dict_entries, xvar, yvar, xrange, xmin, xmax, yresmin, yresmax, xbinsize, xvararr, yvardiff, xunit, yunit): 
  h_resVsX_y_dict = {}
  
  for ent in range(len(dict_entries)):
    h_resVsX_y_dict[dict_entries[ent]] = []
  
  for ent in range(len(dict_entries)):
    for i in range(len(xrange)):
      hist = ROOT.TH1F("h_resVs"+xvar+"_"+yvar+"_"+dict_entries[ent]+"_"+str(xrange[i]), ";"+yvar+"residual (reco - gen) "+yunit+";Number of leptons", 100, yresmin, yresmax)
      h_resVsX_y_dict[dict_entries[ent]].append(hist)

  #for ent in range(len(dict_entries)):
  #  for mu in range(len(ak.flatten(xvararr[ent]))):
  #    for x in range(len(xrange)):
  #      if ((ak.flatten(xvararr[ent])[mu] > (xmin + x * xbinsize)) & (ak.flatten(xvararr[ent])[mu] < (xmin + (x + 1) * xbinsize))):
  #        h_resVsX_y_dict[dict_entries[ent]][x].Fill(ak.flatten(yvardiff[ent])[mu])

  #for ent in range(len(dict_entries)):
  #  for x in range(len(xrange)):
  #    for mu in range(len(ak.flatten(xvararr[ent][(xvararr[ent] > (xmin + x * xbinsize)) & (xvararr[ent] < (xmin + (x + 1) * xbinsize))]))):
  #      h_resVsX_y_dict[dict_entries[ent]][x].Fill(ak.flatten(yvardiff[ent][(xvararr[ent] > (xmin + x * xbinsize)) & (xvararr[ent] < (xmin + (x + 1) * xbinsize))])[mu])

  for ent in range(len(dict_entries)):
    for x in range(len(xrange)):
      hist, bins, other = plt.hist(ak.flatten(yvardiff[ent][(xvararr[ent] > (xmin + x * xbinsize)) & (xvararr[ent] < (xmin + (x + 1) * xbinsize))]), bins=np.linspace(yresmin, yresmax, 101))
      for i in range(len(hist)):
        h_resVsX_y_dict[dict_entries[ent]][x].SetBinContent(i+1, hist[i])
  
  h2_resVsX_y_dict = {}
  for ent in range(len(dict_entries)):
    h2_resVsX_y_dict[dict_entries[ent]] = ROOT.TH1F("h2_resVsX_y_"+dict_entries[ent], file.split(".")[0]+";gen "+lepton+" "+xvar+" "+xunit+";"+yvar+" resolution "+yunit, len(xrange), xmin, xmax)

  for ent in range(len(dict_entries)):
    for i in range(len(xrange)): 
      h2_resVsX_y_dict[dict_entries[ent]].SetBinContent(i + 1, h_resVsX_y_dict[dict_entries[ent]][i].GetRMS())
      h2_resVsX_y_dict[dict_entries[ent]].SetBinError(i + 1, h_resVsX_y_dict[dict_entries[ent]][i].GetRMSError())

  l_resVsX_y = ROOT.TLegend()
  l_resVsX_y.SetFillStyle(0)

  for ent in range(len(dict_entries)):
    h2_resVsX_y_dict[dict_entries[ent]].SetMinimum(0)
    h2_resVsX_y_dict[dict_entries[ent]].SetMarkerStyle(20)
    h2_resVsX_y_dict[dict_entries[ent]].SetMarkerColor(ent + 1)
    l_resVsX_y.AddEntry(h2_resVsX_y_dict[dict_entries[ent]], dict_entries[ent])
    if ent == 0:
      h2_resVsX_y_dict[dict_entries[ent]].Draw("p")
    else:
      h2_resVsX_y_dict[dict_entries[ent]].Draw("psame")
  
  l_resVsX_y.Draw()
  can.SaveAs("Stauto"+lepton+"_resVs"+xvar+"_"+yvar+".pdf")
