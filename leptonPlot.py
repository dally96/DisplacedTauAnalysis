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
      h_eff_dict[dict_entries[ent]].SetTotalEvents(i + 1, len(ak.flatten(tot_arr[ent][(tot_arr[ent] > (xmin + (i * xbinsize))) & (tot_arr[ent] < (xmin + ((i + 1) * xbinsize)))])))
      h_eff_den_dict[dict_entries[ent]].SetBinContent(i + 1, len(ak.flatten(tot_arr[ent][(tot_arr[ent] > (xmin + (i * xbinsize))) & (tot_arr[ent] < (xmin + ((i + 1) * xbinsize)))])))
      h_eff_dict[dict_entries[ent]].SetPassedEvents(i + 1, len(ak.flatten(pass_arr[ent][(pass_arr[ent] > (xmin + (i * xbinsize))) & (pass_arr[ent] < (xmin + ((i + 1) * xbinsize)))])))
      h_eff_num_dict[dict_entries[ent]].SetBinContent(i + 1, len(ak.flatten(pass_arr[ent][(pass_arr[ent] > (xmin + (i * xbinsize))) & (pass_arr[ent] < (xmin + ((i + 1) * xbinsize)))])))      

  can.SetLogy(log_set)
  l_eff = ROOT.TLegend(0.1, 0.6, 0.25, 0.8)
  l_eff.SetBorderSize(0)
  l_eff.SetFillStyle(0)
  for ent in range(len(dict_entries)):
    h_eff_dict[dict_entries[ent]].SetLineColor(ent + 1)
    if (ent + 1 == 5):
      h_eff_dict[dict_entries[ent]].SetLineColor(ent + 2)
    l_eff.AddEntry(h_eff_dict[dict_entries[ent]], dict_entries[ent])
    if ent == 0:
      h_eff_dict[dict_entries[ent]].Draw()
      ROOT.gPad.Update()
      h_eff_dict[dict_entries[ent]].GetPaintedGraph().GetYaxis().SetRangeUser(0, 1.1)
    else:
      h_eff_dict[dict_entries[ent]].Draw("same")
  if len(dict_entries) > 1:
    l_eff.Draw()
  can.SaveAs("Stauto"+lepton+"_eff_"+plot_type+"_"+xvar+".png")

  l_eff_num = ROOT.TLegend()
  l_eff_num.SetBorderSize(0)
  l_eff_num.SetFillStyle(0)
  for ent in range(len(dict_entries)):
    h_eff_num_dict[dict_entries[ent]].SetLineColor(ent + 1)
    l_eff_num.AddEntry(h_eff_num_dict[dict_entries[ent]], dict_entries[ent])
    if ent == 0:
      h_eff_num_dict[dict_entries[ent]].Draw("hist")
    else:
      h_eff_num_dict[dict_entries[ent]].Draw("samehist")
  if len(dict_entries) > 1:
    l_eff_num.Draw()
  can.SaveAs("Stauto"+lepton+"_eff_"+plot_type+"_"+xvar+"_num.pdf")

  l_eff_den = ROOT.TLegend()
  l_eff_den.SetBorderSize(0)
  l_eff_den.SetFillStyle(0)
  for ent in range(len(dict_entries)):
    h_eff_den_dict[dict_entries[ent]].SetLineColor(ent + 1)
    l_eff_den.AddEntry(h_eff_den_dict[dict_entries[ent]], dict_entries[ent])
    if ent == 0:
      h_eff_den_dict[dict_entries[ent]].Draw("hist")
    else:
      h_eff_den_dict[dict_entries[ent]].Draw("samehist")
  if len(dict_entries) > 1:
    l_eff_den.Draw()
  can.SaveAs("Stauto"+lepton+"_eff_"+plot_type+"_"+xvar+"_den.pdf")

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

def makeEffPlotEta(lepton, dict_entries, xvar, xunit, tot_arr, etatot_arr, pass_arr, etapass_arr, pass_arr_end, etapass_arr_end, log_set, file, bins):
  h_eff_dict = {}
  h_eff_num_dict = {}
  h_eff_den_dict = {}

  ptBins = bins
  ptBins = array.array('d', ptBins)

  etaBins = [0, 1.479, 2.4]
  etaRegions = ["Barrel", "Endcaps"]

  for i in etaRegions:
    h_eff_dict[i] = {}
    h_eff_num_dict[i] = {}
    h_eff_den_dict[i] = {} 
  
  for reg in range(len(etaRegions)):
    for ent in range(len(dict_entries)):
      h_eff_dict[etaRegions[reg]][dict_entries[ent]] = ROOT.TEfficiency("h_eff_"+xvar+"_eta"+etaRegions[reg]+"_"+dict_entries[ent], file.split("_")[0]+file.split("_")[1]+file.split("_")[2]+file.split("_")[3]+file.split("_")[5]+" eta "+etaRegions[reg]+";"+xvar+" "+xunit+" ; Fraction of gen "+lepton+" which are recon'd", len(ptBins) - 1, ptBins)
      h_eff_num_dict[etaRegions[reg]][dict_entries[ent]] = ROOT.TH1F("h_eff_num"+xvar+"_eta"+etaRegions[reg]+"_"+dict_entries[ent], file.split("_")[0]+file.split("_")[1]+file.split("_")[2]+file.split("_")[3]+file.split("_")[5]+" eta "+etaRegions[reg]+";"+xvar+" "+xunit+" ; Number of gen "+lepton+" which are recon'd", len(ptBins) - 1, ptBins)
      h_eff_den_dict[etaRegions[reg]][dict_entries[ent]] = ROOT.TH1F("h_eff_den"+xvar+"_eta"+etaRegions[reg]+"_"+dict_entries[ent], file.split("_")[0]+file.split("_")[1]+file.split("_")[2]+file.split("_")[3]+file.split("_")[5]+" eta "+etaRegions[reg]+";"+xvar+" "+xunit+" ; Number of gen "+lepton, len(ptBins) - 1, ptBins)

  for reg in range(len(etaRegions)):
    for ent in range(len(dict_entries)):
      for i in range(len(ptBins) - 1):
        h_eff_dict[etaRegions[reg]][dict_entries[ent]].SetTotalEvents(i + 1, len(ak.flatten(tot_arr[(abs(etatot_arr) > etaBins[reg]) & (abs(etatot_arr) <= etaBins[reg + 1]) & (tot_arr > ptBins[i]) & (tot_arr <= ptBins[i + 1])])))
        h_eff_den_dict[etaRegions[reg]][dict_entries[ent]].SetBinContent(i + 1, len(ak.flatten(tot_arr[(abs(etatot_arr) > etaBins[reg]) & (abs(etatot_arr) <= etaBins[reg + 1]) & (tot_arr > ptBins[i]) & (tot_arr < ptBins[i + 1])])))
        if etaRegions[reg] ==  "Barrel":
          h_eff_dict[etaRegions[reg]][dict_entries[ent]].SetPassedEvents(i + 1, len(ak.flatten(pass_arr[ent][(abs(etapass_arr[ent]) > etaBins[reg]) & (abs(etapass_arr[ent]) <= etaBins[reg + 1]) & (pass_arr[ent] > ptBins[i]) & (pass_arr[ent] <= ptBins[i + 1])])))
          h_eff_num_dict[etaRegions[reg]][dict_entries[ent]].SetBinContent(i + 1, len(ak.flatten(pass_arr[ent][(abs(etapass_arr[ent]) > etaBins[reg]) & (abs(etapass_arr[ent]) <= etaBins[reg + 1]) & (pass_arr[ent] > ptBins[i]) & (pass_arr[ent] <= ptBins[i + 1])])))      
        if etaRegions[reg] == "Endcaps":
          h_eff_dict[etaRegions[reg]][dict_entries[ent]].SetPassedEvents(i + 1, len(ak.flatten(pass_arr_end[ent][(abs(etapass_arr_end[ent]) > etaBins[reg]) & (abs(etapass_arr_end[ent]) <= etaBins[reg + 1]) & (pass_arr_end[ent] > ptBins[i]) & (pass_arr_end[ent] <= ptBins[i + 1])])))
          h_eff_num_dict[etaRegions[reg]][dict_entries[ent]].SetBinContent(i + 1, len(ak.flatten(pass_arr_end[ent][(abs(etapass_arr_end[ent]) > etaBins[reg]) & (abs(etapass_arr_end[ent]) <= etaBins[reg + 1]) & (pass_arr_end[ent] > ptBins[i]) & (pass_arr_end[ent] <= ptBins[i + 1])])))      
          

  for reg in range(len(etaRegions)):
    can.SetLogy(log_set)
    l_eff = ROOT.TLegend(0.2, 0.6, 0.35, 0.8)
    l_eff.SetBorderSize(0)
    l_eff.SetFillStyle(0)
    for ent in range(len(dict_entries)):
      h_eff_dict[etaRegions[reg]][dict_entries[ent]].SetLineColor(ent + 1)
      if (ent >= 4):
        h_eff_dict[etaRegions[reg]][dict_entries[ent]].SetLineColor(ent + 2)
      l_eff.AddEntry(h_eff_dict[etaRegions[reg]][dict_entries[ent]], dict_entries[ent])
      if ent == 0:
        h_eff_dict[etaRegions[reg]][dict_entries[ent]].Draw()
        ROOT.gPad.Update()
        h_eff_dict[etaRegions[reg]][dict_entries[ent]].GetPaintedGraph().GetYaxis().SetRangeUser(0, 1.1)
      else:
        h_eff_dict[etaRegions[reg]][dict_entries[ent]].Draw("same")
    if len(dict_entries) > 1:
      l_eff.Draw()
    can.SaveAs("Stauto"+lepton+"_eff_"+xvar+"_eta"+etaRegions[reg]+".png")

    l_eff_num = ROOT.TLegend()
    l_eff_num.SetBorderSize(0)
    l_eff_num.SetFillStyle(0)
    for ent in range(len(dict_entries)):
      h_eff_num_dict[etaRegions[reg]][dict_entries[ent]].SetLineColor(ent + 1)
      if (ent + 1 == 5):
        h_eff_num_dict[etaRegions[reg]][dict_entries[ent]].SetLineColor(ent + 2)
      l_eff_num.AddEntry(h_eff_num_dict[etaRegions[reg]][dict_entries[ent]], dict_entries[ent])
      if ent == 0:
        h_eff_num_dict[etaRegions[reg]][dict_entries[ent]].Draw("hist")
      else:
        h_eff_num_dict[etaRegions[reg]][dict_entries[ent]].Draw("samehist")
    if len(dict_entries) > 1:
      l_eff_num.Draw()
    can.SaveAs("Stauto"+lepton+"_eff_"+xvar+"_eta"+etaRegions[reg]+"_num.pdf")

    l_eff_den = ROOT.TLegend()
    l_eff_den.SetBorderSize(0)
    l_eff_den.SetFillStyle(0)
    for ent in range(len(dict_entries)):
      h_eff_den_dict[etaRegions[reg]][dict_entries[ent]].SetLineColor(ent + 1)
      if (ent + 1 == 5):
        h_eff_den_dict[etaRegions[reg]][dict_entries[ent]].SetLineColor(ent + 2)
      l_eff_den.AddEntry(h_eff_den_dict[etaRegions[reg]][dict_entries[ent]], dict_entries[ent])
      if ent == 0:
        h_eff_den_dict[etaRegions[reg]][dict_entries[ent]].Draw("hist")
      else:
        h_eff_den_dict[etaRegions[reg]][dict_entries[ent]].Draw("samehist")
    if len(dict_entries) > 1:
      l_eff_den.Draw()
    can.SaveAs("Stauto"+lepton+"_eff_"+xvar+"_eta"+etaRegions[reg]+"_den.pdf")

