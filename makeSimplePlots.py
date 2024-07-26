import ROOT
import os
#execfile("basic_plotting.py")
exec(open("basic_plotting.py").read())
import NanoAOD_Dict

ROOT.gROOT.SetBatch(1)

floc = "."
#fnames = [
    #"Stau_M_500_10mm_Summer22EE_NanoAOD.root",
    #"Stau_M_500_100mm_Summer22EE_NanoAOD.root"
    #"041223_FullSample_with-disTauTagScore.root"
#]

fnames = NanoAOD_Dict.Nano_Dict.values()

trees = {}
for fname in fnames:
    fullname = os.path.join(floc,fname)
    tname = "Events"
    mass = fname.split("_")[1]+"_"+fname.split("_")[2]+"_"+fname.split("_")[3]+"_"+fname.split("_")[4]
    trees[mass] = getTree(fullname, tname)


variables = {
        #"ele_eta": {"varname": "gen_ele_eta", "nbins": 100, "xmin": -10, "xmax": 10, "label": "Ele_Eta"},
        #"ele_pt": {"varname": "gen_ele_pt", "nbins": 100, "xmin": 0, "xmax": 2000, "label": "Ele_pT [GeV]"},
        #"ele_dxy": {"varname": "gen_ele_dxy", "nbins": 100, "xmin": -50, "xmax": 50, "label": "Ele_dxy [mm]"},
        #"had_pt" : {"varname": "gen_had_pt", "nbins": 100, "xmin": 0, "xmax": 2000, "label": "#tau_{h} pt [GeV]"},
        #"had_eta": {"varname": "gen_had_eta", "nbins": 100, "xmin": -10, "xmax": 10, "label": "Had_Eta"},
        #"had_dxy": {"varname": "gen_had_dxy", "nbins": 100, "xmin": -100, "xmax": 100, "label": "dxy [mm]"},
        "disTauTag_score1" : {"varname": "Jet_disTauTag_score1", "nbins": 200, "xmin": 0, "xmax" :1, "label": "score1"}
        }

hists = {}
for v in variables:

    hists[v] = {}
    for mass in trees:
        hists[v][mass] = getHist(trees[mass], "h_%s_%s"%(v,mass), variables[v]["varname"], getBinStr(variables[v]))
    plotHistograms(hists[v], "h_%s.pdf"%v, variables[v]["label"], "A.U.", logy = True)


