import glob
import ROOT
from array import array
from ROOT import TLatex

lumi = 138965.188103

ROOT.gROOT.LoadMacro("AtlasStyle.C")
#ROOT.gROOT.LoadMacro("/afs/cern.ch/work/l/lhoryn/public/displacedLepton/scripts/atlasstyle-00-04-02/AtlasStyle.C")
#ROOT.gROOT.LoadMacro("/afs/cern.ch/work/l/lhoryn/public/displacedLepton/scripts/atlasstyle-00-04-02/AtlasLabels.C");
ROOT.SetAtlasStyle()

colors = [ROOT.TColor.GetColor('#e23f59'), ROOT.TColor.GetColor('#eab508'), ROOT.TColor.GetColor('#7fb00c'), ROOT.TColor.GetColor('#8a15b1'),
                ROOT.TColor.GetColor('#57a1e8'), ROOT.TColor.GetColor('#e88000'), ROOT.TColor.GetColor('#1c587e'), ROOT.TColor.GetColor('#b8a91e'),
                ROOT.TColor.GetColor('#19cbaf'), ROOT.TColor.GetColor('#322a2d'),]

more_colors = [ROOT.kAzure+2, ROOT.kGreen+2, ROOT.kPink+4, ROOT.kOrange+10, ROOT.kOrange, ROOT.kSpring+7]

theme_colors = [ROOT.TColor.GetColor("#263EA8"),    # Dark Blue
                ROOT.TColor.GetColor("#60BC94"),    # Seafoam
                ROOT.TColor.GetColor("#F7AA02"),    # Saffron
                ROOT.TColor.GetColor("#E86C6D"),    # Coral
                ROOT.TColor.GetColor("#2D84C0"),    # ATLAS Blue
                ROOT.TColor.GetColor("#6A5ACD"),    # Slate blue
                ROOT.TColor.GetColor("#666666"),    # Slate blue
                ROOT.TColor.GetColor("#000000")]    # Slate blue

theme_colors2 = [ROOT.TColor.GetColor("#FFA900"),   # Sunny Orange
                ROOT.TColor.GetColor("#3629AC"),    # Dark Blue
                ROOT.TColor.GetColor("#2FC494"),    # Seafoam
                ROOT.TColor.GetColor("#F65866"),    # Pink
                ROOT.TColor.GetColor("#0E81C4"),    # Light Blue
                ]

# Make a custom color palette
NRGBs = 5
NCont = 99
stops = [ 0.0, 0.34, 0.51, 0.75, 1.00 ]
red = [  0.00, 0.00, 0.87, 1.00, 0.51 ]
green = [  0.00, 0.81, 1.00, 0.20, 0.00 ]
blue = [  0.51, 1.00, 0.12, 0.00, 0.00 ]
stopsArray = array('d', stops)
redArray = array('d', red)
greenArray = array('d', green)
blueArray = array('d', blue)
ROOT.TColor.CreateGradientColorTable(NRGBs, stopsArray, redArray, greenArray, blueArray, NCont)
ROOT.gStyle.SetNumberContours(NCont)

#label makers
def ATLAS_LABEL(x,y,color,tsize=0.04):
    l = TLatex()
    l.SetTextAlign(12)
    l.SetTextSize(tsize)
    l.SetNDC();
    l.SetTextFont(72);
    l.SetTextColor(color);
    l.DrawLatex(x,y,"ATLAS");

def myText(x,y,color,text, tsize=0.04):
    l =TLatex()
    l.SetTextAlign(12)
    l.SetTextSize(tsize)
    l.SetNDC();
    l.SetTextFont(42);
    l.SetTextColor(color);
    l.DrawLatex(x,y,text)


# Get maximum of a series of histograms
def getMax(hists):
    ymax = max([h.GetMaximum() for h in hists])
    return ymax

# Get a crosssection from cvmfs
def getXS(dsid):
    xs_file = "/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/dev/PMGTools/PMGxsecDB_mc16.txt"
    #xs_file = "xsec.txt"
    with open(xs_file, "r") as f:
        for line in f:
            columns = line.split()
            if columns[0] == str(dsid):
                return float(columns[2])#*float(columns[3])*float(columns[4])
    print "Couldn't find cross section for dsid", dsid, "so setting to 1."
    return 1


# Get event count
def getNEvents(ntup_file):

    files = glob.glob(ntup_file)

    n_events = 0
    for fname in files:
        try:
            f = ROOT.TFile(fname, "READ")
            h = f.Get("MetaData_EventCount")
            n_events +=  h.GetBinContent(3)
        except:
            print "Couldn't get events for file", fname
            continue
    return n_events

# Get one of our trees -- takes an input with wildcards
def getTree(file_name, tree_name="trees_SR_highd0_", verbose=False):
    if file_name.startswith("root://"): files = [file_name]
    else:
        files = glob.glob(file_name)
        if len(files)==0:
            print "Couldn't find any files matching string", file_name
            return -1

    print "Found %s files"%len(files)
    t = ROOT.TChain(tree_name)
    for f in files:
        if verbose: print "Adding", f
        t.Add(f)
    return t

# Get one of our trees -- takes a list of files
def getTreeFromList(files, tree_name="trees_SR_highd0_"):
    if len(files)==0:
        print "Couldn't find any files matching string", file_name
        return -1

    print "Found %s files"%len(files)
    t = ROOT.TChain(tree_name)
    for f in files: t.Add(f)
    return t

# Get a histogram with TTree Draw
def getHist(t, hname, value_name, binstr="", sel_name="", binning="", verbose=False):
    #hname = hname.replace("(","-").replace(")","-")
    hname_nobin = hname
    if binning != "":
        hname = hname + "(" + binning +")"
    print hname
    draw_cmd = '%s>>%s' % (value_name, hname)
    if binstr != "":
        draw_cmd += "(%s)" % (binstr)

    print draw_cmd
    print sel_name
    if sel_name != "": t.Draw( draw_cmd, '(%s)' % (sel_name) ,"COLZ")
    else: t.Draw( draw_cmd )
    if verbose: print "\"%s\", \"%s\""%(draw_cmd, sel_name)
    h = ROOT.gROOT.FindObject(hname_nobin)
    print h
    return h


# From a map of histograms (key as label name), plot them all on a canvas and save
def plotHistograms(h_map, save_name, xlabel="", ylabel="", interactive=False, logy=False, atltext=""):

    can = ROOT.TCanvas("can", "can")
    h_keys = h_map.keys()
    h_values = h_map.values()

    if len(h_keys)<1:
        print "No histograms found in h_map. Drawing blank canvas."
        return

    # Get maxes/mins of all hists
    maxy = 1.5*getMaximum(h_values)
    miny = 0
    if logy:
        maxy*=1e4
        miny = 1e-1

    # Draw histograms
    colorHists(h_values)
    h_map[h_keys[0]].GetXaxis().SetTitle(xlabel)
    h_map[h_keys[0]].GetYaxis().SetTitle(ylabel)
    h_values[0].SetMinimum(miny)
    h_values[0].SetMaximum(maxy)
    h_map[h_keys[0]].Draw("hist", 1/h_map[h_keys[0]].GetEntries())
  
    print 1/h_map[h_keys[0]].GetEntries()  
  
    if logy: ROOT.gPad.SetLogy(1)
    for k in h_keys[1:]:
        h_map[k].Draw("hist same", 1/h_map[h_keys[k]].GetEntries())

    leg = ROOT.TLegend(.66, .64, .8, .88)
    for k in h_keys:
        leg.AddEntry(h_map[k], k, "l")
    leg.Draw()
    if interactive: raw_input("...")

    if atltext != "":
        ROOT.ATLASLabel(0.2,0.85, atltext[0])
        text = ROOT.TLatex()
        text.SetNDC()
        text.SetTextSize(0.04)
        for i, t in enumerate(atltext[1:]):
            text.DrawLatex(0.2,0.85-0.07*(i+1), t)

    can.SaveAs(save_name)
    return

# From a map of efficiencies (key as label name), plot them all on a canvas and save
def plotEfficiencies(eff_map, save_name, xlabel="", ylabel=""):

    can = ROOT.TCanvas("can", "can")
    if len(eff_map)<1:
        print "No efficiency plots found in eff_map. Drawing blank canvas."
        return

    colorHists(eff_map.values())

    for i, k in enumerate(eff_map):
        print k
        if i==0:
            eff_map[k].Draw("alpe")
            eff_map[k].SetTitle(";%s;%s"%(xlabel,ylabel))
            ROOT.gPad.Update()

            eff_map[k].GetPaintedGraph().GetXaxis().SetTitle(xlabel)
            eff_map[k].GetPaintedGraph().GetYaxis().SetTitle(ylabel)
            eff_map[k].GetPaintedGraph().SetMinimum(0)
            eff_map[k].Draw("alpe")
            ROOT.gPad.Update()

        else: eff_map[k].Draw("lpe same")
        ROOT.gPad.Update()

    leg = ROOT.TLegend(.66, .64, .8, .88)
    for k in eff_map:
        leg.AddEntry(eff_map[k], k, "lp")
    leg.Draw()

    can.SaveAs(save_name)
    return

def colorHists(hists):
    i=0
    for h in hists:
        h.SetLineColor(more_colors[i])
        h.SetMarkerColor(more_colors[i])
        i += 1
        i %= len(more_colors)
    return

def drawTwoPads(top_hists, bottom_hist, savename="", logTop=False, logBottom=False):

    c = ROOT.TCanvas("c", "c")
    leg = ROOT.TLegend(.66, .6, .8, .88)

    # Get maxes/mins of all hists
    maxy_top = 1.5*getMaximum(top_hists)
    miny_top = 0
    if logTop:
        maxy_top*=1e2
        miny_top = 1e-3

    # Upper plot will be in pad1
    pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
    pad1.SetBottomMargin(0.1) # Upper and lower plot are close
    #pad1.SetGridx()         # Vertical grid
    pad1.SetGridy()         # horizontal grid
    pad1.Draw()             # Draw the upper pad: pad1
    pad1.cd()               # pad1 becomes the current pad

    top_hists[0].Draw("hist e")
    i=0
    for h in top_hists:
        h.SetMarkerSize(0)
        h.SetLineWidth(2)
        h.SetLineColor(more_colors[i])
        h.SetMaximum(maxy_top)
        h.SetMinimum(miny_top)
        h.Draw("hist e same")
        leg.AddEntry(h, h.GetName(), "l")
        i+=1
    leg.Draw()

    # lower plot will be in pad
    c.cd()          # Go back to the main canvas before defining pad2
    pad2 = ROOT.TPad("pad2", "pad2", 0, 0.05, 1, 0.3)
    pad2.SetTopMargin(0)
    pad2.SetBottomMargin(0.5)
    #pad2.SetGridx() # vertical grid
    pad2.SetGridy() # horizontal grid
    pad2.Draw()
    pad2.cd()       # pad2 becomes the current pad

    if not logBottom:
        bottom_hist.SetMinimum(0)
    bottom_hist.SetMarkerSize(0)
    bottom_hist.SetFillStyle(3003)
    bottom_hist.SetFillColor(ROOT.kBlue-9)
    bottom_hist.SetLineColor(ROOT.kBlue-9)
    bottom_hist.Draw("e hist")

    # Top plot settings
    top_hists[0].GetXaxis().SetLabelSize(0)
    top_hists[0].GetYaxis().SetTitleSize(.045)
    top_hists[0].GetYaxis().SetTitleOffset(1.1)
    top_hists[0].GetYaxis().SetLabelSize(.045)

    # Y axis ratio plot settings
    bottom_hist.GetYaxis().SetNdivisions(505)
    bottom_hist.GetYaxis().SetTitleSize(15)
    bottom_hist.GetYaxis().SetTitleFont(43)
    bottom_hist.GetYaxis().SetTitleOffset(1.55)
    bottom_hist.GetYaxis().SetLabelFont(43) # Absolute font size in pixel (precision 3)
    bottom_hist.GetYaxis().SetLabelSize(15)

    # X axis ratio plot settings
    bottom_hist.GetXaxis().SetTitleSize(15)
    #bottom_hist.GetXaxis().SetTitleSize(0)
    bottom_hist.GetXaxis().SetTitleFont(43)
    bottom_hist.GetXaxis().SetTitleOffset(7)
    bottom_hist.GetXaxis().SetLabelFont(43) # Absolute font size in pixel (precision 3)
    #bottom_hist.GetXaxis().SetLabelSize(15)
    bottom_hist.GetXaxis().SetLabelSize(10)
    bottom_hist.GetXaxis().SetLabelOffset(.02)

    if logTop: pad1.SetLogy()
    if logBottom: pad2.SetLogy()
    c.Update()
    #raw_input("...")
    if savename != "": c.SaveAs(savename)

def scaleHists(hists, scaleMax=False):
    for h in hists:
        #h.Sumw2()
        if scaleMax:
            if (h.GetMaximum()>0): h.Scale(1/h.GetMaximum())
        else:
            if (h.Integral()>0): h.Scale(1/h.Integral(0,h.GetNbinsX()+1))

def getMaximum(hists, isLog=False):
    max_factor = 1.1
    if isLog: max_factor = 10
    maximum = max(hists[i].GetMaximum() for i in xrange(len(hists)))
    return maximum*max_factor

def getMinimum(hists, isLog=False):
    minimum = min(hists[i].GetMinimum() for i in xrange(len(hists)))
    return minimum

def addOverflow(h):
    h_tmp = h.Clone(h.GetName()+"_tmp")
    h_tmp.Reset()
    last_bin = h_tmp.GetNbinsX()
    h_tmp.SetBinContent(last_bin, h.GetBinContent(last_bin+1))
    h_tmp.SetBinError(last_bin, h.GetBinError(last_bin+1))
    h.Add(h_tmp)
    return h

def addUnderflow(h):
    h_tmp = h.Clone(h.GetName()+"_tmp")
    h_tmp.Reset()
    last_bin = h_tmp.GetNbinsX()
    h_tmp.SetBinContent(1, h.GetBinContent(0))
    h_tmp.SetBinError(1, h.GetBinError(0))
    h.Add(h_tmp)
    return h

def addOverflowBin(h):
    name = h.GetName()
    h.SetName("old")
    nbins = h.GetNbinsX()
    h_tmp = ROOT.TH1F(name, h.GetTitle(), nbins+1, h.GetBinLowEdge(1), h.GetBinLowEdge(nbins+1) + h.GetBinWidth(nbins))
    h_tmp.GetXaxis().SetTitle(h.GetXaxis().GetTitle())
    h_tmp.GetYaxis().SetTitle(h.GetYaxis().GetTitle())

    #0 to nbins+2 to bring over and underflow bins along
    for b in xrange(0,nbins+2):
        h_tmp.SetBinContent(b, h.GetBinContent(b))

    h_tmp.SetBinContent(nbins+1, h.GetBinContent(nbins+1))
    h_tmp.SetBinError(nbins+1, h.GetBinError(nbins+1))
    return h_tmp

def addUnderflowBin(h):
    name = h.GetName()
    h.SetName("old")

    nbins = h.GetNbinsX()
    h_tmp = ROOT.TH1F(name, h.GetTitle(), nbins+1, h.GetBinLowEdge(1) - h.GetBinWidth(1), h.GetBinLowEdge(nbins+1))
    h_tmp.GetXaxis().SetTitle(h.GetXaxis().GetTitle())
    h_tmp.GetYaxis().SetTitle(h.GetYaxis().GetTitle())

    #0 to nbins+2 to bring over and underflow bins along
    for b in xrange(0,nbins+2):
        h_tmp.SetBinContent(b+1, h.GetBinContent(b))

    h_tmp.SetBinContent(1, h.GetBinContent(0))
    h_tmp.SetBinError(1, h.GetBinError(0))

    return h_tmp


def addOverflow2D(h):
    h_tmpx = h.Clone(h.GetName()+"_tmpx")
    h_tmpy = h.Clone(h.GetName()+"_tmpy")
    h_tmpxy = h.Clone(h.GetName()+"_tmpxy")
    h_tmpx.Reset()
    h_tmpy.Reset()
    h_tmpxy.Reset()
    last_bin_x = h.GetNbinsX()
    last_bin_y = h.GetNbinsY()
    # Add in y overflow
    for x in xrange(last_bin_x):
        h_tmpx.SetBinContent(x+1, last_bin_y, h.GetBinContent(x+1, last_bin_y+1))
        h_tmpx.SetBinError(x+1, last_bin_y, h.GetBinError(x+1, last_bin_y+1))
    # Add in x overflow
    for y in xrange(last_bin_y):
        h_tmpy.SetBinContent(last_bin_x, y+1, h.GetBinContent(last_bin_x+1, y+1))
        h_tmpy.SetBinError(last_bin_x, y+1, h.GetBinError(last_bin_x+1, y+1))
    # Add in x+y overflow
    h_tmpxy.SetBinContent(last_bin_x, last_bin_y, h.GetBinContent(last_bin_x+1, last_bin_y+1))
    h_tmpxy.SetBinError(last_bin_x, last_bin_y, h.GetBinError(last_bin_x+1, last_bin_y+1))
    h.Add(h_tmpx)
    h.Add(h_tmpy)
    h.Add(h_tmpxy)
    return h

#h1: numerator, h2: denomenator, h2_copy: h2 error, h3: ratio, h4: h3 error, h11 is a second h1-type hist, h33 the same for h3
def makeRatioPlot(h1, h2, xname, savename="", logy = False, ytitle="", doEfficiency = False, h11="", rmin=0.5, rmax=1.5):

    ROOT.gStyle.SetErrorX(0.5)
    c = ROOT.TCanvas("c_"+savename, "c_"+savename, 600, 450)

    if ytitle == "":
        ytitle = "ratio"
        if doEfficiency: ytitle = "efficiency"

    # Upper plot will be in pad1
    pad1 = ROOT.TPad("pad1_"+savename, "pad1_"+savename, 0, 0.3, 1, 1.0)
    pad1.SetBottomMargin(0) # Upper and lower plot are joined
    #pad1.SetGridx()         # Vertical grid
    #pad1.SetGridy()         # horizontal grid
    pad1.Draw()             # Draw the upper pad: pad1
    pad1.cd()               # pad1 becomes the current pad
    h2.SetStats(0)          # No statistics on upper plot
    maxy = 1.4*getMaximum([h1,h2])
    #if h11!="": maxy = 1.4*getMaximum([h1,h2,h11])
    if h11!="": maxy = getMaximum([h1,h2,h11])
    if logy:
        maxy = 100*maxy
        pad1.SetLogy()
    h2.SetMaximum(maxy)
    h2.SetMinimum(0)
    if logy: h2.SetMinimum(0.01)
    h2.Draw("hist")
    h1.Draw("hist e same")       # Draw h1
    if h11!="":
        h11.SetLineStyle(2)
        h11.Draw("hist e same")       # Draw h11

    #shows the error on h2
    h2_copy = h2.Clone("h2_copy"+savename)
    h2_copy.SetDirectory(0)
    h2_copy.SetFillStyle(3004)
    h2_copy.SetFillColor(ROOT.kBlack)
    h2_copy.SetMarkerSize(0)
    h2_copy.Draw("e2 same")

    # Do not draw the Y axis label on the upper plot and redraw a small
    # axis instead, in order to avoid the first label (0) to be clipped.
    h2.GetYaxis().SetLabelSize(0.)
    if logy:
        axis = ROOT.TGaxis( h1.GetXaxis().GetXmin(), 1, h1.GetXaxis().GetXmin(), maxy, 1, maxy, 510, "G")
    else:
        #axis = ROOT.TGaxis( h1.GetXaxis().GetXmin(), 1, h1.GetXaxis().GetXmin(), maxy, 1, maxy,510)
        axis = ROOT.TGaxis( h1.GetXaxis().GetXmin(), 0.001*maxy, h1.GetXaxis().GetXmin(), maxy, 0.001*maxy, maxy, 510)
    axis.SetLabelFont(43) # Absolute font size in pixel (precision 3)
    axis.SetLabelSize(17)

    # lower plot will be in pad
    c.cd()          # Go back to the main canvas before defining pad2
    pad2 = ROOT.TPad("pad2_"+savename, "pad2_"+savename, 0, 0.05, 1, 0.3)
    pad2.SetLogy(0)
    pad2.SetTopMargin(0)
    pad2.SetBottomMargin(0.4)
    pad2.SetGridx() # vertical grid
    pad2.SetGridy() # horizontal grid
    pad2.Draw()
    pad2.cd()       # pad2 becomes the current pad

    # Define the ratio plot
    h3 = h1.Clone("h3_"+savename)
    h3.SetDirectory(0)
    h3.SetLineColor(ROOT.kBlack)
    h3.SetStats(0)      # No statistics on lower plot
    #h3.Divide(h3, h2,1, 1,"B")
    h3.Divide(h3, h2)

    if h11!="":
        h33 = h11.Clone("h33_"+savename)
        h33.SetDirectory(0)
        h33.SetLineColor(ROOT.kBlack)
        #h33.Divide(h33, h2,1, 1,"B")
        h33.Divide(h33, h2)

    #Errors on ratio
    h4 = h2.Clone("h4_"+savename)     # Make hist with error for h2
    h4.SetDirectory(0)
    for x in xrange(h4.GetNbinsX()):
        if h2.GetBinContent(x+1) != 0:
            h3.SetBinError(x+1,h1.GetBinError(x+1)/h2.GetBinContent(x+1))
        if h11!="" and  h2.GetBinContent(x+1) != 0:
            h33.SetBinError(x+1,h11.GetBinError(x+1)/h2.GetBinContent(x+1))
        if h2.GetBinContent(x+1) != 0:
            h4.SetBinError(x+1,h2.GetBinError(x+1)/h2.GetBinContent(x+1))
        h4.SetBinContent(x+1,1)

    if h3.GetBinContent(h3.GetMinimumBin()) > 0: rmin=min(rmin, .95*h3.GetBinContent(h3.GetMinimumBin()))
    if h4.GetBinContent(h4.GetMinimumBin()) > 0: rmin=min(rmin, .95*h4.GetBinContent(h4.GetMinimumBin()))
    if h11!="":
        if h33.GetBinContent(h33.GetMinimumBin()) > 0: rmin=min(rmin, .95*h33.GetBinContent(h33.GetMinimumBin()))
    #if rmin>0.5: rmin=0.5
    rmax = max(rmax, 1.05*h3.GetBinContent(h3.GetMaximumBin()),1.05*h4.GetBinContent(h4.GetMaximumBin()))
    if h11!="": rmax = max(rmax, 1.05*h3.GetBinContent(h3.GetMaximumBin()),1.05*h4.GetBinContent(h4.GetMaximumBin()),1.05*h33.GetBinContent(h33.GetMaximumBin()))
    #if rmax<1.5: rmax=1.5

    if doEfficiency:
        rmax = 1.1
        rmin = 0.01

    h3.SetMinimum(rmin)  # Define Y ..
    h3.SetMaximum(rmax) # .. range

    h3.Draw("hist e")
    if h11!="": h33.Draw("hist e same")
    h4.Draw("e2 same")

    # h1 settings
    h1.SetLineColor(ROOT.kBlack)
    h1.SetLineWidth(2)
    h1.SetMarkerSize(0)
    if h11!="":
        h11.SetLineColor(ROOT.kBlack)
        h11.SetLineWidth(2)
        h11.SetMarkerSize(0)

    # Y axis h1 plot settings
    h1.GetYaxis().SetTitleSize(20)
    h1.GetYaxis().SetTitleFont(43)
    h1.GetYaxis().SetTitleOffset(1.55)

    # h2 settings
    h2.SetLineColor(theme_colors2[2])
    h2.SetFillColor(theme_colors2[2])
    #h2.SetLineColor(ROOT.kTeal-9)
    #h2.SetFillColor(ROOT.kTeal-9)
    h2.SetMarkerSize(0)
    h2.SetLineWidth(2)

    # Ratio plot (h3) settings
    h3.GetXaxis().SetTitle(xname)
    h3.SetTitle("") # Remove the ratio title
    h3.SetLineColor(ROOT.kBlack)
    h3.SetLineWidth(2)
    h3.SetMarkerSize(0)
    if h11!="":
        h33.GetXaxis().SetTitle(xname)
        h33.SetTitle("") # Remove the ratio title
        h33.SetLineColor(ROOT.kBlack)
        h33.SetLineWidth(2)
        h33.SetMarkerSize(0)
    h4.SetLineWidth(2)
    h4.SetFillStyle(3004)
    h4.SetFillColor(ROOT.kBlack)
    h4.SetLineColor(ROOT.kBlack)
    h4.SetMarkerSize(0)

    # Y axis ratio plot settings
    h3.GetYaxis().SetTitle(ytitle)
    h3.GetYaxis().SetNdivisions(303)
    h3.GetYaxis().SetTitleSize(20)
    h3.GetYaxis().SetTitleFont(43)
    h3.GetYaxis().SetTitleOffset(1.)
    h3.GetYaxis().SetLabelFont(43) # Absolute font size in pixel (precision 3)
    h3.GetYaxis().SetLabelSize(17)
    h3.GetYaxis().SetMaxDigits(3)

    # X axis ratio plot settings
    h3.GetXaxis().SetTitleSize(20)
    h3.GetXaxis().SetTitleFont(43)
    h3.GetXaxis().SetTitleOffset(3.7)
    h3.GetXaxis().SetLabelFont(43) # Absolute font size in pixel (precision 3)
    h3.GetXaxis().SetLabelSize(17)
    h3.GetXaxis().SetLabelOffset(.02)

    l = ROOT.TLine(h3.GetXaxis().GetXmin(),1.0,h3.GetXaxis().GetXmax(),1.0)
    l.SetLineColor(ROOT.kBlack)
    l.SetLineWidth(2)
    l.Draw("same")
    ROOT.gPad.Update()

    # Draw legend
    pad1.cd()
    ROOT.ATLASLabel(0.6,0.85, "Internal")
    leg = ROOT.TLegend(.6,.65,.8,.8)
    leg.AddEntry(h1, h1.GetTitle(), "l")
    if h11!="": leg.AddEntry(h11, h11.GetTitle(), "l")
    leg.AddEntry(h2, h2.GetTitle(), "f")
    leg.Draw("same")
    axis.Draw()

    ROOT.gPad.Update()
    c.Update()
    #raw_input("...")
    if savename != "": c.SaveAs(savename)

    return c

def getBinStr(entry):
    binstr = "%d,%f,%f"%(entry["nbins"],entry["xmin"],entry["xmax"])
    return binstr



def getHistFromEff(eff):

    h_out = eff.GetPassedHistogram().Clone()
    h_out.Clear()

    for i in xrange(h_out.GetNbinsX()+1):
        h_out.SetBinContent(i,eff.GetEfficiency(i))
        h_out.SetBinError(i,eff.GetEfficiencyErrorUp(i))
    return h_out

