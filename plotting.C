#include "ROOT/RDataFrame.hxx"
#include "glob.h"
#include "ROOT/RDFHelpers.hxx"
#include "ROOT/RVec.hxx"
#include "TCanvas.h"
#include "TH1F.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TStyle.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"


void plotting() {
    // Load the ROOT file
    TFile* QCD_file        = TFile::Open("output_QCD.root");
    TFile* TTto2L2Nu_file  = TFile::Open("output_TTto2L2Nu.root");
    TFile* TTtoLNu2Q_file  = TFile::Open("output_TTtoLNu2Q.root");
    TFile* TTto4Q_file     = TFile::Open("output_TTto4Q.root");
    TFile* DYJetsToLL_file = TFile::Open("output_DYJetsToLL.root");

    TH1F* QCD_e_hist       = (TH1F*)QCD_file->Get("h_e_eta");
    TH1F* QCD_recoe_hist   = (TH1F*)QCD_file->Get("h_recoe_eta");
    TH1F* QCD_mu_hist      = (TH1F*)QCD_file->Get("h_mu_eta");
    TH1F* QCD_recomu_hist  = (TH1F*)QCD_file->Get("h_recomu_eta");
    TH1F* QCD_tau_hist     = (TH1F*)QCD_file->Get("h_tau_eta");
    TH1F* QCD_recotau_hist = (TH1F*)QCD_file->Get("h_recotau_eta"); 

    TH1F* TTto2L2Nu_e_hist       = (TH1F*)TTto2L2Nu_file->Get("h_e_eta");
    TH1F* TTto2L2Nu_recoe_hist   = (TH1F*)TTto2L2Nu_file->Get("h_recoe_eta");
    TH1F* TTto2L2Nu_mu_hist      = (TH1F*)TTto2L2Nu_file->Get("h_mu_eta");
    TH1F* TTto2L2Nu_recomu_hist  = (TH1F*)TTto2L2Nu_file->Get("h_recomu_eta");
    TH1F* TTto2L2Nu_tau_hist     = (TH1F*)TTto2L2Nu_file->Get("h_tau_eta");
    TH1F* TTto2L2Nu_recotau_hist = (TH1F*)TTto2L2Nu_file->Get("h_recotau_eta"); 

    TH1F* TTtoLNu2Q_e_hist       = (TH1F*)TTtoLNu2Q_file->Get("h_e_eta");
    TH1F* TTtoLNu2Q_recoe_hist   = (TH1F*)TTtoLNu2Q_file->Get("h_recoe_eta");
    TH1F* TTtoLNu2Q_mu_hist      = (TH1F*)TTtoLNu2Q_file->Get("h_mu_eta");
    TH1F* TTtoLNu2Q_recomu_hist  = (TH1F*)TTtoLNu2Q_file->Get("h_recomu_eta");
    TH1F* TTtoLNu2Q_tau_hist     = (TH1F*)TTtoLNu2Q_file->Get("h_tau_eta");
    TH1F* TTtoLNu2Q_recotau_hist = (TH1F*)TTtoLNu2Q_file->Get("h_recotau_eta"); 

    TH1F* TTto4Q_e_hist       = (TH1F*)TTto4Q_file->Get("h_e_eta");
    TH1F* TTto4Q_recoe_hist   = (TH1F*)TTto4Q_file->Get("h_recoe_eta");
    TH1F* TTto4Q_mu_hist      = (TH1F*)TTto4Q_file->Get("h_mu_eta");
    TH1F* TTto4Q_recomu_hist  = (TH1F*)TTto4Q_file->Get("h_recomu_eta");
    TH1F* TTto4Q_tau_hist     = (TH1F*)TTto4Q_file->Get("h_tau_eta");
    TH1F* TTto4Q_recotau_hist = (TH1F*)TTto4Q_file->Get("h_recotau_eta"); 

    TH1F* DYJetsToLL_e_hist       = (TH1F*)DYJetsToLL_file->Get("h_e_eta");
    TH1F* DYJetsToLL_recoe_hist   = (TH1F*)DYJetsToLL_file->Get("h_recoe_eta");
    TH1F* DYJetsToLL_mu_hist      = (TH1F*)DYJetsToLL_file->Get("h_mu_eta");
    TH1F* DYJetsToLL_recomu_hist  = (TH1F*)DYJetsToLL_file->Get("h_recomu_eta");
    TH1F* DYJetsToLL_tau_hist     = (TH1F*)DYJetsToLL_file->Get("h_tau_eta");
    TH1F* DYJetsToLL_recotau_hist = (TH1F*)DYJetsToLL_file->Get("h_recotau_eta"); 

    TCanvas* c = new TCanvas("c", "Histogram Canvas", 800, 600);
    
    TLegend* QCD_e_leg = new TLegend();
    QCD_e_hist->Draw();
    QCD_recoe_hist->Draw("SAME");
    QCD_recoe_hist->SetLineColor(kRed);
    QCD_e_leg->AddEntry("h_e_eta", "Gen electrons");
    QCD_e_leg->AddEntry("h_recoe_eta", "Reco electrons");
    QCD_e_leg->Draw();
    c->SaveAs("QCD_electron_eta_hist.pdf"); 

    TLegend* QCD_mu_leg = new TLegend();
    QCD_mu_hist->Draw();
    QCD_recomu_hist->Draw("SAME");
    QCD_recomu_hist->SetLineColor(kRed); 
    QCD_mu_leg->AddEntry("h_mu_eta", "Gen muons");
    QCD_mu_leg->AddEntry("h_recomu_eta", "Reco muons");
    QCD_mu_leg->Draw();
    c->SaveAs("QCD_muon_eta_hist.pdf"); 

    TLegend* QCD_tau_leg = new TLegend();
    QCD_tau_hist->Draw();
    QCD_recotau_hist->Draw("SAME");
    QCD_recotau_hist->SetLineColor(kRed); 
    QCD_tau_leg->AddEntry("h_tau_eta", "Gen tau");
    QCD_tau_leg->AddEntry("h_recotau_eta", "Reco tau");
    QCD_tau_leg->Draw();
    c->SaveAs("QCD_tau_eta_hist.pdf");

    TLegend* TTto2L2Nu_e_leg = new TLegend();
    TTto2L2Nu_e_hist->Draw();
    TTto2L2Nu_recoe_hist->Draw("SAME");
    TTto2L2Nu_recoe_hist->SetLineColor(kRed);
    TTto2L2Nu_e_leg->AddEntry("h_e_eta", "Gen electrons");
    TTto2L2Nu_e_leg->AddEntry("h_recoe_eta", "Reco electrons");
    TTto2L2Nu_e_leg->Draw();
    c->SaveAs("TTto2L2Nu_electron_eta_hist.pdf"); 

    TLegend* TTto2L2Nu_mu_leg = new TLegend();
    TTto2L2Nu_mu_hist->Draw();
    TTto2L2Nu_recomu_hist->Draw("SAME");
    TTto2L2Nu_recomu_hist->SetLineColor(kRed); 
    TTto2L2Nu_mu_leg->AddEntry("h_mu_eta", "Gen muons");
    TTto2L2Nu_mu_leg->AddEntry("h_recomu_eta", "Reco muons");
    TTto2L2Nu_mu_leg->Draw();
    c->SaveAs("TTto2L2Nu_muon_eta_hist.pdf"); 

    TLegend* TTto2L2Nu_tau_leg = new TLegend();
    TTto2L2Nu_tau_hist->Draw();
    TTto2L2Nu_recotau_hist->Draw("SAME");
    TTto2L2Nu_recotau_hist->SetLineColor(kRed); 
    TTto2L2Nu_tau_leg->AddEntry("h_tau_eta", "Gen tau");
    TTto2L2Nu_tau_leg->AddEntry("h_recotau_eta", "Reco tau");
    TTto2L2Nu_tau_leg->Draw();
    c->SaveAs("TTto2L2Nu_tau_eta_hist.pdf");

    TLegend* TTtoLNu2Q_e_leg = new TLegend();
    TTtoLNu2Q_e_hist->Draw();
    TTtoLNu2Q_recoe_hist->Draw("SAME");
    TTtoLNu2Q_recoe_hist->SetLineColor(kRed);
    TTtoLNu2Q_e_leg->AddEntry("h_e_eta", "Gen electrons");
    TTtoLNu2Q_e_leg->AddEntry("h_recoe_eta", "Reco electrons");
    TTtoLNu2Q_e_leg->Draw();
    c->SaveAs("TTtoLNu2Q_electron_eta_hist.pdf"); 

    TLegend* TTtoLNu2Q_mu_leg = new TLegend();
    TTtoLNu2Q_mu_hist->Draw();
    TTtoLNu2Q_recomu_hist->Draw("SAME");
    TTtoLNu2Q_recomu_hist->SetLineColor(kRed); 
    TTtoLNu2Q_mu_leg->AddEntry("h_mu_eta", "Gen muons");
    TTtoLNu2Q_mu_leg->AddEntry("h_recomu_eta", "Reco muons");
    TTtoLNu2Q_mu_leg->Draw();
    c->SaveAs("TTtoLNu2Q_muon_eta_hist.pdf"); 

    TLegend* TTtoLNu2Q_tau_leg = new TLegend();
    TTtoLNu2Q_tau_hist->Draw();
    TTtoLNu2Q_recotau_hist->Draw("SAME");
    TTtoLNu2Q_recotau_hist->SetLineColor(kRed); 
    TTtoLNu2Q_tau_leg->AddEntry("h_tau_eta", "Gen tau");
    TTtoLNu2Q_tau_leg->AddEntry("h_recotau_eta", "Reco tau");
    TTtoLNu2Q_tau_leg->Draw();
    c->SaveAs("TTtoLNu2Q_tau_eta_hist.pdf");

    TLegend* TTto4Q_e_leg = new TLegend();
    TTto4Q_e_hist->Draw();
    TTto4Q_recoe_hist->Draw("SAME");
    TTto4Q_recoe_hist->SetLineColor(kRed);
    TTto4Q_e_leg->AddEntry("h_e_eta", "Gen electrons");
    TTto4Q_e_leg->AddEntry("h_recoe_eta", "Reco electrons");
    TTto4Q_e_leg->Draw();
    c->SaveAs("TTto4Q_electron_eta_hist.pdf"); 

    TLegend* TTto4Q_mu_leg = new TLegend();
    TTto4Q_mu_hist->Draw();
    TTto4Q_recomu_hist->Draw("SAME");
    TTto4Q_recomu_hist->SetLineColor(kRed); 
    TTto4Q_mu_leg->AddEntry("h_mu_eta", "Gen muons");
    TTto4Q_mu_leg->AddEntry("h_recomu_eta", "Reco muons");
    TTto4Q_mu_leg->Draw();
    c->SaveAs("TTto4Q_muon_eta_hist.pdf"); 

    TLegend* TTto4Q_tau_leg = new TLegend();
    TTto4Q_tau_hist->Draw();
    TTto4Q_recotau_hist->Draw("SAME");
    TTto4Q_recotau_hist->SetLineColor(kRed); 
    TTto4Q_tau_leg->AddEntry("h_tau_eta", "Gen tau");
    TTto4Q_tau_leg->AddEntry("h_recotau_eta", "Reco tau");
    TTto4Q_tau_leg->Draw();
    c->SaveAs("TTto4Q_tau_eta_hist.pdf");

    TLegend* DYJetsToLL_e_leg = new TLegend();
    DYJetsToLL_e_hist->Draw();
    DYJetsToLL_recoe_hist->Draw("SAME");
    DYJetsToLL_recoe_hist->SetLineColor(kRed);
    DYJetsToLL_e_leg->AddEntry("h_e_eta", "Gen electrons");
    DYJetsToLL_e_leg->AddEntry("h_recoe_eta", "Reco electrons");
    DYJetsToLL_e_leg->Draw();
    c->SaveAs("DYJetsToLL_electron_eta_hist.pdf"); 

    TLegend* DYJetsToLL_mu_leg = new TLegend();
    DYJetsToLL_mu_hist->Draw();
    DYJetsToLL_recomu_hist->Draw("SAME");
    DYJetsToLL_recomu_hist->SetLineColor(kRed); 
    DYJetsToLL_mu_leg->AddEntry("h_mu_eta", "Gen muons");
    DYJetsToLL_mu_leg->AddEntry("h_recomu_eta", "Reco muons");
    DYJetsToLL_mu_leg->Draw();
    c->SaveAs("DYJetsToLL_muon_eta_hist.pdf"); 

    TLegend* DYJetsToLL_tau_leg = new TLegend();
    DYJetsToLL_tau_hist->Draw();
    DYJetsToLL_recotau_hist->Draw("SAME");
    DYJetsToLL_recotau_hist->SetLineColor(kRed); 
    DYJetsToLL_tau_leg->AddEntry("h_tau_eta", "Gen tau");
    DYJetsToLL_tau_leg->AddEntry("h_recotau_eta", "Reco tau");
    DYJetsToLL_tau_leg->Draw();
    c->SaveAs("DYJetsToLL_tau_eta_hist.pdf");

    QCD_file->Close();          
    TTto2L2Nu_file->Close(); 
    TTtoLNu2Q_file->Close(); 
    TTto4Q_file->Close();    
    DYJetsToLL_file->Close();
}


    

 
