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


// Function to list files matching a wildcard pattern
//std::vector<std::string> getFilesFromWildcard(const std::string& wildcard) {
//  std::vector<std::string> files;
//  
//  // Use the TSystem to list files matching the wildcard
//  TSystemDirectory dir(".", wildcard.c_str());
//  TList *filesList = dir.GetListOfFiles();
//  if (filesList) {
//    TSystemFile *file;
//    TIter next(filesList);
//    while ((file = (TSystemFile*)next())) {
//      const char *fileName = file->GetName();
//      if (!file->IsDirectory()) {
//        files.push_back((wildcard + fileName).c_str());
//      }
//    }
//  }
//  return files;
//}

std::vector<std::string> expandWildcard(const std::string& pattern) {
  glob_t glob_result;
  glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  std::vector<std::string> fileNames;
  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    fileNames.push_back(glob_result.gl_pathv[i]);
  }
  globfree(&glob_result);
  return fileNames;
}

void LepVeto() {
  gStyle->SetOptStat(0);


  std::string subdir = "/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v7/";
  std::string eossubdir = "/eos/uscms/store/";
  

  std::vector<std::string> fileName = {
                                       eossubdir + "group/lpcdisptau" + subdir + "QCD_PT-1000to1400_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "QCD_PT-120to170_TuneCP5_13p6TeV_pythia8/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "QCD_PT-120to170_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       eossubdir + "group/lpcdisptau" + subdir + "QCD_PT-1400to1800_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "QCD_PT-170to300_TuneCP5_13p6TeV_pythia8/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "QCD_PT-170to300_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       eossubdir + "group/lpcdisptau" + subdir + "QCD_PT-1800to2400_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       eossubdir + "group/lpcdisptau" + subdir + "QCD_PT-2400to3200_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       eossubdir + "group/lpcdisptau" + subdir + "QCD_PT-3200_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "QCD_PT-300to470_TuneCP5_13p6TeV_pythia8/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "QCD_PT-300to470_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "QCD_PT-470to600_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "QCD_PT-50to80_TuneCP5_13p6TeV_pythia8/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "QCD_PT-600to800_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "QCD_PT-800to1000_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "QCD_PT-80to120_TuneCP5_13p6TeV_pythia8/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/*.root",
                                       eossubdir + "user/fiorendi"    + subdir + "DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/*.root",
                            };     

  std::vector<std::string> sampleName = {
                                         "QCD_PT-1000to1400ext",
                                         "QCD_PT-120to170",
                                         "QCD_PT-120to170ext",
                                         "QCD_PT-1400to1800ext",
                                         "QCD_PT-170to300",
                                         "QCD_PT-170to300ext",
                                         "QCD_PT-1800to2400ext",
                                         "QCD_PT-2400to3200ext",
                                         "QCD_PT-3200ext",
                                         "QCD_PT-300to470",
                                         "QCD_PT-300to470ext",
                                         "QCD_PT-470to600ext",
                                         "QCD_PT-50to80",
                                         "QCD_PT-600to800ext",
                                         "QCD_PT-800to1000ext",
                                         "QCD_PT-80to120",
                                         "TTto2L2Nu",
                                         "TTto4Q",
                                         "TTtoLNu2Q",
                                         "DYJetsToLL",
                            };     
                                 
  for (int f = 0; f < (int)fileName.size(); f++) {
    std::cout << "Working with file " << fileName[f] << std::endl;
    ROOT::EnableImplicitMT();
    
    std::vector<std::string> files = expandWildcard(fileName[f]);
    std::vector<std::string> infiles;

    for (const auto& fname : files) {
      TFile *file = TFile::Open(fname.c_str());
      TTree *tree = dynamic_cast<TTree*>(file->Get("Events"));
      if (!tree) { 
        std::cerr << "No TTree 'Events' in: " << fname << std::endl;
        file->Close();
        delete file;
        continue;
        }   
      infiles.push_back(fname);
      file->Close();
      delete file;
    }      


    ROOT::RDataFrame df("Events", infiles);

    auto pt_cut       = [](float pt) { return pt > 20; };
    auto eta_cut      = [](float eta) { return std::abs(eta) < 2.4; };
    auto convVeto     = [](bool convVeto) { return convVeto == true; };
    auto leptonVeto   = [](const ROOT::RVec<float> &jet_eta,
                                 const ROOT::RVec<float> &jet_phi,
                                 const ROOT::RVec<float> &lep_eta,
                                 const ROOT::RVec<float> &lep_phi) {
      ROOT::RVec<bool> matched;
      for (size_t i = 0; i < jet_eta.size(); ++i) {
        bool isMatched = false;
        for (size_t j = 0; j < lep_eta.size(); ++j) {
          float dR = ROOT::VecOps::DeltaR(jet_eta[i], jet_phi[i], lep_eta[j], lep_phi[j]);
          if (dR > 0.4) {
            isMatched = true;
            break;
          }
        }
        matched.push_back(isMatched);
      }
      return matched;
    }; 

    auto getJetDxy    = [](const ROOT::RVec<float> &jet_pt,
                           const ROOT::RVec<float> &pfcands_pt,
                           const ROOT::RVec<float> &pfcands_dxy,
                           const ROOT::RVec<int> &pfcands_charge,
                           const ROOT::RVec<int> &jetpfcands_jetidx,
                           const ROOT::RVec<int> &jetpfcands_pfcandsidx) {
      ROOT::RVec<float> dxy;
      dxy.reserve(jet_pt.size());
      
      for (int j = 0; j < (int)jet_pt.size(); j++) {
        float max_pf_pt  = -9999.;
        float max_pf_dxy = -9999.;
        
        for (int jpc = 0; jpc < (int)jetpfcands_jetidx.size(); jpc++){ 
          if ((int)jetpfcands_jetidx[jpc] != j) { continue; }
          auto pfCandIdx = jetpfcands_pfcandsidx[jpc];
          if (pfcands_charge[pfCandIdx] == 0) { continue; }
          if (pfcands_pt[pfCandIdx] > max_pf_pt) { 
            max_pf_pt  = pfcands_pt[pfCandIdx]; 
            max_pf_dxy = pfcands_dxy[pfCandIdx];
          }
        }
        dxy.emplace_back(max_pf_dxy);
      }
      return dxy;
    };

    auto new_df = df.Define("Jet_dxy", getJetDxy, {"Jet_pt", "PFCands_pt", "PFCands_d0", "PFCands_charge", "JetPFCands_jetIdx", "JetPFCands_pFCandsIdx"});

    auto filtered_Jet = new_df.Define("jet_mask", "Jet_pt > 20 && abs(Jet_eta) < 2.4")
                        .Define("jet_filtered_pt"   , "Jet_pt[jet_mask]")
                        .Define("jet_filtered_score", "Jet_disTauTag_score1[jet_mask]")
                        .Define("jet_filtered_dxy",   "Jet_dxy[jet_mask]")
                        .Define("jet_filtered_eta",   "Jet_eta[jet_mask]")
                        .Define("jet_filtered_phi",   "Jet_phi[jet_mask]");

    auto noId_Jet     = filtered_Jet.Define("photon_veto",   leptonVeto, {"jet_filtered_eta", "jet_filtered_phi", "Photon_eta", "Photon_phi"})
                                    .Define("electron_veto", leptonVeto, {"jet_filtered_eta", "jet_filtered_phi", "Electron_eta", "Electron_phi"}) 
                                    .Define("muon_veto",     leptonVeto, {"jet_filtered_eta", "jet_filtered_phi", "Muon_eta", "Muon_phi"})
                                    .Define("dismuon_veto",  leptonVeto, {"jet_filtered_eta", "jet_filtered_phi", "DisMuon_eta", "DisMuon_phi"})
                                    .Define("all_lepton_veto", "photon_veto && electron_veto && muon_veto && dismuon_veto")
                                    .Define("noId_jet_pt", "jet_filtered_pt[all_lepton_veto]")
                                    .Define("noId_jet_score", "jet_filtered_score[all_lepton_veto]")
                                    .Define("noId_jet_dxy", "jet_filtered_dxy[all_lepton_veto]");

    auto Id_Jet       = filtered_Jet.Define("photon_cuts",   "Photon_pt   > 20 && abs(Photon_eta)   < 2.4 && Photon_electronVeto")
                                    .Define("electron_cuts", "Electron_pt > 20 && abs(Electron_eta) < 2.4 && Electron_convVeto  ")
                                    .Define("muon_cuts",     "Muon_pt     > 20 && abs(Muon_eta)     < 2.4 && Muon_looseId       ")
                                    .Define("dismuon_cuts",  "DisMuon_pt  > 20 && abs(DisMuon_eta)  < 2.4 && DisMuon_looseId    ")
                                    .Define("photon_filtered_eta", "Photon_eta[photon_cuts]")
                                    .Define("photon_filtered_phi", "Photon_phi[photon_cuts]")
                                    .Define("electron_filtered_eta", "Electron_eta[electron_cuts]")
                                    .Define("electron_filtered_phi", "Electron_phi[electron_cuts]")
                                    .Define("muon_filtered_eta", "Muon_eta[muon_cuts]")
                                    .Define("muon_filtered_phi", "Muon_phi[muon_cuts]")
                                    .Define("dismuon_filtered_eta", "DisMuon_eta[dismuon_cuts]")
                                    .Define("dismuon_filtered_phi", "DisMuon_phi[dismuon_cuts]")
                                    .Define("photon_veto",   leptonVeto, {"jet_filtered_eta", "jet_filtered_phi", "photon_filtered_eta", "photon_filtered_phi"})
                                    .Define("electron_veto", leptonVeto, {"jet_filtered_eta", "jet_filtered_phi", "electron_filtered_eta", "electron_filtered_phi"}) 
                                    .Define("muon_veto",     leptonVeto, {"jet_filtered_eta", "jet_filtered_phi", "muon_filtered_eta", "muon_filtered_phi"})
                                    .Define("dismuon_veto",  leptonVeto, {"jet_filtered_eta", "jet_filtered_phi", "dismuon_filtered_eta", "dismuon_filtered_phi"})
                                    .Define("all_lepton_veto", "photon_veto && electron_veto && muon_veto && dismuon_veto")
                                    .Define("Id_jet_pt", "jet_filtered_pt[all_lepton_veto]")
                                    .Define("Id_jet_score", "jet_filtered_score[all_lepton_veto]")
                                    .Define("Id_jet_dxy", "jet_filtered_dxy[all_lepton_veto]");

    auto jet_pt_hist         = filtered_Jet.Histo1D({"h_jet_pt",         (sampleName[f] + ";p_{T} [GeV];").c_str(), 245, 20, 1000}, "jet_filtered_pt");
    auto jet_dxy_hist        = filtered_Jet.Histo1D({"h_jet_dxy",        (sampleName[f] + ";d_{xy} [cm];").c_str(), 20, -5,  5},    "jet_filtered_dxy");
    auto jet_score_hist      = filtered_Jet.Histo1D({"h_jet_score",      (sampleName[f] + ";score;"      ).c_str(), 20,  0,  1},    "jet_filtered_score");

    auto noId_jet_pt_hist    = noId_Jet.Histo1D(    {"h_noId_jet_pt",    (sampleName[f] + ";p_{T} [GeV];").c_str(), 245, 20, 1000}, "noId_jet_pt");
    auto noId_jet_dxy_hist   = noId_Jet.Histo1D(    {"h_noId_jet_dxy",   (sampleName[f] + ";d_{xy} [cm];").c_str(), 20, -5,  5},    "noId_jet_dxy");
    auto noId_jet_score_hist = noId_Jet.Histo1D(    {"h_noId_jet_score", (sampleName[f] + ";score;"      ).c_str(), 20,  0,  1},    "noId_jet_score");

    auto Id_jet_pt_hist      = Id_Jet.Histo1D(      {"h_Id_jet_pt",      (sampleName[f] + ";p_{T} [GeV];").c_str(), 245, 20, 1000}, "Id_jet_pt");
    auto Id_jet_dxy_hist     = Id_Jet.Histo1D(      {"h_Id_jet_dxy",     (sampleName[f] + ";d_{xy} [cm];").c_str(), 20, -5,  5},    "Id_jet_dxy");
    auto Id_jet_score_hist   = Id_Jet.Histo1D(      {"h_Id_jet_score",   (sampleName[f] + ";score;"      ).c_str(), 20,  0,  1},    "Id_jet_score");



    auto c1 = new TCanvas();
    jet_pt_hist->Draw();
    c1->SaveAs((sampleName[f] + "_jet_pt.pdf").c_str());

    auto c2 = new TCanvas();
    jet_score_hist->Draw();
    c2->SaveAs((sampleName[f] + "_jet_score.pdf").c_str());

    auto c3 = new TCanvas();
    jet_dxy_hist->Draw();
    c3->SaveAs((sampleName[f] + "_jet_dxy.pdf").c_str());

    auto c4 = new TCanvas();
    noId_jet_pt_hist->Draw();
    c4->SaveAs((sampleName[f] + "_noId_jet_pt.pdf").c_str());

    auto c5 = new TCanvas();
    noId_jet_score_hist->Draw();
    c5->SaveAs((sampleName[f] + "_noId_jet_score.pdf").c_str());

    auto c6 = new TCanvas();
    noId_jet_dxy_hist->Draw();
    c6->SaveAs((sampleName[f] + "_noId_jet_dxy.pdf").c_str());

    auto c7 = new TCanvas();
    Id_jet_pt_hist->Draw();
    c7->SaveAs((sampleName[f] + "_Id_jet_pt.pdf").c_str());

    auto c8 = new TCanvas();
    Id_jet_score_hist->Draw();
    c8->SaveAs((sampleName[f] + "_Id_jet_score.pdf").c_str());

    auto c9 = new TCanvas();
    Id_jet_dxy_hist->Draw();
    c9->SaveAs((sampleName[f] + "_Id_jet_dxy.pdf").c_str());
    TFile outfile(("output_" + sampleName[f] + ".root").c_str(), "RECREATE");

    jet_pt_hist->Write();
    jet_score_hist->Write();
    jet_dxy_hist->Write();
   
    noId_jet_pt_hist->Write();
    noId_jet_score_hist->Write();
    noId_jet_dxy_hist->Write();

    Id_jet_pt_hist->Write();
    Id_jet_score_hist->Write();
    Id_jet_dxy_hist->Write();

    outfile.Close();
  }
}

int main() {
LepVeto();
}





