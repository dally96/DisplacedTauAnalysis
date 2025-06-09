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
std::vector<std::string> getFilesFromWildcard(const std::string& wildcard) {
  std::vector<std::string> files;
  
  // Use the TSystem to list files matching the wildcard
  TSystemDirectory dir(".", wildcard.c_str());
  TList *filesList = dir.GetListOfFiles();
  if (filesList) {
    TSystemFile *file;
    TIter next(filesList);
    while ((file = (TSystemFile*)next())) {
      const char *fileName = file->GetName();
      if (!file->IsDirectory()) {
        files.push_back((wildcard + fileName).c_str());
      }
    }
  }
  return files;
}

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

void ZLeptonEtaPlot() {
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


    ROOT::RDataFrame df("Events", fileName[f]);

    auto pt_cut       = [](float pt) { return pt > 20; };
    auto eta_cut      = [](float eta) { return std::abs(eta) < 2.4; };
    auto id_cut       = [](int 
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
                           const ROOT::RVec<short> &pfcands_charge,
                           const ROOT::RVec<short> &jetpfcands_jetidx,
                           const ROOT::RVec<short> &jetpfcands_pfcandsidx) {
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
    };

  df = df.Define("Jet_dxy", getJetDxy, {"Jet_pt", "PFCands_pt", "PFCands_dxy", "PFCands_charge", "JetPFCands_jetIdx", "JetPFCands_pFCandsIdx"})

  auto filtered_Jet = df.Define("jet_mask", "Jet_pt > 20 && abs(Jet_eta) < 2.4")
                        .Define("jet_filtered_pt"   , "Jet_pt[jet_mask]")
                        .Define("jet_filtered_score", "Jet_disTauTag_score1[jet_mask]")
                        .Define("jet_filtered_dxy",   "Jet_dxy[jet_mask]")
                        .Define("jet_filtered_eta",   "Jet_eta[jet_mask]")
                        .Define("jet_filtered_phi",   "Jet_phi[jet_mask]")

  auto noId_Jet     = filtered_Jet.Define(  
