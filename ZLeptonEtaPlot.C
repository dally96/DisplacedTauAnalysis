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
                                       //eossubdir + "group/lpcdisptau" + subdir + "QCD_PT-1000to1400_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "QCD_PT-120to170_TuneCP5_13p6TeV_pythia8/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "QCD_PT-120to170_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       //eossubdir + "group/lpcdisptau" + subdir + "QCD_PT-1400to1800_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "QCD_PT-170to300_TuneCP5_13p6TeV_pythia8/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "QCD_PT-170to300_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       //eossubdir + "group/lpcdisptau" + subdir + "QCD_PT-1800to2400_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       //eossubdir + "group/lpcdisptau" + subdir + "QCD_PT-2400to3200_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       //eossubdir + "group/lpcdisptau" + subdir + "QCD_PT-3200_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "QCD_PT-300to470_TuneCP5_13p6TeV_pythia8/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "QCD_PT-300to470_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "QCD_PT-470to600_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "QCD_PT-50to80_TuneCP5_13p6TeV_pythia8/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "QCD_PT-600to800_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "QCD_PT-800to1000_TuneCP5_13p6TeV_pythia8_ext/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "QCD_PT-80to120_TuneCP5_13p6TeV_pythia8/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/*.root",
                                       //eossubdir + "user/fiorendi"    + subdir + "DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/*.root",
                                       //"/afs/cern.ch/work/f/fiorendi/public/displacedTaus/customNano_TT4Q.root",
                                       //"/afs/cern.ch/work/f/fiorendi/public/displacedTaus/customNano_TT4Q_precision23.root",
                                       //"/afs/cern.ch/work/f/fiorendi/public/displacedTaus/customNano_TT4Q_63kevents.root",
                                       "/eos/cms/store/user/fiorendi/displacedTaus/cmsRun_out_precision23_morefiles.root",
                            };     

  std::vector<std::string> sampleName = {
                                         //"QCD_PT-1000to1400ext",
                                         //"QCD_PT-120to170",
                                         //"QCD_PT-120to170ext",
                                         //"QCD_PT-1400to1800ext",
                                         //"QCD_PT-170to300",
                                         //"QCD_PT-170to300ext",
                                         //"QCD_PT-1800to2400ext",
                                         //"QCD_PT-2400to3200ext",
                                         //"QCD_PT-3200ext",
                                         //"QCD_PT-300to470",
                                         //"QCD_PT-300to470ext",
                                         //"QCD_PT-470to600ext",
                                         //"QCD_PT-50to80",
                                         //"QCD_PT-600to800ext",
                                         //"QCD_PT-800to1000ext",
                                         //"QCD_PT-80to120",
                                         //"TTto2L2Nu",
                                         "TTto2L2NuMorePrecision",
                                         //"TTto4Q",
                                         //"TTtoLNu2Q",
                                         //"DYJetsToLL",
                                         //"TTto4QNominal",
                                         //"TTto4QMorePrecision",
                            };     
                                 
  for (int f = 0; f < (int)fileName.size(); f++) {
    std::cout << "Working with file " << fileName[f] << std::endl;
    ROOT::EnableImplicitMT();
    
    //std::vector<std::string> files = expandWildcard(fileName[f]);
    //std::vector<std::string> infiles;

    //for (const auto& fname : files) {
    //  TFile *file = TFile::Open(fname.c_str());
    //  TTree *tree = dynamic_cast<TTree*>(file->Get("Events"));
    //  if (!tree) { 
    //    std::cerr << "No TTree 'Events' in: " << fname << std::endl;
    //    file->Close();
    //    delete file;
    //    continue;
    //    }   
    //  infiles.push_back(fname);
    //  file->Close();
    //  delete file;
    //}      


    ROOT::RDataFrame df("Events", fileName[f]);

    auto pt_cut       = [](float pt) { return pt > 20; };
    auto eta_cut      = [](float eta) { return std::abs(eta) < 2.4; };
    auto pdgid_e_cut  = [](int id) { return std::abs(id) == 11; };
    auto pdgid_mu_cut = [](int id) { return std::abs(id) == 13; };
    auto lastcopy_cut = [](int flag) { return (flag & (1 << 13)) != 0;};
    auto matchJetsToGenTaus = [](const ROOT::RVec<float> &jet_eta,
                                 const ROOT::RVec<float> &jet_phi,
                                 const ROOT::RVec<float> &gen_eta,
                                 const ROOT::RVec<float> &gen_phi) {
      ROOT::RVec<bool> matched;
      for (size_t i = 0; i < jet_eta.size(); ++i) {
        bool isMatched = false;
        for (size_t j = 0; j < gen_eta.size(); ++j) {
          float dR = ROOT::VecOps::DeltaR(jet_eta[i], jet_phi[i], gen_eta[j], gen_phi[j]);
          if (dR < 0.3) {
            isMatched = true;
            break;
          }
        }
        matched.push_back(isMatched);
      }
      return matched;
    }; 


    auto isMuon = [](const ROOT::RVec<int> &pdgIds) {
      return Map(pdgIds, [](int pdgId) { return std::abs(pdgId) == 13; });
    };

    auto idxToPdgId = [](const ROOT::RVec<short> &idxVec, const ROOT::RVec<int> &pdgIdVec) {
      ROOT::RVec<int> out;
      out.reserve(idxVec.size());
      for (size_t i = 0; i < idxVec.size(); ++i) {
        int idx = idxVec[i];
        out.emplace_back(idx >= 0 ? pdgIdVec[idx] : -999);
      }
      return out;
    };

    auto idxToGenVar = [](const ROOT::RVec<short> &idxVec, const ROOT::RVec<float> &varVec) {
      ROOT::RVec<float> out;
      out.reserve(idxVec.size());
      for (size_t i = 0; i < idxVec.size(); ++i) {
        int idx = idxVec[i];
        if (idx >= 0) {
          out.emplace_back(varVec[idx]);
        }
      }
      return out;
    };


    auto isElectron = [](const ROOT::RVec<int> &pdgIds) {
      return Map(pdgIds, [](int pdgId) { return std::abs(pdgId) == 11; });
    };


    //auto electron = df.Filter(pt_cut, {"GenPart_pt"}).Filter(eta_cut, {"GenPart_eta"}).Filter(pdgid_e_cut, {"GenPart_pdgId"}).Filter(lastcopy_cut, {"GenPart_statusFlags"});
    auto electron = df.Define("e_mask", "GenPart_pt > 20 && abs(GenPart_eta) < 2.4 && abs(GenPart_pdgId) == 11 && (GenPart_statusFlags & (1 << 13)) != 0")
                      .Define("e_Filtered_eta", "GenPart_eta[e_mask]");
    //auto muon     = df.Filter(pt_cut, {"GenPart_pt"}).Filter(eta_cut, {"GenPart_eta"}).Filter(pdgid_mu_cut, {"GenPart_pdgId"}).Filter(lastcopy_cut, {"GenPart_statusFlags"});
    auto muon = df.Define("mu_mask", "GenPart_pt > 20 && abs(GenPart_eta) < 2.4 && abs(GenPart_pdgId) == 13 && (GenPart_statusFlags & (1 << 13)) != 0")
                      .Define("mu_Filtered_eta", "GenPart_eta[mu_mask]");
    
    auto df_HadTau_pt = df.Define("HadTau_pt", idxToGenVar, {"GenVisTau_genPartIdxMother", "GenPart_pt"});
    auto df_HadTau_phi = df_HadTau_pt.Define("HadTau_phi", idxToGenVar, {"GenVisTau_genPartIdxMother", "GenPart_phi"});
    auto df_HadTau_eta = df_HadTau_phi.Define("HadTau_eta", idxToGenVar, {"GenVisTau_genPartIdxMother", "GenPart_eta"});

    auto df_HadTau_masked = df_HadTau_eta.Define("eta_mask", "abs(HadTau_eta) < 2.4").Define("pt_mask", "HadTau_pt > 20");
    auto df_HadTau_combomask = df_HadTau_masked.Define("CombinedMask", "eta_mask && pt_mask");
    auto df_HadTau_filtered = df_HadTau_combomask.Define("Filtered_eta", "HadTau_eta[CombinedMask]").Define("Filtered_pt", "HadTau_pt[CombinedMask]").Define("Filtered_phi", "HadTau_phi[CombinedMask]");  

    auto df_Muon_matchId = df.Define("Muon_genPdgId", idxToPdgId, {"Muon_genPartIdx", "GenPart_pdgId"});
    auto df_Muon_mask = df_Muon_matchId.Define("Muon_isGenMuon", isMuon, {"Muon_genPdgId"});
    auto df_Muon_pt = df_Muon_mask.Define("MatchedMuon_pt", "Muon_pt[Muon_isGenMuon]");
    auto df_Muon_eta = df_Muon_pt.Define("MatchedMuon_eta", "Muon_eta[Muon_isGenMuon]");

    auto df_Muon_masked = df_Muon_eta.Define("eta_mask", "abs(MatchedMuon_eta) < 2.4").Define("pt_mask", "MatchedMuon_pt > 20");
    auto df_Muon_combomask = df_Muon_masked.Define("CombinedMask", "eta_mask && pt_mask");
    auto df_Muon_filtered = df_Muon_combomask.Define("Filtered_eta", "MatchedMuon_eta[CombinedMask]").Define("Filtered_pt", "MatchedMuon_pt[CombinedMask]");

    auto df_Electron_matchId = df.Define("Electron_genPdgId", idxToPdgId, {"Electron_genPartIdx", "GenPart_pdgId"}); 
    auto df_Electron_mask = df_Electron_matchId.Define("Electron_isGenElectron", isElectron, {"Electron_genPdgId"}); 
    auto df_Electron_pt = df_Electron_mask.Define("MatchedElectron_pt", "Electron_pt[Electron_isGenElectron]");
    auto df_Electron_eta = df_Electron_pt.Define("MatchedElectron_eta", "Electron_eta[Electron_isGenElectron]");

    auto df_Electron_masked = df_Electron_eta.Define("eta_mask", "abs(MatchedElectron_eta) < 2.4").Define("pt_mask", "MatchedElectron_pt > 20");
    auto df_Electron_combomask = df_Electron_masked.Define("CombinedMask", "eta_mask && pt_mask");
    auto df_Electron_filtered = df_Electron_combomask.Define("Filtered_eta", "MatchedElectron_eta[CombinedMask]").Define("Filtered_pt", "MatchedElectron_pt[CombinedMask]");

    auto df_Jet_mask = df_HadTau_filtered.Define("Jet_matchedToGenTau", matchJetsToGenTaus, {"Jet_eta", "Jet_phi", "Filtered_eta", "Filtered_phi"});
    auto df_Jet_pt = df_Jet_mask.Define("MatchedJet_pt", "Jet_pt[Jet_matchedToGenTau]");
    auto df_Jet_eta = df_Jet_pt.Define("MatchedJet_eta", "Jet_eta[Jet_matchedToGenTau]");

    auto df_Jet_masked = df_Jet_eta.Define("jet_eta_mask", "abs(MatchedJet_eta) < 2.4").Define("jet_pt_mask", "MatchedJet_pt > 20");
    auto df_Jet_combomask = df_Jet_masked.Define("JetCombinedMask", "jet_eta_mask && jet_pt_mask");
    auto df_Jet_filtered = df_Jet_combomask.Define("JetFiltered_eta", "MatchedJet_eta[JetCombinedMask]").Define("JetFiltered_pt", "MatchedJet_pt[JetCombinedMask]");


    auto electron_eta_hist = electron.Histo1D({"h_e_eta", (sampleName[f] + ";#eta;").c_str(), 500, -2.4, 2.4}, "e_Filtered_eta");
    auto recoelectron_eta_hist = df_Electron_filtered.Histo1D({"h_recoe_eta", (sampleName[f] + ";#eta;").c_str(), 500, -2.4, 2.4}, "Filtered_eta");

    auto muon_eta_hist = muon.Histo1D({"h_mu_eta", (sampleName[f] + ";#eta;").c_str(), 500, -2.4, 2.4}, "mu_Filtered_eta");
    auto recomuon_eta_hist = df_Muon_filtered.Histo1D({"h_recomu_eta", (sampleName[f] + ";#eta;").c_str(), 500, -2.4, 2.4}, "Filtered_eta");

    auto tau_eta_hist = df_HadTau_filtered.Histo1D({"h_tau_eta", (sampleName[f] + ";#eta;").c_str(), 500, -2.4, 2.4}, "Filtered_eta");
    auto recotau_eta_hist = df_Jet_filtered.Histo1D({"h_recotau_eta", (sampleName[f] + ";#eta;").c_str(), 500, -2.4, 2.4}, "JetFiltered_eta");


    auto c1 = new TCanvas();
    electron_eta_hist->Draw();
    recoelectron_eta_hist->Draw("SAME");
    recoelectron_eta_hist->SetLineColor(kRed);
    auto l1 = new TLegend();
    l1->AddEntry("h_e_eta", "Gen electrons");
    l1->AddEntry("h_recoe_eta_hist", "Reco electrons");
    l1->Draw();
    c1->SaveAs((sampleName[f] + "electron_eta.pdf").c_str());

    auto c2 = new TCanvas();
    muon_eta_hist->Draw();
    recomuon_eta_hist->Draw("SAME");
    recomuon_eta_hist->SetLineColor(kRed);
    auto l2 = new TLegend();
    l2->AddEntry("h_mu_eta", "Gen muons");
    l2->AddEntry("h_recomu_eta", "Reco muons");
    l2->Draw();
    c2->SaveAs((sampleName[f] + "muon_eta.pdf").c_str());

    auto c3 = new TCanvas();
    tau_eta_hist->Draw();
    recotau_eta_hist->Draw("SAME");
    recotau_eta_hist->SetLineColor(kRed);
    auto l3 = new TLegend();
    l3->AddEntry("h_tau_eta", "Gen had taus");
    l3->AddEntry("h_recotau_eta", "Reco jets");
    l3->Draw();
    c3->SaveAs((sampleName[f] + "tau_eta.pdf").c_str());

    TFile outfile(("output_" + sampleName[f] + ".root").c_str(), "RECREATE");

    electron_eta_hist->Write();
    recoelectron_eta_hist->Write();
    muon_eta_hist->Write();
    recomuon_eta_hist->Write();
    tau_eta_hist->Write();
    recotau_eta_hist->Write();
  
    outfile.Close();
  }
}

int main() {
ZLeptonEtaPlot();
}
