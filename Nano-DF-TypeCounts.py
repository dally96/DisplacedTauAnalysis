import ROOT
import sys
import os
from itertools import product
import argparse
import numpy as np

# ROOT.gROOT.SetBatch(True)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

ROOT.gInterpreter.Declare(f"#include \"{os.path.dirname(__file__)}/../inc/GenLeptonNano.h\"")

ROOT.gInterpreter.Declare("""
namespace JetTypeSelection {

static const double jet_pt = 20; //GeV
static const double jet_eta = 2.4;

static const double genJet_pt = 20; //GeV
static const double genJet_eta = 2.4;

static const double gen_tau_pt = 20; //GeV
static const double gen_tau_eta = 2.4; //GeV
static const double gen_z = 100; //cm
static const double gen_rho = 50; //cm
static const double genLepton_jet_dR = 0.4;
static const double genLepton_iso_dR = 0.4;

}                          
""")

# Function to count mathed jet/taus 
ROOT.gInterpreter.Declare("""
int n_taus(std::vector<reco_tau::gen_truth::GenLepton> genLeptons, 
           ROOT::VecOps::RVec<Float_t>& Jet_pt,
           ROOT::VecOps::RVec<Float_t>& Jet_eta,
           ROOT::VecOps::RVec<Float_t>& Jet_phi,
           ROOT::VecOps::RVec<Float_t>& Jet_mass)
{

   int tau_n = 0;
   for(auto lepton: genLeptons) {
         if(static_cast<int>(lepton.kind())!=5) continue;
         for(int jet_i=0; jet_i < Jet_pt.size(); jet_i++) {
            ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>> jet_p4
                        ( Jet_pt[jet_i], Jet_eta[jet_i], Jet_phi[jet_i], Jet_mass[jet_i] );
            if(jet_p4.pt() < JetTypeSelection::jet_pt ||
               std::abs(jet_p4.eta()) > JetTypeSelection::jet_eta) continue;
            auto visible_tau = lepton.visibleP4();
            auto vertex = lepton.lastCopy().vertex;
            if( std::abs(lepton.lastCopy().pdgId) != 15 ) throw;
            if( std::abs(vertex.z()) > JetTypeSelection::gen_z ||
                std::abs(vertex.rho()) > JetTypeSelection::gen_rho ||
                visible_tau.pt() < JetTypeSelection::gen_tau_pt ||
                std::abs(visible_tau.eta()) > JetTypeSelection::gen_tau_eta)  continue;
            double dR = ROOT::Math::VectorUtil::DeltaR(jet_p4, visible_tau);
            if( dR < JetTypeSelection::genLepton_jet_dR ) tau_n++;       
         }
   }
   return tau_n;
}
""")
ROOT.gInterpreter.Declare("""
int n_jets(std::vector<reco_tau::gen_truth::GenLepton> genLeptons, 
           ROOT::VecOps::RVec<Float_t>& Jet_pt,
           ROOT::VecOps::RVec<Float_t>& Jet_eta,
           ROOT::VecOps::RVec<Float_t>& Jet_phi,
           ROOT::VecOps::RVec<Float_t>& Jet_mass,
           ROOT::VecOps::RVec<Float_t>& GenJet_pt,
           ROOT::VecOps::RVec<Float_t>& GenJet_eta,
           ROOT::VecOps::RVec<Float_t>& GenJet_phi,
           ROOT::VecOps::RVec<Float_t>& GenJet_mass)
{

   int jet_n = 0;
   for(int jet_i=0; jet_i < Jet_pt.size(); jet_i++) {
      ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>> jet_p4
                  ( Jet_pt[jet_i], Jet_eta[jet_i], Jet_phi[jet_i], Jet_mass[jet_i] );
      if(jet_p4.pt() < JetTypeSelection::jet_pt ||
         std::abs(jet_p4.eta()) > JetTypeSelection::jet_eta) continue;
         
      // Drop if match to gen lepton
      bool leptonVeto = false;
      for(auto lepton: genLeptons) {
         auto visible_tau = lepton.visibleP4();
         double dR = ROOT::Math::VectorUtil::DeltaR(jet_p4, visible_tau);
         if( dR < JetTypeSelection::genLepton_iso_dR ) leptonVeto=true;       
      }
      if(leptonVeto) continue;
      
      bool matchGenJet = false;
      for(int gen_i=0; gen_i < GenJet_pt.size(); gen_i++) {
         ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>> gen_p4
                  ( GenJet_pt[gen_i], GenJet_eta[gen_i], GenJet_phi[gen_i], GenJet_mass[gen_i] );
         if(gen_p4.pt() < JetTypeSelection::genJet_pt ||
            std::abs(gen_p4.eta()) > JetTypeSelection::genJet_eta) continue;
         double dR = ROOT::Math::VectorUtil::DeltaR(gen_p4, jet_p4);
         if( dR < JetTypeSelection::genLepton_jet_dR ) matchGenJet=true;
      }
      if(matchGenJet) jet_n++;
      
   }
   return jet_n;
}
""")

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='Count the objects consistenly with training definition.')
   parser.add_argument('--input', help='path to the NanoAOD file')
   args = parser.parse_args()

   ROOT.EnableImplicitMT(1)

   df = ROOT.RDataFrame("Events", args.input)

   df = df.Define("genLeptons", """reco_tau::gen_truth::GenLepton::fromNanoAOD(GenPart_pt, GenPart_eta,
                              GenPart_phi, GenPart_mass, GenPart_vertexX,GenPart_vertexY, GenPart_vertexZ, GenPart_genPartIdxMother, GenPart_pdgId,
                              GenPart_statusFlags, event)""")
   df = df.Define("n_taus", "n_taus(genLeptons, Jet_pt, Jet_eta, Jet_phi, Jet_mass)")
   df = df.Define("n_jets", "n_jets(genLeptons, Jet_pt, Jet_eta, Jet_phi, Jet_mass, GenJet_pt, GenJet_eta, GenJet_phi, GenJet_mass)")
   
   n_taus = df.AsNumpy(columns=["n_taus"])['n_taus']
   print(n_taus)
   print("taus:",np.sum(n_taus))
   
   n_jets = df.AsNumpy(columns=["n_jets"])['n_jets']
   print(n_jets)
   print("jet:",np.sum(n_jets))
