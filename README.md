To make the current reconstruction efficiency plots:
  1. Get the list of leptons that decay from a stau, I run [ZtomumuIdx.py](https://github.com/dally96/DisplacedTauAnalysis/blob/main/ZtomumuIdx.py)
  - This gives me the indices for every copy of the leptons in the GenPart_pdgId branch as well as a separate array of just the indices of the last copies.
  - This gets saved in parquet files, however for different Stau samples, I would have to run each time I change the input sample.
  - This takes a little bit of time, but I only have to do it once per sample.
  2. Next, I run [StautomuReco.py](https://github.com/dally96/DisplacedTauAnalysis/blob/main/StautomuReco.py) if I'm just looking at the overall lepton reconstruction, so the prompt and displaced plots.
     If I'm looking at the IDs and their effect on the reconstruction, I use [StautoLepIDReco.py](https://github.com/dally96/DisplacedTauAnalysis/blob/main/StautoLepIDReco.py)
  3. It was when I was writing up StautoLepIDReco.py, I decided to store the plotting functions for efficiency and resolution in [leptonPlot.py](https://github.com/dally96/DisplacedTauAnalysis/blob/main/leptonPlot.py)
  - You see it used in StautoLepIDReco.py, however there are a lot of arguments to these functions
  4. For [StauTriggIdx.py](https://github.com/dally96/DisplacedTauAnalysis/blob/main/StauTriggIdx.py), I started to do what I did in ZtomumuIdx.py, but then I pivoted.
    Now I combine steps 1 and 2 in the same file where it runs once, stores values in a pkl file, and then, if I run it again to add or fix something, then it doesn't go through the beginning loop again.
    This isn't finished yet because plotting is going to be slightly different than what I've been doing before

   I guess the biggest concerns are can I speed up the process that gets the indices of the displaced leptons? And is there anything I can do for the length of my files since I'm making multiple selections? 
