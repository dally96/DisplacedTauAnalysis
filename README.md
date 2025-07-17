To run skimming on NanoAODs:
  1. Run `./shell coffeateam/coffea-dask-almalinux9:latest` to enter a singularity container
     
       a. These are all my current package versions:
     
         - uproot 5.6.2
         - awkward 2.7.4
         - dask-awkward 2025.2.0
         - dask 2024.8.0
         - coffea 2025.1.1
         - numpy 1.24.0
         - hist 2.8.0
         - dask-histogram 2025.2.0
     
  3. Next run `ulimit -n 20000` -- this circumvents a "Too many files open error"
  4. Create a dictionary of the datasets like in fileset.py
  5. Add the dictionary to preprocess.py and copy the sturcture of one of the loops to dump the output of the preprocess step to a pkl file
  6. Once that's done running, the first skimming step is done by mutau_bkgd_skim.py

       a. The second argument in Line 243 changes where the output root files are written https://github.com/dally96/DisplacedTauAnalysis/blob/c00810709f2c8cfa79e9042536027859d9f7dd00/mutau_bkgd_skim.py#L243
       b. Line 264 is where you import the pkl file from preprocess.py https://github.com/dally96/DisplacedTauAnalysis/blob/c00810709f2c8cfa79e9042536027859d9f7dd00/mutau_bkgd_skim.py#L264-L265
       c. Line 296 is there so that a dataset isn't skimmed again in case there were other datasets that failed. Change the directory to wherever you're saving the root files  https://github.com/dally96/DisplacedTauAnalysis/blob/c00810709f2c8cfa79e9042536027859d9f7dd00/mutau_bkgd_skim.py#L296
       d. Line 297 is there so that if there is a only some datasets you want to run over, you ignore those you don't. You can comment it out if it's not useful to you https://github.com/dally96/DisplacedTauAnalysis/blob/c00810709f2c8cfa79e9042536027859d9f7dd00/mutau_bkgd_skim.py#L297
       e. If a dataset has successfully been processed, you should see line 315 https://github.com/dally96/DisplacedTauAnalysis/blob/c00810709f2c8cfa79e9042536027859d9f7dd00/mutau_bkgd_skim.py#L315
       f. If it errors out, the jobs should close themselves
     
  8. Next, I usually run mergeFiles.py in the directory the root files have been written, but this is not necessary.
  9. Create another dictionary with the output root files from the first skimming step and follow step 5 for this dictionary
  10. The second skimming step is done by selection_cuts.py. Steps 5a-5f apply here at the respective lines.
  11. Again, I usually run mergeFiles.py here since it makes plotting faster
  12. Plotting is done in plotting_processor_mu.py

       a. Lines 327-329 will dump the QCD histograms into a pkl file to use later on in ROCC-coffea.py which is where we plot the isolation ROC curves https://github.com/dally96/DisplacedTauAnalysis/blob/main/plotting_processor_mu.py#L327-L329
  13. Load pkl file into ROCC-coffea.py https://github.com/dally96/DisplacedTauAnalysis/blob/867fcf29256cdb810015265283a31859e96ee211/ROCC-coffea.py#L76
  14. Make sure bins the bins match the ones from plotting_processor_mu.py
  15. Now run the script, and it should output ROC curves
