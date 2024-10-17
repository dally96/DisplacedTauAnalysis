import hist
import hist.dask as hda

variables_with_bins = {
    "pt": (50, 0, 500),
    "eta": (40, -2.5, 2.5)
}

# Create a test histogram to check if axis.Regular works
for var, bin_info in variables_with_bins.items():
    print(f"Creating histogram for {var}")
    hist_test = hda.Hist(
        hist.axis.Regular(*bin_info, name=var),  # Should work if everything is correct
        hist.storage.Weight(),
    )
    print(hist_test)
