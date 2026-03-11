from scipy.ndimage import gaussian_filter1d
from .base_relevance_function import RelevanceFunctionBase
import numpy as np



class LabelDistributionSmoothing(RelevanceFunctionBase):
    '''Implements label distribution smoothing, but not as stated in the LDS paper with kernel density, but as actually implemented in the paper with a histogram.'''
    def __init__(self) -> None:
        super().__init__()
        self.type = "LabelDistributionSmoothing"

    def fit(self, data: np.ndarray, bins: int = 50, kernel_size: int = 5, kernel_sigma: int = 2) -> None:
        self.data = data
        self.bins = bins
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.bin_density, self.bin_boundaries = np.histogram(data, bins = bins, density=True)

        kernel = self.get_lds_kernel_window()
        self.smoothed_bin_density = np.convolve(self.bin_density, kernel, mode='same')
        

        # Calculate weights
        # der LDS Artikel invertiert die Gewichte einfach. Für OOD samples mit einer Density nahe Null ergibt das jedoch ein nahe unendliches Gewicht.
        # Im Artikel wird dies verhindert indem die Häufigkeit geclipped wird, jedoch nicht in Abhängigkeit der Dichte sondern in Abhängigkeit der Anzahl der Sample im Bin.
        # Im Artikel werden daher auch nur spezifische Datensätze behandelt und das Clipping ist individuell pro Datensatz.
        # Anschließend werden die invertierten Gewichte mit dem Durchschnitt skaliert.
        # weights = 1 / densities
        # # Rescale weights so their mean becomes 1 (without changing their relative proportions)
        # weights = weights * (len(weights) / np.sum(weights))
        # nicht wirklich zielführend, wir machen das daher anders.

        # minmax scale to [0,1]
        self.bin_relevance = (self.smoothed_bin_density - self.smoothed_bin_density.min()) / (self.smoothed_bin_density.max() - self.smoothed_bin_density.min())
        # invert to make frequent samples to be high relevant
        self.bin_relevance = np.maximum(1- self.bin_relevance, 1E-6)
        self.is_fitted = True

    def eval(self, y: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("eval() can not be called before fit().")
        # determine bin indices
        indices = np.digitize(y, bins=self.bin_boundaries) - 1 # subtract 1 so first bin is index 0
        # Create a valid mask: mask samples that are out of distribution; only indices within [0, len(bin_density)-1]
        valid_mask = (indices >= 0) & (indices < len(self.bin_density))
        # Assign relevance
        relevance = np.zeros_like(y, dtype=float)
        relevance[valid_mask] = self.bin_relevance[indices[valid_mask]]

        return relevance


    def get_lds_kernel_window(self):
        # code is partially taken from https://github.com/YyzHarry/imbalanced-regression
        '''
        MIT License

        Copyright (c) 2021 Yuzhe Yang

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        '''
        half_ks = (self.kernel_size - 1) // 2
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=self.kernel_sigma)
        kernel_window /= max(kernel_window)
        return kernel_window