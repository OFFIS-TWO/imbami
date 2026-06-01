# Imbami
This package called "imbami" (Imbalance Mitigation) implements the methods for the mitigation of data imbalance as described in the corresponding article "Model-agnostic Mitigation Strategies of Data Imbalance for Regression". The article is available [as a pre print on Arxiv](https://arxiv.org/abs/2506.01486).

The goal of imbami is to provide the scientific community with a set of relevance functions and mitigation methods to tackle problem of data imbalance. The following relevance functions and mitigation methods are included: 
* Relevance Functions
  * **Density-distance relevance**
  * **Density-ratio relevance**
* Mitigation Methods
  * **continuous SMOGN (cSMOGN)**
  * **continuous ratio-based SMOGN (crbSMOGN)**



If used, please cite:
```
@misc{wibbeke_model-agnostic_2025,
	title = {Model-agnostic {Mitigation} {Strategies} of {Data} {Imbalance} for {Regression}},
	url = {https://arxiv.org/abs/2506.01486},
	author = {Wibbeke, Jelke and Rohjans, Sebastian and Rauh, Andreas},
	year = {2025},
	note = {\_eprint: 2506.01486},
}
```

## Requirements and Installation
Download the package from GitHub, navigate to the `setup.py` and install it using pip:
```
pip install .
```
The package was tested using:
```
python=3.12.8
numpy=1.26.4
pandas=2.2.3
KDEpy=1.1.11
```

Other versions may also work, but have not been tested.

## Usage
For explanation see the [example notebook](explanatory_example.ipynb).
And for a more advanced example see the [advanced example notebook](article_example.ipynb) covering one of the examples from the article.
