# Peaks Theory Codes
## Folder structure
This folder contains some code I built to perform, visualize and simulates results and calculations relative to random Gaussian field 
statistics. Each of the Python notebooks and scripts is dedicated to a particular job explained as follow

## ```BBKS_peaks_density&profiles.ipynb```
This notebooks contains all the calculations and plots relative to the work done in BBKS article:
* Differential and cumulative number density
* Probability distributions of peaks geometrical parameters
* Peak profiles means and distributions

## ```peaks_generator_general.py```
This script is the one responsible for the numerical simulations of the gaussian field. Generates a given number of realization of a Gaussian or non-Gaussian by the map
$$F_{NG} = -\frac{1}{\beta}\log(1-\beta F_{G}))$$
given beta.

Then it saves in a .csv file all the peaks profiles over a certain given threshold $\nu$ and in a .txt file the info about the simulation run. Look at the comments in the code for futher details.

## ```Density_analysis.ipynb```
Extract the differential number density of peaks from the realizations

## ```Peaks_analysis_gaussian/nongaussian.ipynb```
Contains all the code to extract the profiles for Gaussian and non-Gaussian field realizations and comparisons with the BBKS results.