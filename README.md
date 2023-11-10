# ml-extreme-precip

This project builds on the analysis performed in Davenport & Diffenbaugh (2021) to understand the atmospheric circulation patterns leading to extreme precipitation, using a convolutional neural network. 

## Input data 
*Features*: [NCEP-NCAR Renanalysis-1](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html)
- Daily anomalies of sea level pressure 
- Daily anomalies of detrended geopotential height at 500 hPa

<br>*Labels*: [CHIRPS precipitation]([https://gpm.nasa.gov/data/imerg](https://www.chc.ucsb.edu/data/chirps))
- Daily extreme precipitation classes, either class 0 (no extreme precip) or class 1 (extreme precip)

## References
- [Davenport & Diffenbaugh (2021)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL093787), including the [project code](https://github.com/fdavenport/GRL2021)
- [Code repository](https://github.com/eabarnes1010/course_ml_ats) from Prof. Elizabeth Barnes' Machine Learning for Atmospheric Sciences course at Colorado State University
