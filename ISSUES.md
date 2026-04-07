# Titan Habitability Pipeline - Outstanding Issues in current commit

## Improvements
- [x] cirs_temperature is based on present day only. What can be done for modelling other epochs?
- [x] how to use the /palermo mappings/shapefiles. Which ones are they in the Birch dataset?
- [ ] swap out sklearn for pymc or numpyro  
- [ ] the vims parquet file was created from code that is not available. Is it needed? If it as not available, 
how would I know which VIMS cubes to use?
- [x] Add text sections at the bottom of video for which features are and are not included. Add a more in 
depth explanation for what is happening for each feature that is being used and which one dominates that explains the 
coming colour trends in locations of interest, and also what assumptions are being made.
- [ ] Make set of labelled sites consistent across phases

## Video/static frames
- [x] Use the same habitability scoring data for all three maps in the video. The habitability colouring of eg.g Ligeia is 
different on the equirectangular map vs the North Polar map. And if the same habitability scoring 
data can being used, why cut off at +/- 50 degrees latitude on the North/Polar maps? That is, 
we can use all data on all maps, do it, and all three maps should be cross-referenceable. 
The underlying habitability data should be the same, and the maps are just views onto that data.
- [x] The longitude ticks/graticules for the north/south plots should be on the outside of the circumference, 
not on inside.
- [x] If we do not have +/- 0-40 polar data North and South, declare it. N/S animations should probably shrink
otherwise they give the impression all the data is there, when it isn't
- [x] Check that text changes match the frame of image changes, e.g. Red Giant Ramp image changes, but text 
change is two frames later.
- [x] Make video pausing on/off optional, with pausing off by default
- [ ] Cross reference that all downloaded map data is actually used. If not, why not and delete, or use it.
- [x] Check INSTALL instructions are correct; maybe script up the manual downloaders
- [ ] Resolve use of GT2ED00N090_T126_V01 vs GT2ED00N090_T090_V01 data
- [ ] Explain "[D3] Subsurface ocean prior  : 0.030  (Neish et al. 2024: ~1 elephant/yr organic flux to ocean)" better

## Other output
- [x] liquid_hydrocarbon.tif has some invalid values and does not render - check. 
- [x] organic_abundance.tif has an imbalance in resolution in E vs W hemispheres - can anything be done. 
- [x] why do the figure/* diagrams look different to the animation frames
- [ ] Fix framing in 'importances' pdf
- [ ] Enumerate/give ordinals to 'top sites'
- [ ] Change the 'interactive' map to include P
- [ ] Investigate why the 'temporal comparison' is almost yellow everywhere, but still dotted, whereas video
is not dotted (see e.g. frame 65). There is inconsistency going on. 
- [x] Investigate how VIMS parquet file is being used; it's just an index. The code to pull in actual cubes 
from Nantes is either being used minimally or not not being used. Is VIMS data not useful?
- [ ] What does 'organic flux to ocean ~1 elephant/year' mean in the subsurface_ocean prior_mean?

## Housekeeping

### Audit code for:
- [x] All comments and doc strings shouldn't use characters that are not available on a UK Mac keyboard. Make substitutions as necessary. For example, 
  - [x] the ° should be swapped to 'degrees'
  - [x] the ≈ should be =~
  - [x] all greek letters α, β, Σ, λ in comments etc should be swaped for expanded names lambda etc. But keep the Greek
  in all equations in thesis, obvs.
- [ ] local variables without types. I still see many across the codebase
- [ ] If the unit tests don't use fixture data that is used in the real pipeline, consider changing teh fixture (and 
getting rid of the data that is only used by fixtures)
- [ ] Unused code. Pycharm reports a lot of declared variables that are unused
- [x] Comments that refer to bugs in the code that I've fixed should be removed. To be clear, keep comments that refer 
to data/library patching/manipulation
- [ ] Add comments/references to any geometry processing
- [ ] Create file-level constants for colours expressed as hexadecimals
- [ ] All comments, methods, class names, variable names and documentation should use British English spelling. 
- [ ] Is it possible to convert this project to snakemake?
- [x] add (c) line to all files and all terminal output, 
something like (for code)
 

    Titan Habitability Pipeline - Compute P(Habitable | features) over Geologic Time
    Copyright (C) 2025/2026  Chris Meadows, cm10004@cam.ac.uk

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>

and for console output:

    Titan Habitability Pipeline  Copyright (C) 2025/2026  Chris Meadows
    This program comes with ABSOLUTELY NO WARRANTY; for details, see the 
    README.md at the project root.
    This is free software, and you are welcome to redistribute it
    under certain conditions; see the LICENSE.md file at the project 
    root for details.


