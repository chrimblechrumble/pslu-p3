# Titan Habitability Pipeline - Outstanding Issues in current commit

## Video/static frames
- [ ] Use the same habitability scoring data for all three maps in the video. The habitability colouring of eg.g Ligeia is 
different on the equirectangular map vs the North Polar map. And if the same habitability scoring 
data can being used, why cut off at +/- 50 degrees latitude on the North/Polar maps? That is, 
we can use all data on all maps, dio it, and all three maps should be cross-referenceable. 
THe underlying habitability data should be the same, and the maps are just views onto that data.

- [ ] The longitude ticks/graticules for the north/south plots should be on the outside of the circumference, 
not on inside.

- [ ] If we do not gave +/- 0-40 polar data North and South, declare it. N/S animations should probably shrink
otherwise they give the impression all teh sata is there, when it isn't

- [ ] Check that text changes match the frame of image changes, e.g. Red Giant Ramp image changes, but text 
change is two frames later.

- [ ] Make video pausing on/off optional, with pausing off by default

## Housekeeping

### Audit code for:
- [ ] All comments and doc strings shouldn't use characters that are not available on a UK Mac keyboard. Make substitutions as necessary. For example, 
  - [ ] the ° should be swapped to 'degrees'
  - [ ] all greek letters α, β, Σ, λ in comments etc should be swaped for expanded names lambda etc. But keep the Greek
  in all equations in thesis, obvs.
- [ ] local variables without types. I still see many across the codebase
- [ ] Unused code. Pycharm reports a lot of declared variables that are unused
- [ ] Comments that refer to bugs in my code that I have fixed should be removed. To be clear, keep comments that refer 
to data/library patching/manipulation
- [ ] Add comments/references to any geometry processing
- [ ] Create file-level constants for colours expressed as hexadecimals
- [ ] All comments, methods, class names, variable names and documentation should use British English spelling. 
- [ ] Is it possible to convert this project to snakemake?
- [ ] add (c) line to all files and all terminal output, something like (for code)

   
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

    <program>  Copyright (C) 2025/2026  Chris Meadows
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

