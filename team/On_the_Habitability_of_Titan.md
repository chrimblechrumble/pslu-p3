# On the Habitability of Titan
### P3 team presentation — slide-by-slide outline

**Format:** Examiner panel · 12 min presentation + 8 min questions
**Structure:** Geologic layer, inside-out (ocean → atmosphere), with a bombardment hook and a red-giant close
**Weighting:** Equal billing for all five studies (one body slide each); the *synthesis* is the climax, not any single study
**Central tension (confronted head-on):** does Titan have a subsurface ocean? — Petricca et al. (2025) vs the classical view
**Figures:** slides 3–8 each carry a figure (interior cross-section + one per teammate); source images are staged in `_figs/`

### The five studies

| Author | Topic | Layer it owns | Timescale |
|---|---|---|---|
| Charlotte Crick | Tidal heating & the subsurface ocean | Deep interior / ocean | 5 Gyr (to red giant) |
| Helena Nicolaides | Impact cratering & melt production | Ice shell / crust | 4.5 Gyr (bombardment history) |
| Lucca Castlevetro | Abiogenesis in brine pockets | Within the ice shell | 200 ps (one impact) |
| Chris Meadows | Bayesian surface habitability map | Surface | 5 epochs (LHB → red giant) |
| Imaan Islam Rahim | Longitudinal atmospheric differentiation | Atmosphere | 13 yr (Cassini), seasonal |

**Timing budget:** ≈ 11.5 min, leaving margin. To reach ~15 slides without changing pace, split slides 4 and 10 into two each.

---

## Slide 1 — Title (~15 s)
- **On the Habitability of Titan**
- Five studies, one moon: interior → atmosphere
- Crick · Nicolaides · Castlevetro · Meadows · Islam Rahim

*Talk track: set the frame — one moon, five layers, one question.*

---

## Slide 2 — Why Titan? (~45 s)
- Life needs three things: a liquid solvent, organic building blocks (CHNOPS), an energy source
- Titan is the only body with a candidate for **all three**
- The catch: they are kept apart (organics on top, water below ~100 km of ice)

*Talk track: "the whole question is whether Titan ever brings them together."*

---

## Slide 3 — The team's map (~45 s)
- Five studies tile Titan from the deep interior outward:
  - Ocean & deep interior — Crick (tidal heating: is there an ocean?)
  - Ice shell — Nicolaides (impact melt) + Castlevetro (brine-pocket chemistry)
  - Surface — Meadows (Bayesian habitability map)
  - Atmosphere — Islam Rahim (vertical profiles)
- *Figure (right): Titan interior cross-section, team poster wedge (`_figs/fig_crosssection.png`)*

*Talk track: "we descend to the ocean, then climb back out to the sky."*

---

## Slide 4 — The deep interior: is there an ocean? (Charlotte) (~60 s)
- Tidal Love number k₂: high value **0.616 ± 0.067** vs recent low value **0.375 ± 0.06**
- PyALMA3 across three interiors (fully liquid A / hp-ice + ocean B / fully solid C)
- All three fit the **high** k₂; none fit the low → observations still allow an ocean
- Even a solid Titan grows a liquid layer within **0.71 Gyr**, sustained to the red giant; orbital migration negligible (~2.7%)
- **Plant the flag:** a 2025 result says the ocean may not be there at all — we return to this

---

## Slide 5 — The ice shell: impacts as the bridge (Helena) (~60 s)
- Bombardment history (Bottke / Nesvorný): **339 impacts**, most in the early giant-planet instability
- Only ~24 reach the surface; melt budget **8.98 × 10⁶ km³** (water ice) vs **8.75 × 10⁵** (clathrate) — an order of magnitude
- ~23% of melt can drain to the ocean; the rest makes surface / shell melt pools
- Catch: pools freeze in 10²–10⁴ yr → habitable, but on the clock

---

## Slide 6 — Inside a brine pocket: does chemistry start? (Lucca) (~60 s)
- One 1.5 km comet → ~16.6 km crater → warm brine (83% water, 12% methane, tholins, cometary organics)
- Reactive MD (GFN2-xTB), 6 replicas: spontaneous **C–N bonds**, glycine mobilised (all 6), P oxidised to **phosphate** and bonded into organics, sulphur chemistry
- Prebiotic **first steps** appear reproducibly
- Caveat, stated plainly: run hot (1000 K) and tiny (311 atoms) → a screen for *possible* chemistry, not rates

---

## Slide 7 — The surface: where is it most habitable? (Chris) (~60 s)
- Bayesian habitability map, whole disc (~6.5M pixels), 8 weighted features, 5 epochs (LHB → red giant)
- North-polar lake shores rank highest; Selk crater among top sites
- Present ≈ 0.33 rising to ≈ 0.70 in the far future
- Note: this map is **ocean-agnostic** — it scores the surface, whoever is right below

---

## Slide 8 — The atmosphere: can we see any of this from orbit? (Imaan) (~60 s)
- Cassini VIMS-IR limb, 2004–2017: two distinct vertical profiles, Sub- vs Anti-Saturn (120° transition)
- Varies over years, with an equinox minimum in 2009
- Drivers on the table: magnetosphere (favoured), tides, seasonal insolation
- Sharp implication: tidally-locked worlds may **not** be longitudinally uniform — matters for exoplanets too

---

## Slide 9 — Where we agree (~45 s)
- Impact cratering is the master bridge (Nicolaides + Castlevetro + Meadows all hinge on it)
- Habitability may **not** need the global ocean: shell pockets and melt pools are self-contained habitats
- Titan is a moving target: everyone works through time
- Shared next step: Selk crater, Dragonfly (2034), JWST

---

## Slide 10 — Open questions (~90 s) — *give this the most air*
- **Q1 — Does the ocean exist?** Crick: models allow it. Islam Rahim: if her atmospheric asymmetry is tidal, it *supports* Petricca 2025's no-ocean. Nicolaides: "if Petricca is right, the ocean-transport story doesn't apply."
  - **My read:** for *surface* habitability the ocean debate is second-order (my map scores the surface either way); for *interior* habitability it's decisive. The honest answer today is "unresolved," and Dragonfly won't settle it — this is a gravity/geophysics question that needs a dedicated orbiter, not a lander.
- **Q2 — Does transient chemistry matter?** Pools freeze in 10²–10⁴ yr (Nicolaides, Castlevetro), yet the chemistry is reproducible.
  - **My read:** I think the team under-sells the pockets. 10²–10⁴ years is enormous next to reaction timescales, and a frozen pocket *preserves* its products instead of diluting them into the ocean. The real limiting factor is abundance (how many pockets, how often) — Helena's 339-impact budget — not whether any single pocket has "time."
- **Q3 — What is the crust?** Water ice vs clathrate changes melt by 10× and underpins two studies.
  - **My read:** clathrate is the right call (it fits the shallow crater topography and insulates the ocean), and it's the *conservative* one — it yields ~10× less melt. So the melt-driven habitability case is made on the pessimistic setting, which makes it more robust, not less.

*Talk track: this is the slide examiners will probe — slow down, invite the ocean debate.*

---

## Slide 11 — Synthesis: a layered case (~60 s) — *the climax, credits all five*
- Interior energy (Crick) → impacts breach the shell (Nicolaides) → concentrated prebiotic chemistry (Castlevetro) → mappable surface niches (Meadows) → expressed in the atmosphere (Islam Rahim)
- Key move: the shell-and-surface niches **survive even if the ocean is disproved**
- Titan's habitability case does not stand or fall on the ocean

---

## Slide 12 — The arc of time & what's next (~45 s)
- Bombardment past → Cassini present → red-giant future (Titan may **peak** in habitability as the Sun expands)
- Near term: Dragonfly at Selk (2034), JWST seasonal monitoring
- The bookend that opened the interior question now closes it

---

## Slide 13 — Verdict (~30 s)
- One line per study
- Collective answer: not "is Titan alive?" but **"Titan has more independent, testable habitable niches than any other world we can currently reach"**

---

*Prepared for the P3 group presentation. Structure and figures to be finalised by the team.*
