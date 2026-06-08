# DEFINITIVE RANKINGS — BLENDED VIMS (28 April 2026)

## PAST (-3.5 Gya) [0.138, 0.515]
 1. Freeman     lake  210.0  +83.0  0.464
 2. Oib         lake  220.0  +82.0  0.432
 3. Koitere     lake  154.0  +73.0  0.432
 4. Ontario     lake  179.0  -72.0  0.352
 5. Hammar      lake   93.5  +70.7  0.349
 6. Mackay      lake  262.8  +77.5  0.346
 7. Paxsi       lake  243.0  +72.0  0.344
 8. Neagh       lake  327.0  +72.0  0.337
 9. Bolsena     lake   12.3  +75.4  0.331
10. Logtak      lake  197.5  +74.0  0.331
    Selk        lander 199.0  +7.0  0.258
    Huygens     lander 192.3 -10.6  0.239

## LAKE FORMATION (-1.0 Gya) [0.153, 0.538]
 1. Freeman     lake  210.0  +83.0  0.489
 2. Oib         lake  220.0  +82.0  0.461
 3. Koitere     lake  154.0  +73.0  0.459
 4. Ontario     lake  179.0  -72.0  0.392
 5. Mackay      lake  262.8  +77.5  0.384
 6. Hammar      lake   93.5  +70.7  0.384
 7. Paxsi       lake  243.0  +72.0  0.383
 8. Neagh       lake  327.0  +72.0  0.381
 9. Bolsena     lake   12.3  +75.4  0.370
10. Cardiel     lake  128.0  +78.0  0.364
    Selk        lander 199.0  +7.0  0.248
    Huygens     lander 192.3 -10.6  0.247

## PRESENT (Cassini era) [0.155, 0.601]
 1. Freeman     lake  210.0  +83.0  0.541
 2. Koitere     lake  154.0  +73.0  0.509
 3. Oib         lake  220.0  +82.0  0.509
 4. Ontario     lake  179.0  -72.0  0.421
 5. Hammar      lake   93.5  +70.7  0.409
 6. Mackay      lake  262.8  +77.5  0.408
 7. Neagh       lake  327.0  +72.0  0.404
 8. Paxsi       lake  243.0  +72.0  0.403
 9. Bolsena     lake   12.3  +75.4  0.394
10. Cardiel     lake  128.0  +78.0  0.393
    Huygens     lander 192.3 -10.6  0.260
    Selk        lander 199.0  +7.0  0.243

## NEAR FUTURE (+0.25 Gya) [0.154, 0.600]
 1. Freeman     lake  210.0  +83.0  0.540
 2. Koitere     lake  154.0  +73.0  0.509
 3. Oib         lake  220.0  +82.0  0.508
 4. Ontario     lake  179.0  -72.0  0.420
 5. Hammar      lake   93.5  +70.7  0.408
 6. Mackay      lake  262.8  +77.5  0.407
 7. Neagh       lake  327.0  +72.0  0.403
 8. Paxsi       lake  243.0  +72.0  0.402
 9. Bolsena     lake   12.3  +75.4  0.393
10. Cardiel     lake  128.0  +78.0  0.392
    Huygens     lander 192.3 -10.6  0.259
    Selk        lander 199.0  +7.0  0.242

## FUTURE (+5.9 Gya) [0.316, 0.829]
 1. Mackay      lake  262.8  +77.5  0.763
 2. Freeman     lake  210.0  +83.0  0.723
 3. Hammar      lake   93.5  +70.7  0.708
 4. Uvs         lake  262.0  +74.5  0.700
 5. Waikare     lake  185.0  +75.0  0.682
 6. Ontario     lake  179.0  -72.0  0.679
 7. Logtak      lake  197.5  +74.0  0.675
 8. Paxsi       lake  243.0  +72.0  0.671
 9. Oib         lake  220.0  +82.0  0.641
10. Cardiel     lake  128.0  +78.0  0.641
    Selk        lander 199.0  +7.0  0.419
    Huygens     lander 192.3 -10.6  0.409

## KEY OBSERVATIONS
- Freeman Lacus leads Past/LF/Present/NF (#1 at all 4)
- Mackay Lacus leads Future (#1, P(H)=0.763)
- ALL top 10 are lake/sea shores at EVERY epoch
- Kraken/Ligeia/Punga no longer in top 10 at any epoch
- Selk > Huygens at Past/LF, Huygens > Selk at Present/NF/Future
- P(H) max increased to 0.829 (was 0.696 theoretical)

## THESIS UPDATE CHECKLIST

### results.tex — ALL 5 top-10 tables
- Replace all site names, coordinates, and P(H) values
- Add Huygens/Selk below midrule in each table
- Update captions (all are lake shores now, no land sites in top 10)

### results.tex — Prose
- Present: Freeman #1 (0.541), not Kraken. All top 10 are lake shores.
- Past: Freeman #1 (0.464). All top 10 are lake shores (future lake basins).
  OLD narrative "equatorial land sites dominate Past" is WRONG — polar basins lead.
- LF: Freeman #1 (0.489). All lake shores.
- NF: Nearly identical to Present (~0.001 differences).
- Future: Mackay #1 (0.763). All lake shores.
- Selk subsection: f7=0.629 unchanged, P(H)=0.243 (was 0.240)

### methods.tex — Worked example
- Replace Kraken (0.436) with Freeman (0.541) as the worked example
- Freeman weighted sum: (0.541 × 11 - alpha0) / 6 = ...
- Update alpha_post, beta_post, HDI values

### discussion.tex
- All P(H) values referencing specific sites
- Sensitivity analysis may need rerunning
- "Three regimes" narrative needs checking
- Kraken references → Freeman where appropriate

### introduction.tex — No changes needed (no specific P(H) values)
### appendices — Minimal changes (parameter values unchanged)
### methods.tex f2 description — Update to document blended VIMS mosaic
