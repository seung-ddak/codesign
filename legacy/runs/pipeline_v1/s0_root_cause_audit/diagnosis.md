# Generated E/S0 penetration diagnosis

Result: FAIL

## 0-prerequisites : PASS
Input artifacts and expert model located

## 1-original-MJX-physics : PASS
Original physical MJX states pass the current collision gate

## 2-real-S0-feature-roundtrip : PASS
Real E/S0 preserved across TT features and execution decoder

## 3a-reproduce-saved-geometry : PASS
Current model reproduces the saved smoke feet geometry

## 3b-generated-E-S0-physics : FAIL
2/2 generated pairs fail geometry; MJX execution is blocked

## Next action
The original real states and encoder-decoder passed; the SAVED generated E/S0 pairs are inconsistent. Next compare many seeds and analyze E-dependent body height/joints. Do not shift height or relax collision tolerance to force a pass.