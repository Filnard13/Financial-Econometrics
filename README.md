# Financial-Econometrics
Financial Econometrics Project ~ On the Euro-Dollar Exchange Rate ~
# Euro–Dollar Exchange Rate Econometric Analysis

**Author:** Filippo Nardoni  
**Affiliation:** University of Bologna  
**Student ID:** 0001172086  

---

## Project Overview

This project develops an extensive **econometric analysis of the EUR/USD exchange rate**.  
The analysis is structured into **four main sections**:

1. **Introduction** – Background and motivation  
2. **Daily Data Analysis** – Econometric modeling using daily observations  
3. **Weekly Data Analysis** – Econometric modeling using weekly observations  
4. **Bootstrap Methodology for ARCH(1)** – Bootstrap-based inference for volatility models  

Both **daily and weekly data** are sourced from the **FRED database**.

---

## Project Structure
```
Project/
│
├── DATA/
│   └── Raw and processed datasets used in the analysis
│
├── CODE/
│   └── Source code for data processing, estimation, and analysis
│
├── FIGURES/
│   ├── DAILY/
│   │   └── Figures based on daily data
│   └── WEEKLY/
│       └── Figures based on weekly data
│
└── TABLES/
    ├── DAILY/
    │   └── Tables based on daily data
    └── WEEKLY/
        └── Tables based on weekly data
```

---

## Data

- Source: **Federal Reserve Economic Data (FRED)**
- Frequency:
  - Daily
  - Weekly
- Exchange Rate: **EUR/USD**

---

## Methods

The project applies standard and advanced econometric techniques, including:

- ARCH and GARCH-type volatility models  
- Model estimation on different data frequencies  
- Diagnostic analysis  
- Bootstrap inference for ARCH(1) models  

---

## Output

- **Figures** are saved in `/FIGURES/DAILY` and `/FIGURES/WEEKLY`
- **Tables** are saved in `/TABLES/DAILY` and `/TABLES/WEEKLY`

---

## Reproducibility

All scripts are organized to allow **full reproducibility** of the empirical results, provided the required data are available in the `DATA/` directory.

---

## Notes

This project was developed for academic purposes within the econometrics curriculum at the University of Bologna.
