# Macro Regime Allocator

Ce repo implémente une stratégie d'allocation d'actifs **entièrement data-driven**, sans prédiction explicite du futur :

1. Analyse des indicateurs macro (croissance, inflation, taux, chômage, volatilité, etc.).
2. Identification du régime macroéconomique courant (Croissance / Stable / Récession).
3. Recherche des périodes historiques les plus similaires (même régime + niveaux macro proches).
4. Observation de la performance des différentes classes d’actifs après ces périodes.
5. Construction de moyennes de rendements et d'une matrice de covariance (avec shrinkage).
6. Optimisation d’un portefeuille sous contraintes (long-only, poids max par actif, volatilité cible).
7. Rééquilibrage mensuel avec backtest hors-échantillon (no look-ahead).

## Structure du projet

- `src/macro_regime_allocator/` : cœur du modèle (macro, similarité, optimisation, backtest)
- `data/` : fichiers de données 
- `main.py` : point d'entrée simple pour lancer un backtest

## Installation

```bash
git clone https://github.com/ton-user/macro-regime-allocator.git
cd macro-regime-allocator
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
