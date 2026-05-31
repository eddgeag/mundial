# Predicción del Mundial 2026 con Machine Learning  
### Un proyecto serio sobre fútbol hecho por alguien que no quiere ver 90 minutos de fútbol

Este repositorio contiene un pipeline de análisis, modelado predictivo y simulación Monte Carlo para estimar resultados del Mundial 2026.

La idea general es simple: si yo no voy a ver todos los partidos, al menos que una máquina los sufra por mí.

---

## ¿Qué hace este proyecto?

Este proyecto construye modelos predictivos para partidos internacionales de fútbol usando variables históricas y externas, y luego simula el Mundial 2026 muchas veces para estimar probabilidades de:

- Resultado de cada partido: victoria local, empate o victoria visitante.
- Posición de cada selección en fase de grupos.
- Clasificación a rondas eliminatorias.
- Probabilidad de llegar a octavos, cuartos, semifinales, final y campeonato.
- Ranking de posibles campeones.
- Escenarios tipo “polla mundialista”, porque aparentemente eso también importa, aunque no me guste esa palabra. 

---

## Motivación

No me gusta el fútbol.

Pero sí me gustan:

- Los modelos predictivos.
- Las simulaciones Monte Carlo.
- Los pipelines reproducibles.
- XGBoost haciendo cosas que parecen magia.
- Ganar una polla mundialista sin fingir que sé quién juega de lateral derecho.

Entonces este proyecto existe.

---

## Estructura general

El flujo del proyecto es:

```text
datos/
├── training_matches_model.csv
├── fixtures_2026_master.csv
├── teams_2026_master.csv

scripts_final/
├── 00_config_server.sh
├── 00_test_v6_param_fast.sh
├── 01_train_v6_single_seed_param.R
├── 02_run_v6_seed_123_full.sh
├── 03_run_v6_multiseed.sh
├── in progress....

scripts_archive/
├── 06_train_models_v6_feature_engineered.R
├── 06D_monte_carlo_worldcup_v6_official_annexC.R

