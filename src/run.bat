@echo off
echo ===================================================
echo INICIANDO BATERIA DE EXPERIMENTOS
echo ===================================================

echo.
echo [1/3] Ejecutando Placebo (Baseline - Seed 42)...
python experiment_runner.py --act placebo --condition baseline --seeds 42

echo.
echo [2/3] Ejecutando Single Agent (Seed 42)...
python experiment_runner.py --act single --seeds 42

echo.
echo [3/3] Ejecutando Baseline (Seeds 123 y 456)...
python experiment_runner.py --act 1 --condition baseline --seeds 123,456

echo.
echo ===================================================
echo ¡TODOS LOS EXPERIMENTOS HAN TERMINADO!
echo ===================================================
pause