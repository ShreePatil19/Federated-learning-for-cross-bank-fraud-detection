@echo off
title MoE-FL Dashboard
cd /d "D:\Masters\Sem 1\Neural Network and Fuzzy Logic\PROJECT\moe-fl-per-dataset-alpha-sweep-GROUP-A\06_reports"
echo Starting MoE-FL dashboard at http://localhost:8501 ...
py -m streamlit run dashboard.py --server.port 8501 --browser.gatherUsageStats false
pause
