#!/usr/bin/env bash
# Reproduce every experiment in the paper. Stages can be run separately: bash scripts/run_all.sh <stage>
# Stages: forecast | heterogeneity | siting | permutation | ceiling | peak | models | checks | summary | all
# Independent jobs inside a stage run in parallel (JOBS, default 8). Logs go to outputs/logs/.
# The MILPs are CPU-bound; one GPU is only useful for the forecasting stage.
set -u
ROOT=${EV_RFP_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
export EV_RFP_ROOT=$ROOT
export PY=${PY:-python}
JOBS=${JOBS:-8}
SRC=$ROOT/src
LOG=$ROOT/outputs/logs
mkdir -p "$LOG"
CITIES="LOA SZH MEL"

par() {  # run the job lines on stdin in parallel from src/
  (cd "$SRC" && xargs -d '\n' -P "$JOBS" -I{} bash -c '{}')
}

stage_forecast() {
  (cd "$SRC" && EVDATA=urbanev $PY exp1_forecast.py --splits A B > "$LOG/exp1_urbanev.txt" 2>&1)
  (cd "$SRC" && $PY preprocess_charged.py > "$LOG/preprocess_charged.txt" 2>&1)
  for c in $CITIES; do
    (cd "$SRC" && EVDATA=charged_$c $PY exp1_forecast.py --models SNaive DLinear iTransformer --splits C \
      > "$LOG/exp1_$c.txt" 2>&1)
  done
}

stage_heterogeneity() {
  {
    echo "EVDATA=urbanev \$PY exp2_gradient.py > $LOG/exp2_urbanev.txt 2>&1"
    for c in $CITIES; do echo "EVDATA=charged_$c \$PY exp2_gradient.py > $LOG/exp2_$c.txt 2>&1"; done
  } | par
}

stage_siting() {  # four plans under observed, uniform and rank-flipped radii
  {
    for s in A B; do
      echo "EVDATA=urbanev \$PY exp3_siting.py --split $s --gammas 1 2 3 --taus 0.3 0.4 0.5 > $LOG/exp3_$s.txt 2>&1"
    done
    echo "EVDATA=urbanev \$PY exp3_siting.py --split A --model DLinear --gammas 1 2 3 --taus 0.5 > $LOG/exp3_A_DLinear.txt 2>&1"
    for s in A B; do
      for v in "--radius_km 2 --tag _R2" "--radius_km 4 --tag _R4" "--fmult 1 --tag _F1" "--fmult 4 --tag _F4"; do
        echo "EVDATA=urbanev \$PY exp3_siting.py --split $s --gammas 2 --modes real uniform $v > $LOG/exp3_${s}_${v##*_}.txt 2>&1"
      done
      echo "EVDATA=urbanev \$PY exp3_siting.py --split $s --gammas 2 --taus 0.5 --modes real uniform --grids M1 M2 M3 --feeders 50 --base_scale 0.4 --tag _grid > $LOG/exp3_${s}_grid.txt 2>&1"
    done
    for c in $CITIES; do
      echo "EVDATA=charged_$c \$PY exp3_siting.py --split C --gammas 2 --taus 0.5 > $LOG/exp3_$c.txt 2>&1"
    done
  } | par
}

stage_permutation() {  # random error layouts (30 per setting, 100 for the UrbanEV baseline)
  {
    for s in A B; do
      echo "EVDATA=urbanev \$PY exp4_alignment.py --split $s > $LOG/exp4_$s.txt 2>&1"
      echo "EVDATA=urbanev \$PY exp4_alignment.py --split $s --perms 100 --tag _p100 > $LOG/exp4_${s}_p100.txt 2>&1"
      echo "EVDATA=urbanev \$PY exp4_alignment.py --split $s --model DLinear > $LOG/exp4_${s}_DLinear.txt 2>&1"
      for gtb in "1 0.4 0.7" "2 0.4 0.7" "3 0.4 0.7" "2 0.4 0.6" "2 0.4 0.8" "2 0.3 0.7"; do
        set -- $gtb
        echo "EVDATA=urbanev \$PY exp4_alignment.py --split $s --gamma $1 --tau $2 --beta $3 --tag _t$2_b$3 > $LOG/exp4_${s}_g$1_t$2_b$3.txt 2>&1"
      done
      for v in "--radius_km 2 --tag _R2" "--radius_km 4 --tag _R4" "--fmult 1 --tag _F1" "--fmult 4 --tag _F4"; do
        echo "EVDATA=urbanev \$PY exp4_alignment.py --split $s $v > $LOG/exp4_${s}_${v##*_}.txt 2>&1"
      done
    done
    for c in $CITIES; do echo "EVDATA=charged_$c \$PY exp4_alignment.py --split C > $LOG/exp4_$c.txt 2>&1"; done
  } | par
}

stage_ceiling() {  # fairness ceiling tau* without and with feeder limits
  {
    echo "EVDATA=urbanev \$PY exp5_fair_ceiling.py > $LOG/exp5_urbanev.txt 2>&1"
    for s in A B; do
      for v in "--vmin 0.95 --tag _v95" "--vmin 0.90 --tag _v90" "--feeders 30 --tag _K30" "--base_scale 0.6 --tag _bs60"; do
        echo "EVDATA=urbanev \$PY exp5_fair_ceiling.py --splits $s --grids M1 M2 M3 --modes real $v""_$s > $LOG/exp5_${s}_${v##*_}.txt 2>&1"
      done
    done
    for c in LOA SZH; do
      echo "EVDATA=charged_$c \$PY exp5_fair_ceiling.py --splits C --grids none --modes real uniform flipped > $LOG/exp5_$c.txt 2>&1"
    done
  } | par
}

stage_peak() {  # peak-consistent radii (RADIUS=peak)
  {
    for s in A B; do
      echo "RADIUS=peak EVDATA=urbanev \$PY exp3_siting.py --split $s --gammas 1 2 3 --taus 0.3 0.4 0.5 --tag _peak > $LOG/exp3_${s}_peak.txt 2>&1"
      echo "RADIUS=peak EVDATA=urbanev \$PY exp5_fair_ceiling.py --splits $s --tag _peak_$s > $LOG/exp5_${s}_peak.txt 2>&1"
    done
    for c in LOA SZH; do
      echo "RADIUS=peak EVDATA=charged_$c \$PY exp3_siting.py --split C --gammas 2 --taus 0.3 0.4 0.5 --tag _peak > $LOG/exp3_${c}_peak.txt 2>&1"
      echo "RADIUS=peak EVDATA=charged_$c \$PY exp5_fair_ceiling.py --splits C --grids none --modes real uniform --tag _peak > $LOG/exp5_${c}_peak.txt 2>&1"
    done
    for job in "urbanev A 0.5" "urbanev A 0.4" "urbanev B 0.4" "urbanev B 0.3" \
               "charged_SZH C 0.5" "charged_SZH C 0.4" "charged_LOA C 0.4" "charged_LOA C 0.3"; do
      set -- $job
      echo "RADIUS=peak EVDATA=$1 \$PY exp4_alignment.py --split $2 --tau $3 --tag _peak_t$3 > $LOG/exp4_$1_$2_peak_t$3.txt 2>&1"
    done
  } | par
}

stage_models() {  # weaker forecasters: SNaive, GRU, Graph-GRU
  {
    for s in A B; do
      for m in SNaive GRU GraphGRU; do
        echo "EVDATA=urbanev \$PY exp3_siting.py --split $s --model $m --gammas 2 --taus 0.3 0.4 0.5 --tag _models > $LOG/exp3_${s}_$m.txt 2>&1"
        echo "EVDATA=urbanev \$PY exp4_alignment.py --split $s --model $m --tau 0.4 --tag _t0.4 > $LOG/exp4_${s}_${m}_t0.4.txt 2>&1"
        echo "EVDATA=urbanev \$PY exp5_fair_ceiling.py --splits $s --model $m --grids none --tag _models_$s > $LOG/exp5_${s}_$m.txt 2>&1"
      done
    done
  } | par
}

stage_checks() {  # radius coverage, ex ante dodgeable share and tightness of the dodging bound
  {
    echo "EVDATA=urbanev \$PY diagnostics.py --splits A B --models iTransformer DLinear > $LOG/diag_urbanev.txt 2>&1"
    for c in LOA SZH; do
      echo "EVDATA=charged_$c \$PY diagnostics.py --splits C --models iTransformer > $LOG/diag_$c.txt 2>&1"
    done
    for s in A B; do
      echo "EVDATA=urbanev \$PY bound_check.py --split $s --file alignment_${s}_iTransformer_g2.csv > $LOG/bound_$s.txt 2>&1"
    done
    for c in LOA SZH; do
      echo "EVDATA=charged_$c \$PY bound_check.py --split C --file alignment_C_iTransformer_g2.csv > $LOG/bound_$c.txt 2>&1"
    done
  } | par
  for s in A B; do  # wall time of one baseline run of the four plans (run alone for a clean measurement)
    S0=$SECONDS
    (cd "$SRC" && EVDATA=urbanev $PY exp3_siting.py --split $s --gammas 2 --taus 0.5 --modes real --tag _timing \
      > "$LOG/timing_$s.txt" 2>&1)
    echo "split $s four plans wall seconds $((SECONDS - S0))" >> "$ROOT/outputs/timing_summary.txt"
  done
}

stage_summary() {
  (cd "$ROOT/analysis" && $PY summarize_results.py > "$ROOT/outputs/summary.txt" 2>&1)
  (cd "$ROOT/analysis" && $PY make_figures.py > "$LOG/figures.txt" 2>&1)
}

case "${1:-all}" in
  all)
    for st in forecast heterogeneity siting permutation ceiling peak models checks summary; do
      echo "== $st $(date)"; "stage_$st"
    done ;;
  *) "stage_$1" ;;
esac
