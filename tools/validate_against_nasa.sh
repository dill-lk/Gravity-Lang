#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GRAVITY_BIN="$ROOT_DIR/build/gravity"
STRICT=0
[[ "${1:-}" == "--strict" ]] && STRICT=1

if [[ ! -x "$GRAVITY_BIN" ]]; then
  echo "error: gravity binary not found at $GRAVITY_BIN" >&2
  exit 2
fi

# NASA reference values
MOON_A_M=384400000
MERCURY_A_M=57910000000
MERCURY_E=0.2056

failures=0

extract_orbital_values() {
  local text="$1"
  echo "$text" | sed -n 's/.*semi_major_axis=\([^ ]*\) m, eccentricity=\([^ ]*\).*/\1 \2/p' | tail -n 1
}

rel_err() {
  awk -v measured="$1" -v expected="$2" 'BEGIN{d=measured-expected; if(d<0)d=-d; print d/expected}'
}

# Earth-Moon check (uses repo example)
earth_moon_out="$($GRAVITY_BIN run "$ROOT_DIR/examples/earth_moon.gravity")"
read -r moon_a moon_e < <(extract_orbital_values "$earth_moon_out")
moon_a_err="$(rel_err "$moon_a" "$MOON_A_M")"
moon_ok="$(awk -v ae="$moon_a_err" -v e="$moon_e" 'BEGIN{print (ae<=0.03 && e>=0 && e<0.1)?1:0}')"
if [[ "$moon_ok" -eq 1 ]]; then
  echo "[PASS] Earth-Moon: a=${moon_a} m (NASA ${MOON_A_M}, rel_err=$(awk -v x="$moon_a_err" 'BEGIN{printf "%.3f%%",x*100}')), e=${moon_e}"
else
  echo "[FAIL] Earth-Moon: a=${moon_a} m (NASA ${MOON_A_M}, rel_err=$(awk -v x="$moon_a_err" 'BEGIN{printf "%.3f%%",x*100}')), e=${moon_e}"
  failures=$((failures+1))
fi

# Mercury check (perihelion initialization from NASA orbital elements + vis-viva)
MU="$(awk 'BEGIN{printf "%.15e", 6.67430e-11*1.989e30}')"
R_PERI="$(awk -v a="$MERCURY_A_M" -v e="$MERCURY_E" 'BEGIN{printf "%.10f", a*(1-e)}')"
V_PERI="$(awk -v mu="$MU" -v a="$MERCURY_A_M" -v e="$MERCURY_E" 'BEGIN{printf "%.10f", sqrt(mu*(1+e)/(a*(1-e)))}')"

tmp_script="$(mktemp)"
cat > "$tmp_script" <<MERCURY_EOF
sphere Sun at [0,0,0] radius 696000[km] mass 1.989e30[kg] fixed
sphere Mercury at [${R_PERI},0,0][m] radius 2439[km] mass 3.285e23[kg]
Mercury.velocity = [0,${V_PERI},0][m/s]
simulate orbit in 0..3000 dt 3600[s] integrator leapfrog {
    Sun pull Mercury
}
orbital_elements Mercury around Sun
MERCURY_EOF

mercury_out="$($GRAVITY_BIN run "$tmp_script")"
rm -f "$tmp_script"
read -r mercury_a mercury_e < <(extract_orbital_values "$mercury_out")
mercury_a_err="$(rel_err "$mercury_a" "$MERCURY_A_M")"
mercury_e_err="$(rel_err "$mercury_e" "$MERCURY_E")"
mercury_a_pct="$(awk -v x="$mercury_a_err" 'BEGIN{printf "%.3f%%",x*100}')"
mercury_e_pct="$(awk -v x="$mercury_e_err" 'BEGIN{printf "%.3f%%",x*100}')"
mercury_ok="$(awk -v ae="$mercury_a_err" -v ee="$mercury_e_err" 'BEGIN{print (ae<=0.01 && ee<=0.02)?1:0}')"
if [[ "$mercury_ok" -eq 1 ]]; then
  echo "[PASS] Mercury: a=${mercury_a} m (NASA ${MERCURY_A_M}, rel_err=${mercury_a_pct}), e=${mercury_e} (NASA ${MERCURY_E}, rel_err=${mercury_e_pct})"
else
  echo "[FAIL] Mercury: a=${mercury_a} m (NASA ${MERCURY_A_M}, rel_err=${mercury_a_pct}), e=${mercury_e} (NASA ${MERCURY_E}, rel_err=${mercury_e_pct})"
  failures=$((failures+1))
fi

if [[ "$STRICT" -eq 1 && "$failures" -gt 0 ]]; then
  exit 1
fi
exit 0
