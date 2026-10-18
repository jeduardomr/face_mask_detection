#!/usr/bin/env bash
# Verificacion completa: corre los tres checks en orden.
# Sin pytest, sin GPU, sin camara, sin dataset. Tarda ~3 s.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

status=0
for check in tests/test_constraints.py tests/test_state_machine.py tests/test_split.py; do
    echo "── $check"
    if python3 "$check" | tail -20; then
        :
    else
        status=1
    fi
    echo
done

if [ "$status" -eq 0 ]; then
    echo "TODO VERDE"
else
    echo "HAY FALLOS"
fi
exit "$status"