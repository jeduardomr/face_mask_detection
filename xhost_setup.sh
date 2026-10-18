#!/usr/bin/env bash
# `xhost +local:root` abre el servidor X a cualquier root local: permite leer el
# teclado o inyectar eventos, y persistirlo lo deja abierto para siempre.
#
# Uso:  bash xhost_setup.sh          # solo comprueba
#       bash xhost_setup.sh --fix    # anade el permiso a ~/.bashrc
set -uo pipefail

RULE='xhost +local:root'
BASHRC="$HOME/.bashrc"

if grep -qF "$RULE" "$BASHRC" 2>/dev/null; then
    echo "OK  $RULE ya esta en $BASHRC"
else
    echo "AVISO  $RULE no esta en $BASHRC"
    if [ "${1:-}" = "--fix" ]; then
        printf '\n# Permiso X11 para la ventana de la camara en Docker\n%s\n' "$RULE" >>"$BASHRC"
        echo "Anadido a $BASHRC. Abre una terminal nueva o ejecuta: $RULE"
    else
        echo "Ejecuta '$RULE' en esta terminal, o 'bash xhost_setup.sh --fix' para persistirlo."
    fi
fi

[ -e /dev/video0 ] || echo "AVISO  No existe /dev/video0; anadelo al servicio epp-detector en docker-compose.yml"