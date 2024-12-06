#!/bin/bash

HERE="$(dirname "$0")"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

bash "$HERE/run_base.sh"
bash "$HERE/run_enumerate.sh"
bash "$HERE/run_additional.sh"
