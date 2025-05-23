#!/usr/bin/env bash
set -e

cd ..
set -x
python3 -m tests.bench_all_to_all --dp-size=1
