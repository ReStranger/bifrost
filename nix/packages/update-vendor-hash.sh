#! /usr/bin/env bash

set -eux

cd "$(dirname "${BASH_SOURCE[0]}")"
:> vendorHash # Clear hash to trigger rebuild
BUILD_LOG="$(mktemp)"
nix build -L ../..#bifrost-http |& tee "$BUILD_LOG"
grep -oP 'got: +\K\S+' "$BUILD_LOG" > ./vendorHash