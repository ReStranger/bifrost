#!/usr/bin/env bash
set -euo pipefail

TAG="${RELEASE_TAG}"
VERSION_ONLY="${TAG#transports/v}"

git checkout --detach "$TAG"

set +e
BUILD_LOG=$(mktemp)
nix build -L .#bifrost-http 2>"$BUILD_LOG"
STATUS=$?
set -e

if [ $STATUS -eq 0 ]; then
echo "Unexpected: build succeeded with fake hash" >&2
exit 1
fi

GOT=$(grep -Eo 'got: sha256-[0-9A-Za-z+/=]+' "$BUILD_LOG" | head -n 1 | awk '{print $2}')
if [ -z "${GOT:-}" ]; then
echo "Failed to parse vendorHash from nix build output" >&2
echo "--- nix build stderr ---" >&2
cat "$BUILD_LOG" >&2
exit 1
fi

echo "$GOT" >> nix/packages/vendorHash

nix build -L .#bifrost-http

echo "version=$VERSION_ONLY" >> "$GITHUB_OUTPUT"
echo "vendorHash=$GOT" >> "$GITHUB_OUTPUT"