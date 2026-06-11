#!/usr/bin/env bash
#
# install_into_ns3.sh — drop the video-mpquic ns3-ai example into an ns-3 tree
# that already has ns3-ai cloned at contrib/ai, and apply the small patches this
# branch needs. Idempotent: safe to re-run.
#
# Usage:
#   ns3/install_into_ns3.sh [NS3_DIR]
# NS3_DIR defaults to $NS3_DIR or ~/ns-3-dev. See ns3/README.md for the full
# build walkthrough.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS3_DIR="${1:-${NS3_DIR:-$HOME/ns-3-dev}}"
AI_DIR="$NS3_DIR/contrib/ai"
EXAMPLES_DIR="$AI_DIR/examples"
DEST="$EXAMPLES_DIR/video-mpquic"

if [[ ! -d "$AI_DIR" ]]; then
  echo "error: $AI_DIR not found. Clone ns3-ai into contrib/ai first:" >&2
  echo "  git clone https://github.com/hust-diangroup/ns3-ai.git $AI_DIR" >&2
  exit 1
fi

# 1) Copy the example sources.
mkdir -p "$DEST"
cp "$REPO_DIR"/ns3/examples/video-mpquic/{video_mpquic.h,video_mpquic.cc,video_mpquic_py.cc,CMakeLists.txt} "$DEST/"
echo "copied example -> $DEST"

# 2) Register the example subdirectory.
if ! grep -q "add_subdirectory(video-mpquic)" "$EXAMPLES_DIR/CMakeLists.txt"; then
  echo "add_subdirectory(video-mpquic)" >> "$EXAMPLES_DIR/CMakeLists.txt"
  echo "registered video-mpquic in examples/CMakeLists.txt"
else
  echo "video-mpquic already registered"
fi

# 3) Patch ns3-ai's CMakeLists: it requires Boost's program_options component but
#    only uses header-only boost/interprocess. Require headers only so a
#    header-only boost (e.g. conda libboost-headers) satisfies it.
AI_CMAKE="$AI_DIR/CMakeLists.txt"
if grep -q "find_package(Boost REQUIRED COMPONENTS program_options)" "$AI_CMAKE"; then
  sed -i 's/find_package(Boost REQUIRED COMPONENTS program_options)/find_package(Boost REQUIRED)/' "$AI_CMAKE"
  echo "patched $AI_CMAKE (Boost: headers only)"
else
  echo "Boost find_package already patched (or differs) in $AI_CMAKE"
fi

echo
echo "Done. Next:"
echo "  cd $NS3_DIR && ./ns3 build ns3ai_video_mpquic"
