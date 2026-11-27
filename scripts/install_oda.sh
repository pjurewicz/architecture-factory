#!/usr/bin/env bash
# Install the ODA File Converter (DWG read and write) and the libraries it needs
# to run headless (Qt6, xcb, Xvfb). Used by the orb setup (.agents/setup) and CI.
#
# ODA File Converter is a free download from opendesign.com, governed by the ODA
# license. ODA_DEB_DIR (default: a temporary folder) keeps the downloaded .deb,
# so CI can cache it between runs.
set -euo pipefail

# The system libraries the converter and its bundled Qt xcb plugin link (the
# same names on Debian 13 and Ubuntu 24.04); the rest are their dependencies.
apt_packages=(
  xvfb xauth
  libegl1 libgl1 libglx0 libopengl0 libfontconfig1 libglib2.0-0t64 libdbus-1-3
  libxkbcommon0 libxkbcommon-x11-0 libx11-xcb1
)
missing=()
for p in "${apt_packages[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
done
if ((${#missing[@]})); then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${missing[@]}"
fi

oda_file="ODAFileConverter_QT6_lnxX64_11dll.deb"
if ! dpkg -s odafileconverter >/dev/null 2>&1; then
  deb_dir="${ODA_DEB_DIR:-$(mktemp -d)}"
  mkdir -p "$deb_dir"
  if [[ ! -s "$deb_dir/$oda_file" ]]; then
    curl -fsSL -o "$deb_dir/$oda_file" "https://www.opendesign.com/guestfiles/get?filename=$oda_file"
  fi
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$deb_dir/$oda_file"
  [[ -n "${ODA_DEB_DIR:-}" ]] || rm -rf "$deb_dir"
fi
command -v ODAFileConverter

# Fail here, naming the library, rather than as exit code 127 inside a test.
oda_dir="$(dirname "$(readlink -f "$(ls -d /usr/bin/ODAFileConverter_*/ODAFileConverter | head -1)")")"
unresolved="$(ldd "$oda_dir/ODAFileConverter" "$oda_dir"/plugins/platforms/*.so | grep 'not found' | sort -u || true)"
if [[ -n "$unresolved" ]]; then
  echo "ODA File Converter is missing system libraries:" >&2
  echo "$unresolved" >&2
  exit 1
fi
