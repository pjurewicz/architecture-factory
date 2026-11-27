#!/usr/bin/env bash
# Install the ODA File Converter (DWG read and write) and the libraries it needs
# to run headless (Qt6, xcb, Xvfb). Used by the orb setup (.agents/setup) and CI.
#
# ODA File Converter is a free download from opendesign.com, governed by the ODA
# license. ODA_DEB_DIR (default: a temporary folder) keeps the downloaded .deb,
# so CI can cache it between runs.
set -euo pipefail

apt_packages=(xvfb xauth libopengl0 libxkbcommon-x11-0)
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
