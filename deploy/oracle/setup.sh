#!/usr/bin/env bash
# One-time setup of a fresh Oracle Cloud VM (Ampere A1, Canonical Ubuntu, aarch64), run as the default "ubuntu" user over SSH:
#   curl -fsSL https://raw.githubusercontent.com/Astha222/Language-Annotation-Tool-with-NER-and-ML-Translation/main/deploy/oracle/setup.sh -o setup.sh
#   less setup.sh        # read it before running it
#   bash setup.sh
# It is safe to run again: every step checks first. See deploy/oracle/README.md for the steps in the Oracle console.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Astha222/Language-Annotation-Tool-with-NER-and-ML-Translation.git}"
DIR="${DIR:-$HOME/annotation-tool}"
SWAP_GB="${SWAP_GB:-4}"

echo "== 1/5 Docker"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sudo sh /tmp/get-docker.sh
  sudo usermod -aG docker "$USER"
fi
sudo docker --version

echo "== 2/5 Open ports 80 and 443 in the VM's own firewall"
# Oracle's Ubuntu images reject every incoming connection except SSH in iptables, even when the console's security list allows it.
# The new rules go before the first REJECT rule; the console's security list must allow 80 and 443 as well.
if ! command -v netfilter-persistent >/dev/null 2>&1; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y netfilter-persistent iptables-persistent
fi
for port in 80 443; do
  if ! sudo iptables -C INPUT -p tcp -m state --state NEW --dport "$port" -j ACCEPT 2>/dev/null; then
    reject_line="$(sudo iptables -L INPUT --line-numbers -n | awk '$2 == "REJECT" {print $1; exit}')"
    if [ -n "$reject_line" ]; then
      sudo iptables -I INPUT "$reject_line" -p tcp -m state --state NEW --dport "$port" -j ACCEPT
    else
      sudo iptables -A INPUT -p tcp -m state --state NEW --dport "$port" -j ACCEPT
    fi
  fi
done
sudo netfilter-persistent save

echo "== 3/5 Swap (${SWAP_GB} GB): a safety margin while models load"
if ! sudo swapon --show | grep -q '/swapfile'; then
  sudo fallocate -l "${SWAP_GB}G" /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
free -h

echo "== 4/5 Code"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only
else
  git clone "$REPO_URL" "$DIR"
fi

echo "== 5/5 Settings"
cd "$DIR/deploy/oracle"
if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  echo
  echo "Created $DIR/deploy/oracle/.env from the example. Fill it in (nano .env), then start everything with:"
  echo "  cd $DIR/deploy/oracle && sudo docker compose up -d --build"
  exit 0
fi
echo "Starting (the first build and the first model downloads take several minutes) ..."
sudo docker compose up -d --build
sudo docker compose ps
