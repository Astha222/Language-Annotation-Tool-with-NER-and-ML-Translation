# Backend on an Oracle Cloud "Always Free" VM

The backend, PostgreSQL, MongoDB and Caddy (HTTPS) run with Docker Compose on one Arm VM; the frontend is a static site on Vercel
(see "Deploying" in the main README). Why Oracle: the backend needs about 3-5 GB of memory for its models, and Oracle's Always Free
Arm allowance (2 OCPUs and 12 GB, plus 200 GB of block storage, in the home region, for the life of the account) is the only free offer found
that is large enough (checked 2026-11-26; free tiers of Render, Koyeb and Railway give about 0.5 GB, and Hugging Face Docker Spaces now need a paid plan).

> These files were checked locally (`docker compose config` and a shell syntax check) but **not yet on a real Oracle VM**. Report what differs.

## 1. Account (once)
- Sign up at oracle.com/cloud/free. Oracle asks for a card to verify identity; Always Free resources are not charged. **Do not upgrade** to
  Pay As You Go unless you mean to: then resources outside the Always Free limits are billed.
- **Home region:** chosen at sign-up and cannot be changed; Always Free VMs exist only there. For India: Mumbai or Hyderabad.

## 2. The VM (Oracle console)
1. **Compute > Instances > Create instance.**
2. **Image:** Canonical Ubuntu 24.04. **Shape:** Ampere, `VM.Standard.A1.Flex`, **2 OCPUs, 12 GB memory**.
3. **Networking:** keep "assign a public IPv4 address". **SSH keys:** "Generate a key pair" and **download the private key** (you cannot download it later).
4. **Boot volume:** 100 GB is enough (the 200 GB Always Free storage is shared by all volumes).
5. "Out of host capacity" means Oracle has no free Arm machine in that availability domain right now: try another availability domain, or again later.
6. Open the ports: **Networking > Virtual cloud networks > (your VCN) > Subnets > (subnet) > Security lists > Default security list > Add ingress rules:**
   source `0.0.0.0/0`, TCP, destination port `80`; and the same for `443`. (The VM's own firewall is opened by `setup.sh`.)

## 3. Install (on the VM)
```bash
ssh -i <downloaded-key> ubuntu@<public-ip>
curl -fsSL https://raw.githubusercontent.com/Astha222/Language-Annotation-Tool-with-NER-and-ML-Translation/main/deploy/oracle/setup.sh -o setup.sh
less setup.sh                    # read it first
bash setup.sh                    # Docker, firewall ports 80/443, 4 GB swap, the code, and deploy/oracle/.env
nano ~/annotation-tool/deploy/oracle/.env
```
In `.env`:
- `API_DOMAIN`: the public IP with dashes plus `.sslip.io`, e.g. IP `129.154.1.2` -> `129-154-1-2.sslip.io` (no sign-up needed). If the certificate fails, use a free DuckDNS name pointing to the IP instead.
- `POSTGRES_PASSWORD`: `openssl rand -hex 24`; `JWT_SECRET`: `openssl rand -hex 32`. New values, never the laptop's.
- `CORS_ORIGINS`: the Vercel address (fill in after step 4; until then the backend works but the website cannot call it).
- `HF_TOKEN`: a Hugging Face **read** token while the English model's repository is private.

Then:
```bash
cd ~/annotation-tool/deploy/oracle
sudo docker compose up -d --build          # first build and first model downloads: several minutes
sudo docker compose logs -f backend        # wait for the startup lines; Ctrl+C stops watching, not the server
curl https://<API_DOMAIN>/                 # {"message":"Backend is running!", ...}
```

## 4. Frontend on Vercel
Set `EXPO_PUBLIC_API_URL=https://<API_DOMAIN>` in the Vercel project and deploy (main README, "Deploying"). Put the Vercel address into
`CORS_ORIGINS` in `.env`, then `sudo docker compose up -d` (recreates the backend with the new setting).

## Day to day
- **Update:** `cd ~/annotation-tool && git pull && cd deploy/oracle && sudo docker compose up -d --build`
- **Logs / state:** `sudo docker compose logs --tail 100 backend`, `sudo docker compose ps`, `free -h`, `df -h`
- **Database backup:** `sudo docker compose exec postgres pg_dump -U annotation annotationtool > backup.sql` (copy it off the VM)
- **Idle reclamation:** Oracle may reclaim an Always Free VM when, over 7 days, CPU (95th percentile), network **and** memory (A1 only) are all below 20%.
  `.env` preloads two models at start, which keeps memory above 20% of 12 GB; keep it that way.
- **Memory:** about 1 GB per NER model and several hundred MB per offline translation model; `NER_MAX_MODELS` and `LOCAL_MT_MAX_LOADED` in `.env` cap them.
