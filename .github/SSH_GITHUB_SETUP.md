# SSH key for Git push (Cursor / CI)

This repo does **not** store private keys in git. A one-time **Ed25519 deploy key** can live on disk under `.github/local-secrets/` (gitignored).

## If keys were generated for you

1. **Public key** (ends in `.pub`): add in GitHub → **Repository** → **Settings** → **Deploy keys** → **Add deploy key**
   - Title: e.g. `xplainverse-lj-workstation`
   - Paste contents of `github_deploy_ed25519.pub`
   - Enable **Allow write access** only if this key must `git push` to this repo.

2. **Private key** (no extension): use where authentication is required, for example:
   - **Cursor / local shell:** `GIT_SSH_COMMAND='ssh -i /path/to/github_deploy_ed25519'` or add `Host github.com` + `IdentityFile` in `~/.ssh/config`.
   - **GitHub Actions** that run `git push`: store the private key as a repository secret (e.g. `SSH_PRIVATE_KEY`) and load it in the workflow (prefer `GITHUB_TOKEN` + `contents: write` for normal CI pushes instead of SSH when possible).

3. Never commit `.github/local-secrets/` or paste the private key into issues/PRs.

## Generate a new key yourself

```bash
mkdir -p .github/local-secrets
ssh-keygen -t ed25519 -N "" -f .github/local-secrets/github_deploy_ed25519 -C "XPlainVerse $(hostname)"
```

Then repeat steps 1–2 with the new `.pub` / private files.

## GHCR login (containers)

Pushing Docker images to GHCR uses **`GITHUB_TOKEN`** in Actions (see `.github/workflows/container-lj.yml`). For `docker pull` / Apptainer on a cluster, use a **Personal Access Token** (classic: `read:packages`) or org robot account, not this SSH deploy key.
