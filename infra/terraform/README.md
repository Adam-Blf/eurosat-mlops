# Terraform: production VM on Hetzner Cloud

This directory creates one VM (Ubuntu 24.04, Docker and the compose plugin installed by
cloud-init), a firewall (SSH only from `admin_cidrs`, HTTP and HTTPS open) and the deploy SSH key.

Status: the configuration passes `terraform fmt -check` and `terraform validate`. It has **not**
been applied, because applying creates a paid resource (about 4 euros per month for a cx22).

## Apply

```bash
export HCLOUD_TOKEN=...            # read/write token from the Hetzner console, never committed
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # set ssh_public_key and admin_cidrs
terraform init
terraform plan
terraform apply
terraform output ipv4              # becomes the DEPLOY_HOST secret of the CD workflow
```

Destroy with `terraform destroy` when the demo is over. The state is local and ignored by git:
keep it outside the repository for anything beyond a school demo (remote backend).

## Choices

| Choice | Reason |
|---|---|
| Hetzner | Cheap, simple API, no cloud account tied to a card with credits |
| Firewall at the provider | Blocks traffic before it reaches the VM, independent of the OS |
| `admin_cidrs` cannot be `0.0.0.0/0` | A validation refuses an open SSH port |
| Non-root `deploy` user, key only | Matches `scripts/deploy.py`, no password login, root login off |
| `ignore_changes = [user_data]` | Editing cloud-init must not destroy a running VM |
