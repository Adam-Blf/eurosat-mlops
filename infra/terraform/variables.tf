variable "name" {
  description = "Prefix for every resource created."
  type        = string
  default     = "eurosat"
}

variable "ssh_public_key" {
  description = "Public key of the deploy user, for example the content of ~/.ssh/id_ed25519.pub."
  type        = string

  validation {
    condition     = can(regex("^ssh-(ed25519|rsa) ", var.ssh_public_key))
    error_message = "Give an OpenSSH public key (ssh-ed25519 or ssh-rsa), never a private key."
  }
}

variable "admin_cidrs" {
  description = "Networks allowed to reach SSH. Use your own address, for example 203.0.113.7/32."
  type        = list(string)

  validation {
    condition     = length(var.admin_cidrs) > 0 && !contains(var.admin_cidrs, "0.0.0.0/0")
    error_message = "SSH must be restricted: give at least one CIDR and never 0.0.0.0/0."
  }
}

variable "server_type" {
  description = "Hetzner server type. cx22 (2 vCPU, 4 GB) runs the whole compose stack."
  type        = string
  default     = "cx22"
}

variable "location" {
  description = "Hetzner location."
  type        = string
  default     = "fsn1"
}

variable "image" {
  description = "Operating system image."
  type        = string
  default     = "ubuntu-24.04"
}
