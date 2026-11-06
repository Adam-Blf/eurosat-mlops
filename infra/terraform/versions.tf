terraform {
  required_version = ">= 1.6"

  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = "~> 1.48"
    }
  }
}

# The API token is read from the HCLOUD_TOKEN environment variable. Nothing secret lives
# in this directory, and the state file stays out of git (see .gitignore).
provider "hcloud" {}
