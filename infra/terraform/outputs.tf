output "ipv4" {
  description = "Public IPv4 address of the VM. Use it as DEPLOY_HOST."
  value       = hcloud_server.this.ipv4_address
}

output "ssh_command" {
  description = "How to log in as the deploy user."
  value       = "ssh deploy@${hcloud_server.this.ipv4_address}"
}
