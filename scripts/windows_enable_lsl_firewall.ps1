#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

$rules = @(
    @{
        Name = "FrontEEG LSL discovery (UDP 16571)"
        Protocol = "UDP"
        Ports = "16571"
    },
    @{
        Name = "FrontEEG LSL data (UDP 16572-16604)"
        Protocol = "UDP"
        Ports = "16572-16604"
    },
    @{
        Name = "FrontEEG LSL data (TCP 16572-16604)"
        Protocol = "TCP"
        Ports = "16572-16604"
    }
)

foreach ($rule in $rules) {
    $existing = Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue
    if ($null -eq $existing) {
        New-NetFirewallRule `
            -DisplayName $rule.Name `
            -Direction Inbound `
            -Action Allow `
            -Protocol $rule.Protocol `
            -LocalPort $rule.Ports `
            -Profile Private | Out-Null
        Write-Host "Added: $($rule.Name)"
    }
    else {
        Write-Host "Already present: $($rule.Name)"
    }
}

Write-Host "LSL firewall rules are enabled for Private networks only."
