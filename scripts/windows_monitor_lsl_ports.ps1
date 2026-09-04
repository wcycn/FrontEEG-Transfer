param(
    [int]$MinimumPort = 16571,
    [int]$MaximumPort = 16604
)

$udp = Get-NetUDPEndpoint -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -ge $MinimumPort -and $_.LocalPort -le $MaximumPort } |
    Sort-Object LocalPort
$tcp = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -ge $MinimumPort -and $_.LocalPort -le $MaximumPort } |
    Sort-Object LocalPort

Write-Host "UDP endpoints in LSL range:"
$udp | Format-Table LocalAddress, LocalPort, OwningProcess -AutoSize
Write-Host "TCP listeners in LSL range:"
$tcp | Format-Table LocalAddress, LocalPort, OwningProcess -AutoSize

if ($null -eq $udp -and $null -eq $tcp) {
    Write-Warning "No active endpoints were visible. Start the vendor LSL output and run again."
}
