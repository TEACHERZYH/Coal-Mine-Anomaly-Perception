param(
    [Parameter(Mandatory = $true)]
    [string]$Uri,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [Parameter(Mandatory = $true)]
    [long]$TotalBytes,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{32}$')]
    [string]$ExpectedMd5,

    [ValidateRange(1, 16)]
    [int]$Parallelism = 8,

    [ValidateRange(1048576, 1073741824)]
    [long]$SegmentBytes = 268435456,

    [ValidateRange(60, 7200)]
    [int]$SignedTransferMaxTimeSeconds = 1800,

    [switch]$NoProxy,

    [switch]$SignedUrlDirect
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$outputFull = [IO.Path]::GetFullPath($OutputPath)
$outputDirectory = [IO.Path]::GetDirectoryName($outputFull)
$partRoot = "$outputFull.range-parts"
$partRootFull = [IO.Path]::GetFullPath($partRoot)

if ([IO.Path]::GetDirectoryName($partRootFull) -ne $outputDirectory -or
    -not $partRootFull.EndsWith('.range-parts', [StringComparison]::Ordinal)) {
    throw "Unsafe range-parts path: $partRootFull"
}
if ($TotalBytes -le 0) {
    throw 'TotalBytes must be positive'
}
if (Test-Path -LiteralPath $outputFull) {
    throw "Output already exists: $outputFull"
}

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $partRootFull | Out-Null

$segments = [Collections.Generic.List[object]]::new()
for ($start = [long]0; $start -lt $TotalBytes; $start += $SegmentBytes) {
    $end = [Math]::Min($start + $SegmentBytes - 1, $TotalBytes - 1)
    $segments.Add([pscustomobject]@{
        Start = $start
        End = $end
        ExpectedBytes = $end - $start + 1
        Path = Join-Path $partRootFull ("{0:D12}-{1:D12}.part" -f $start, $end)
    })
}

$bypassProxy = [bool]$NoProxy
$signedUrlDirect = [bool]$SignedUrlDirect
$manifestPath = Join-Path $partRootFull 'ranges.json'
$downloaderSha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
$manifestPayload = [ordered]@{
    schema_version = 2
    downloader_sha256 = $downloaderSha256
    uri = $Uri
    total_bytes = $TotalBytes
    segment_bytes = $SegmentBytes
    expected_md5 = $ExpectedMd5.ToLowerInvariant()
    transfer_mode = if ($signedUrlDirect) { 'signed_url_direct' } else { 'redirect_follow' }
    segments = $segments
}
$existingRangeArtifacts = @(
    Get-ChildItem -LiteralPath $partRootFull -File -ErrorAction SilentlyContinue
)
if (Test-Path -LiteralPath $manifestPath) {
    try {
        $existingManifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $manifestMatches = (
            [int]$existingManifest.schema_version -eq 2 -and
            [string]$existingManifest.downloader_sha256 -eq $downloaderSha256 -and
            [string]$existingManifest.uri -eq $Uri -and
            [long]$existingManifest.total_bytes -eq $TotalBytes -and
            [long]$existingManifest.segment_bytes -eq $SegmentBytes -and
            [string]$existingManifest.expected_md5 -eq $ExpectedMd5.ToLowerInvariant() -and
            [string]$existingManifest.transfer_mode -eq $manifestPayload.transfer_mode -and
            @($existingManifest.segments).Count -eq $segments.Count
        )
    } catch {
        $manifestMatches = $false
    }
    if (-not $manifestMatches) {
        throw 'Existing range parts belong to a different downloader contract; quarantine them before retrying'
    }
} else {
    if ($existingRangeArtifacts.Count -ne 0) {
        throw 'Existing range artifacts lack a downloader contract; quarantine them before retrying'
    }
    $manifestPayload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8
}

$results = @($segments | ForEach-Object -Parallel {
    $ErrorActionPreference = 'Stop'
    if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
        $PSNativeCommandUseErrorActionPreference = $false
    }

    function Invoke-NativeProcess {
        param(
            [Parameter(Mandatory = $true)]
            [string]$FilePath,

            [Parameter(Mandatory = $true)]
            [string[]]$Arguments,

            [switch]$CaptureOutput
        )

        $startInfo = [Diagnostics.ProcessStartInfo]::new()
        $startInfo.FileName = $FilePath
        $startInfo.UseShellExecute = $false
        foreach ($argument in $Arguments) {
            $startInfo.ArgumentList.Add($argument)
        }
        if ($CaptureOutput) {
            $startInfo.RedirectStandardOutput = $true
            $startInfo.RedirectStandardError = $true
        }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $startInfo
        if (-not $process.Start()) {
            throw "Could not start native process: $FilePath"
        }
        if ($CaptureOutput) {
            $standardOutput = $process.StandardOutput.ReadToEndAsync()
            $standardError = $process.StandardError.ReadToEndAsync()
        }
        $process.WaitForExit()
        $result = [pscustomobject]@{
            ExitCode = $process.ExitCode
            StandardOutput = if ($CaptureOutput) { $standardOutput.Result } else { '' }
            StandardError = if ($CaptureOutput) { $standardError.Result } else { '' }
        }
        $process.Dispose()
        return $result
    }

    function Merge-Fragment {
        param(
            [Parameter(Mandatory = $true)]
            [string]$FragmentPath,

            [Parameter(Mandatory = $true)]
            [string]$TemporaryPath,

            [Parameter(Mandatory = $true)]
            [long]$ExpectedBytes
        )

        if (-not (Test-Path -LiteralPath $FragmentPath)) {
            return
        }
        $fragmentLength = (Get-Item -LiteralPath $FragmentPath).Length
        $downloadedLength = if (Test-Path -LiteralPath $TemporaryPath) {
            (Get-Item -LiteralPath $TemporaryPath).Length
        } else {
            [long]0
        }
        $remaining = $ExpectedBytes - $downloadedLength
        if ($fragmentLength -gt $remaining) {
            throw "Signed range fragment exceeds the remaining segment size: $FragmentPath"
        }
        if ($fragmentLength -gt 0) {
            $destination = [IO.File]::Open(
                $TemporaryPath,
                [IO.FileMode]::Append,
                [IO.FileAccess]::Write
            )
            $source = [IO.File]::OpenRead($FragmentPath)
            try {
                $source.CopyTo($destination, 8388608)
            } finally {
                $source.Dispose()
                $destination.Dispose()
            }
        }
        Remove-Item -LiteralPath $FragmentPath -Force
    }

    function Assert-FragmentResponse {
        param(
            [Parameter(Mandatory = $true)]
            [string]$HeaderPath,

            [Parameter(Mandatory = $true)]
            [string]$FragmentPath,

            [Parameter(Mandatory = $true)]
            [long]$RequestStart,

            [Parameter(Mandatory = $true)]
            [long]$RequestEnd,

            [Parameter(Mandatory = $true)]
            [long]$TotalBytes
        )

        if (-not (Test-Path -LiteralPath $FragmentPath) -or
            (Get-Item -LiteralPath $FragmentPath).Length -eq 0) {
            return
        }
        if (-not (Test-Path -LiteralPath $HeaderPath)) {
            throw "Range fragment lacks response headers: $FragmentPath"
        }
        $headers = Get-Content -LiteralPath $HeaderPath
        $status = @($headers | Where-Object { $_ -match '^HTTP/' }) | Select-Object -Last 1
        if ($status -notmatch '^HTTP/\S+\s+206(?:\s|$)') {
            throw "Range fragment response is not HTTP 206: $status"
        }
        $contentRangeLine = @(
            $headers | Where-Object { $_ -match '^Content-Range:' }
        ) | Select-Object -Last 1
        $contentRange = [string]$contentRangeLine -replace '^Content-Range:\s*', ''
        $expectedContentRange = "bytes $RequestStart-$RequestEnd/$TotalBytes"
        if ($contentRange.Trim() -ne $expectedContentRange) {
            throw "Range fragment Content-Range mismatch: expected $expectedContentRange, got $contentRange"
        }
    }

    $segment = $_
    $existingLength = if (Test-Path -LiteralPath $segment.Path) {
        (Get-Item -LiteralPath $segment.Path).Length
    } else {
        [long]0
    }
    if ($existingLength -ne $segment.ExpectedBytes) {
        $temporary = "$($segment.Path).tmp"
        if (-not $using:signedUrlDirect) {
            Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
            $range = "$($segment.Start)-$($segment.End)"
            $curlArguments = @(
                '--location', '--fail', '--show-error', '--silent',
                '--retry', '20', '--retry-all-errors', '--retry-delay', '5',
                '--range', $range, $using:Uri, '--output', $temporary
            )
            if ($using:bypassProxy) {
                $curlArguments = @('--noproxy', '*') + $curlArguments
            }
            $curlResult = Invoke-NativeProcess -FilePath 'curl.exe' -Arguments $curlArguments
            if ($curlResult.ExitCode -ne 0) {
                throw "curl failed for byte range $range with exit code $($curlResult.ExitCode)"
            }
        } else {
            for ($attempt = 1; $attempt -le 20; $attempt++) {
                $fragment = "$temporary.fragment"
                $fragmentHeaders = "$fragment.headers"
                Remove-Item -LiteralPath $fragment -Force -ErrorAction SilentlyContinue
                Remove-Item -LiteralPath $fragmentHeaders -Force -ErrorAction SilentlyContinue
                $downloaded = if (Test-Path -LiteralPath $temporary) {
                    (Get-Item -LiteralPath $temporary).Length
                } else {
                    [long]0
                }
                if ($downloaded -eq $segment.ExpectedBytes) {
                    break
                }
                if ($downloaded -gt $segment.ExpectedBytes) {
                    throw "Temporary range file exceeds its expected size: $temporary"
                }
                $requestStart = $segment.Start + $downloaded
                $range = "$requestStart-$($segment.End)"
                $headerResult = Invoke-NativeProcess -FilePath 'curl.exe' -Arguments @(
                    '--silent', '--show-error', '--head',
                    '--retry', '5', '--retry-all-errors', '--retry-delay', '2',
                    $using:Uri
                ) -CaptureOutput
                $headers = $headerResult.StandardOutput -split "`r?`n"
                $headerExitCode = $headerResult.ExitCode
                if ($headerExitCode -ne 0) {
                    if ($attempt -eq 20) {
                        throw "Could not obtain a signed URL for byte range $range"
                    }
                    Start-Sleep -Seconds 5
                    continue
                }
                $locationLine = @($headers | Where-Object { $_ -match '^location:' }) |
                    Select-Object -Last 1
                $signedUrl = [string]$locationLine -replace '^location:\s*', ''
                if (-not $signedUrl) {
                    if ($attempt -eq 20) {
                        throw "Signed URL response lacks a Location header for byte range $range"
                    }
                    Start-Sleep -Seconds 5
                    continue
                }
                $fragmentResult = Invoke-NativeProcess -FilePath 'curl.exe' -Arguments @(
                    '--noproxy', '*', '--ipv4', '--fail', '--show-error', '--silent',
                    '--connect-timeout', '30',
                    '--max-time', [string]$using:SignedTransferMaxTimeSeconds,
                    '--range', $range, '--dump-header', $fragmentHeaders,
                    $signedUrl, '--output', $fragment
                )
                $fragmentExitCode = $fragmentResult.ExitCode
                try {
                    Assert-FragmentResponse -HeaderPath $fragmentHeaders `
                        -FragmentPath $fragment `
                        -RequestStart $requestStart `
                        -RequestEnd $segment.End `
                        -TotalBytes $using:TotalBytes
                } catch {
                    Remove-Item -LiteralPath $fragment -Force -ErrorAction SilentlyContinue
                    throw
                } finally {
                    Remove-Item -LiteralPath $fragmentHeaders -Force -ErrorAction SilentlyContinue
                }
                Merge-Fragment -FragmentPath $fragment `
                    -TemporaryPath $temporary `
                    -ExpectedBytes $segment.ExpectedBytes
                $downloaded = if (Test-Path -LiteralPath $temporary) {
                    (Get-Item -LiteralPath $temporary).Length
                } else {
                    [long]0
                }
                if ($downloaded -eq $segment.ExpectedBytes) {
                    break
                }
                if ($attempt -eq 20) {
                    throw "Signed direct download failed for byte range $range with exit code $fragmentExitCode"
                }
                Start-Sleep -Seconds 5
            }
        }
        $downloadedLength = (Get-Item -LiteralPath $temporary).Length
        if ($downloadedLength -ne $segment.ExpectedBytes) {
            throw "Range $range has $downloadedLength bytes; expected $($segment.ExpectedBytes)"
        }
        Move-Item -LiteralPath $temporary -Destination $segment.Path -Force
    }
    [pscustomobject]@{
        Start = $segment.Start
        End = $segment.End
        Bytes = (Get-Item -LiteralPath $segment.Path).Length
        Status = 'ready'
    }
} -ThrottleLimit $Parallelism)

if ($results.Count -ne $segments.Count -or
    @($results | Where-Object Status -ne 'ready').Count -ne 0) {
    throw 'Not all ranges completed successfully'
}

$destination = [IO.File]::Open($outputFull, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try {
    foreach ($segment in $segments) {
        $source = [IO.File]::OpenRead($segment.Path)
        try {
            $source.CopyTo($destination, 8388608)
        } finally {
            $source.Dispose()
        }
    }
} finally {
    $destination.Dispose()
}

$assembledLength = (Get-Item -LiteralPath $outputFull).Length
if ($assembledLength -ne $TotalBytes) {
    throw "Assembled file has $assembledLength bytes; expected $TotalBytes"
}

$actualMd5 = (Get-FileHash -LiteralPath $outputFull -Algorithm MD5).Hash.ToLowerInvariant()
if ($actualMd5 -ne $ExpectedMd5.ToLowerInvariant()) {
    throw "MD5 mismatch: expected $ExpectedMd5, got $actualMd5"
}
$actualSha256 = (Get-FileHash -LiteralPath $outputFull -Algorithm SHA256).Hash.ToLowerInvariant()

Remove-Item -LiteralPath $partRootFull -Recurse -Force

[pscustomobject]@{
    Status = 'pass'
    OutputPath = $outputFull
    Bytes = $assembledLength
    Md5 = $actualMd5
    Sha256 = $actualSha256
    SegmentCount = $segments.Count
    Parallelism = $Parallelism
} | ConvertTo-Json -Depth 3
