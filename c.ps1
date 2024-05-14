# Define common CUDA installation directories
$commonCudaPaths = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    "C:\Program Files\NVIDIA Corporation\CUDA",
    "C:\Program Files\NVIDIA\CUDNN"
)

# Function to find the latest CUDA version installed
function Get-LatestCudaPath {
    $latestCudaPath = $null
    foreach ($path in $commonCudaPaths) {
        Write-Output "Checking path: $path"
        if (Test-Path -Path $path) {
            $versions = Get-ChildItem -Path $path | Where-Object { $_.PSIsContainer }
            if ($versions) {
                # Sort versions and get the latest one
                $latestVersion = $versions | Sort-Object Name -Descending | Select-Object -First 1
                Write-Output "Found CUDA installation at: $($latestVersion.FullName)"
                $latestCudaPath = $latestVersion.FullName
                break
            }
        }
    }
    return $latestCudaPath
}

# Get the latest CUDA installation path
$cudaPath = Get-LatestCudaPath

if ($cudaPath -eq $null) {
    Write-Error "CUDA installation not found."
    exit
}

Write-Output "Setting CUDA_PATH to $cudaPath"

# Set CUDA_PATH environment variable
try {
    [System.Environment]::SetEnvironmentVariable("CUDA_PATH", $cudaPath, [System.EnvironmentVariableTarget]::Machine)
    Write-Output "Set CUDA_PATH to $cudaPath"
} catch {
    Write-Error "Failed to set CUDA_PATH environment variable."
    exit
}

# Paths to add to the system PATH variable
$cudaBinPath = Join-Path -Path $cudaPath -ChildPath "bin"
$cudaLibnvvpPath = Join-Path -Path $cudaPath -ChildPath "libnvvp"

# Get the current system PATH variable
$path = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine)

# Add CUDA paths if they are not already in the PATH
if ($path -notcontains $cudaBinPath) {
    $path += ";$cudaBinPath"
    Write-Output "Added $cudaBinPath to PATH"
} else {
    Write-Output "$cudaBinPath is already in PATH"
}

if ($path -notcontains $cudaLibnvvpPath) {
    $path += ";$cudaLibnvvpPath"
    Write-Output "Added $cudaLibnvvpPath to PATH"
} else {
    Write-Output "$cudaLibnvvpPath is already in PATH"
}

# Update the system PATH variable
try {
    [System.Environment]::SetEnvironmentVariable("Path", $path, [System.EnvironmentVariableTarget]::Machine)
    Write-Output "Updated system PATH variable"
} catch {
    Write-Error "Failed to update system PATH variable."
    exit
}

# Verify if nvcc is accessible
try {
    $nvccVersion = & nvcc --version
    Write-Output "nvcc version:"
    Write-Output $nvccVersion
} catch {
    Write-Error "nvcc is not accessible. Please ensure the CUDA bin directory is correctly added to PATH."
}
