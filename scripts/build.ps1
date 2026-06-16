# Build a single C/C++ source into build/, linking the prebuilt ggml DLLs.
# Usage:  powershell -File scripts\build.ps1 src\gguf_dump.c gguf_dump
# Then run from build\ (DLLs are copied there) so ggml_backend_load_all() finds backends.
param(
    [Parameter(Mandatory=$true)][string]$Src,
    [Parameter(Mandatory=$true)][string]$Out,
    [switch]$WithLlama                                # also link llama.dll (tokenizer/chat-template)
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$gxx  = "C:\msys64\ucrt64\bin\g++.exe"               # native PE; do NOT use MSYS2 shell (fork flakiness)
$inc  = "$root\reference\llama.cpp-src\ggml\include"
$linc = "$root\reference\llama.cpp-src\include"       # llama.h lives here
$dll  = "$root\reference\llama-vulkan"

New-Item -ItemType Directory -Force "$root\build" | Out-Null
# Keep build/ self-contained: ggml runtime DLLs live next to our exes.
Copy-Item "$dll\ggml*.dll" "$root\build\" -Force
Copy-Item "$dll\libomp140.x86_64.dll" "$root\build\" -Force

$srcPath = if ([System.IO.Path]::IsPathRooted($Src)) { $Src } else { Join-Path $root $Src }
$flags = @("-O2","-std=c++17","-I",$inc)
if ($srcPath.EndsWith(".c")) { $flags = @("-O2","-I",$inc) }

$links = @("$dll\ggml.dll","$dll\ggml-base.dll")
if ($WithLlama) {
    $flags += @("-I",$linc)
    Copy-Item "$dll\llama.dll" "$root\build\" -Force   # tokenizer DLL next to the exe
    $links += "$dll\llama.dll"
}

& $gxx @flags $srcPath @links -o "$root\build\$Out.exe"
if ($LASTEXITCODE -ne 0) { throw "compile failed ($LASTEXITCODE)" }
Write-Output "built build\$Out.exe"
