param(
    [string[]]$Modes = @("mixed_core3_router_bsize_sweep"),
    [int[]]$Seeds = @(42),
    [string]$PythonIoEncoding = "utf-8"
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

foreach ($mode in $Modes) {
    foreach ($seed in $Seeds) {
        $log = "modal_recurrent_smol_${mode}_seed${seed}_${timestamp}.log"
        $cmd = "chcp 65001 > NUL && set PYTHONIOENCODING=$PythonIoEncoding && modal run run_recurrent_smol.py --mode $mode --seed $seed > $log 2>&1"
        $process = Start-Process -FilePath "cmd.exe" `
            -ArgumentList @("/c", $cmd) `
            -WorkingDirectory (Get-Location) `
            -WindowStyle Hidden `
            -PassThru
        [pscustomobject]@{
            Mode = $mode
            Seed = $seed
            Pid = $process.Id
            Log = $log
        }
    }
}
