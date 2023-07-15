# PS1 script taken from https://stackoverflow.com/a/45827384 and adapted
[CmdletBinding()] Param
(
    $minicondaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe",
    $downloadPath = "Miniconda3-latest-Windows-x86_64.exe",
	$fullInstallPath = (Get-Item .).FullName + "\Miniconda3"
)

#New-Item -Path "." -Name "Miniconda3" -ItemType "directory"

Write-Host "Downloading Miniconda..."
(New-Object Net.WebClient).DownloadFile($minicondaUrl, $downloadPath)

try {
Start-Process "${downloadPath}" -argumentlist "/InstallationType=JustMe /RegisterPython=0 /S /D=${fullInstallPath}" -wait
} catch {
	# Catch will pick up any non zero error code returned
    # You can do anything you like in this block to deal with the error, examples below:
    # $_ returns the error details
    # This will just write the error
    Write-Host "${downloadPath} returned the following error $_"
    # If you want to pass the error upwards as a system error and abort your powershell script or function
    Throw "Aborted ${downloadPath} returned $_"
}
