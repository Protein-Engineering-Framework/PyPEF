Write-Host "Downloading and extracting PyPEF (test) files..."
(New-Object Net.WebClient).DownloadFile("https://github.com/Protein-Engineering-Framework/PyPEF/archive/refs/heads/master.zip", "master.zip")
Expand-Archive .\master.zip .