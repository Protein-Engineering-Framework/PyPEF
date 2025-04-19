Write-Host "Downloading PyPEF test files..."
mkdir "AVGFP"
(New-Object Net.WebClient).DownloadFile("https://raw.githubusercontent.com/niklases/PyPEF/main/datasets/AVGFP/avGFP.csv", "AVGFP\avGFP.csv")
(New-Object Net.WebClient).DownloadFile("https://raw.githubusercontent.com/niklases/PyPEF/main/datasets/AVGFP/uref100_avgfp_jhmmer_119.a2m", "AVGFP\uref100_avgfp_jhmmer_119.a2m")
(New-Object Net.WebClient).DownloadFile("https://raw.githubusercontent.com/niklases/PyPEF/main/scripts/Encoding_low_N/api_encoding_train_test.py", "api_encoding_train_test.py")
