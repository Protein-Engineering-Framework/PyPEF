Write-Host "Downloading PyPEF test files..."
mkdir "test_dataset_avgfp"
(New-Object Net.WebClient).DownloadFile("https://raw.githubusercontent.com/Protein-Engineering-Framework/PyPEF/master/workflow/test_dataset_avgfp/avGFP.csv", "test_dataset_avgfp\avGFP.csv")
(New-Object Net.WebClient).DownloadFile("https://raw.githubusercontent.com/Protein-Engineering-Framework/PyPEF/master/workflow/test_dataset_avgfp/uref100_avgfp_jhmmer_119.a2m", "test_dataset_avgfp\uref100_avgfp_jhmmer_119.a2m")
(New-Object Net.WebClient).DownloadFile("https://raw.githubusercontent.com/Protein-Engineering-Framework/PyPEF/master/workflow/api_encoding_train_test.py", "api_encoding_train_test.py")
