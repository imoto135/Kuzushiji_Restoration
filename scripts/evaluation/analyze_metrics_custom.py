import re
import os

log_files = [
    "/home/imoto/Kuzushiji_Restoration/nafnet/experiments/NAFNet_Kuzushiji_Pattern1_TVLoss/train_NAFNet_Kuzushiji_Pattern1_TVLoss_20260123_153430.log",
    "/home/imoto/Kuzushiji_Restoration/nafnet/experiments/NAFNet_Kuzushiji_Pattern2_MaskMorph/train_NAFNet_Kuzushiji_Pattern2_MaskMorph_20260123_153433.log",
    "/home/imoto/Kuzushiji_Restoration/nafnet/experiments/NAFNet_Kuzushiji_Pattern3_EdgeOnly/train_NAFNet_Kuzushiji_Pattern3_EdgeOnly_20260123_170010.log"
]

def analyze_log(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"\nAnalyzing: {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    best_psnr = -1
    best_psnr_iter = -1
    best_psnr_metrics = {}

    best_masked_psnr = -1
    best_masked_psnr_iter = -1
    best_masked_psnr_metrics = {}

    current_iter = 0

    # Patterns
    # Log line for iter: ... [epoch:120, iter: 205,000, lr:...] ...
    iter_pattern = re.compile(r"iter:\s*([\d,]+),")
    # Validation line: ... Validation Kuzushiji_Val, 		 # psnr: 36.6687	 # ssim: 0.9806	 # masked_psnr: 33.6789	 # masked_ssim: 0.9806
    val_pattern = re.compile(r"Validation\s+.*#\s*psnr:\s*([\d\.]+)\s*#\s*ssim:\s*([\d\.]+)\s*#\s*masked_psnr:\s*([\d\.]+)\s*#\s*masked_ssim:\s*([\d\.]+)")
    # Fallback for logs that might not have masked metrics if user didn't implement them in all logs?
    # Based on user request, they seem to be present.
    val_pattern_basic = re.compile(r"Validation\s+.*#\s*psnr:\s*([\d\.]+)\s*#\s*ssim:\s*([\d\.]+)")

    for line in lines:
        # Check for iteration update
        iter_match = iter_pattern.search(line)
        if iter_match:
            current_iter = int(iter_match.group(1).replace(',', ''))
        
        # Check for validation
        if "Validation" in line and "psnr" in line:
            val_match = val_pattern.search(line)
            if val_match:
                psnr = float(val_match.group(1))
                ssim = float(val_match.group(2))
                masked_psnr = float(val_match.group(3))
                masked_ssim = float(val_match.group(4))

                if psnr > best_psnr:
                    best_psnr = psnr
                    best_psnr_iter = current_iter
                    best_psnr_metrics = {'psnr': psnr, 'ssim': ssim, 'masked_psnr': masked_psnr, 'masked_ssim': masked_ssim}
                
                if masked_psnr > best_masked_psnr:
                    best_masked_psnr = masked_psnr
                    best_masked_psnr_iter = current_iter
                    best_masked_psnr_metrics = {'psnr': psnr, 'ssim': ssim, 'masked_psnr': masked_psnr, 'masked_ssim': masked_ssim}
            else:
                # Try basic pattern if masked not found (just in case)
                val_match_basic = val_pattern_basic.search(line)
                if val_match_basic:
                    psnr = float(val_match_basic.group(1))
                    ssim = float(val_match_basic.group(2))
                    
                    if psnr > best_psnr:
                        best_psnr = psnr
                        best_psnr_iter = current_iter
                        best_psnr_metrics = {'psnr': psnr, 'ssim': ssim, 'masked_psnr': 'N/A', 'masked_ssim': 'N/A'}

    print(f"  Best PSNR: {best_psnr} at iter {best_psnr_iter}")
    print(f"    Metrics: {best_psnr_metrics}")
    print(f"  Best Masked PSNR: {best_masked_psnr} at iter {best_masked_psnr_iter}")
    print(f"    Metrics: {best_masked_psnr_metrics}")

for log_file in log_files:
    analyze_log(log_file)
