import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from datasets import Dataset
from huggingface_hub import snapshot_download, HfApi

def quantize_model_with_imatrix(
    input_model: str,
    output_model: str,
    dataset: Dataset,
    keep_files: bool = False,
    text_column: str = "text",
    max_samples: int = 1000
) -> bool:
    """
    Quantize a Hugging Face model using imatrix calibration.
    
    Args:
        input_model: Hugging Face model ID (e.g., "qingy2024/GRMR-V3-G4B")
        output_model: Target Hugging Face model ID for upload
        dataset: Hugging Face dataset with text column for calibration
        keep_files: Whether to keep intermediate files after upload
        text_column: Name of the text column in the dataset
        max_samples: Maximum number of samples to use for imatrix calibration
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    model_name = input_model.split("/")[-1]
    work_dir = Path(f"./{model_name}_work")
    upload_dir = Path(f"./{model_name}_upload")
    
    try:
        # Create working directories
        work_dir.mkdir(exist_ok=True)
        upload_dir.mkdir(exist_ok=True)
        
        print(f"Working directory: {work_dir}")
        print(f"Upload directory: {upload_dir}")
        
        # Download the model
        print(f"Downloading model {input_model}...")
        snapshot_download(
            repo_id=input_model,
            local_dir=str(work_dir / "hf_model")
        )
        
        # Convert to GGUF formats
        fp16_gguf = work_dir / f"{model_name}-FP16.gguf"
        bf16_gguf = work_dir / f"{model_name}-BF16.gguf"
        
        print("Converting model to FP16 GGUF format...")
        result = subprocess.run([
            "python3", "convert_hf_to_gguf.py",
            str(work_dir / "hf_model"),
            "--outfile", str(fp16_gguf),
            "--outtype", "f16"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error converting to FP16: {result.stderr}")
            return False
            
        print("Converting model to BF16 GGUF format...")
        result = subprocess.run([
            "python3", "convert_hf_to_gguf.py",
            str(work_dir / "hf_model"),
            "--outfile", str(bf16_gguf),
            "--outtype", "bf16"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error converting to BF16: {result.stderr}")
            return False
        
        # Prepare calibration data
        calibration_file = work_dir / "calibration.txt"
        print(f"Preparing calibration data from dataset (max {max_samples} samples)...")
        
        if text_column not in dataset.column_names:
            print(f"Error: Column '{text_column}' not found in dataset. Available columns: {dataset.column_names}")
            return False
        
        # Take a subset of the dataset and write to file
        sample_count = min(len(dataset), max_samples)
        with open(calibration_file, 'w', encoding='utf-8') as f:
            for i in range(sample_count):
                text = dataset[i][text_column]
                if text and len(text.strip()) > 0:
                    f.write(text.strip() + '\n\n')
        
        print(f"Created calibration file with {sample_count} samples")

        print("------ OS Listdir ------")
        print(os.listdir('.'))
        print('---\n' + os.getcwd())
        
        # Create imatrix
        imatrix_file = work_dir / "imatrix.dat"
        print("Creating imatrix calibration data...")
        
        result = subprocess.run([
            "./llama-imatrix",
            "-m", str(fp16_gguf),
            "-f", str(calibration_file),
            "-o", str(imatrix_file),
            "-ngl", "99"  # Use GPU if available
        ], capture_output=True, text=True, cwd=work_dir)
        
        if result.returncode != 0:
            print(f"Error creating imatrix: {result.stderr}")
            return False
        
        # Quantization types for imatrix
        imatrix_quant_types = [
            "IQ3_M", "IQ3_XXS", "Q4_K_M", "Q4_K_S", 
            "IQ4_XS", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0"
        ]
        
        # Regular quantization types (without imatrix)
        regular_quant_types = ["Q2_K", "Q3_K_L", "Q3_K_M", "Q3_K_S"]
        
        quantized_files = []
        
        # Quantize with imatrix
        for quant_type in imatrix_quant_types:
            output_file = work_dir / f"{model_name}-{quant_type}-imat.gguf"
            print(f"Quantizing model to {quant_type} with imatrix...")
            
            result = subprocess.run([
                "./llama-quantize",
                "--imatrix", str(imatrix_file),
                str(bf16_gguf),
                str(output_file),
                quant_type
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                quantized_files.append(output_file)
                print(f"Successfully quantized to {quant_type}")
            else:
                print(f"Warning: Failed to quantize to {quant_type}: {result.stderr}")
        
        # Quantize without imatrix (for types that don't benefit from it)
        for quant_type in regular_quant_types:
            output_file = work_dir / f"{model_name}-{quant_type}.gguf"
            print(f"Quantizing model to {quant_type}...")
            
            result = subprocess.run([
                "./llama-quantize",
                str(fp16_gguf),
                str(output_file),
                quant_type
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                quantized_files.append(output_file)
                print(f"Successfully quantized to {quant_type}")
            else:
                print(f"Warning: Failed to quantize to {quant_type}: {result.stderr}")
        
        # Copy files to upload directory
        print("Preparing files for upload...")
        shutil.copy2(fp16_gguf, upload_dir / fp16_gguf.name)
        shutil.copy2(bf16_gguf, upload_dir / bf16_gguf.name)
        
        for file_path in quantized_files:
            shutil.copy2(file_path, upload_dir / file_path.name)
        
        # Create README
        readme_content = f"""---
base_model:
- {input_model}
---
# Quantized GGUF models for {model_name}

This repository contains GGUF quantized versions of [{input_model}](https://huggingface.co/{input_model}).

## Available quantizations:

### Full Precision
- FP16 (full precision)
- BF16 (bfloat16 precision)

### imatrix Quantizations
These quantizations use importance matrix (imatrix) calibration for better quality:
"""
        
        for quant_type in imatrix_quant_types:
            readme_content += f"- {quant_type}-imat\n"
        
        readme_content += f"""
### Standard Quantizations
"""
        
        for quant_type in regular_quant_types:
            readme_content += f"- {quant_type}\n"
        
        readme_content += f"""
## About imatrix quantization

The imatrix quantizations in this repository use calibration data to preserve the most important weights during quantization, resulting in better model quality compared to standard quantization methods.

## Original model
This is a quantized version of [{input_model}](https://huggingface.co/{input_model}).

## Generated on
{subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}
"""
        
        with open(upload_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Upload to Hugging Face
        print(f"Uploading all files to Hugging Face repository {output_model}...")
        api = HfApi()
        
        try:
            api.upload_folder(
                folder_path=str(upload_dir),
                repo_id=output_model,
                repo_type="model"
            )
            print(f"Successfully uploaded quantized models to {output_model}")
            
        except Exception as e:
            print(f"Error uploading to Hugging Face: {e}")
            print(f"Files remain in {upload_dir}")
            return False
        
        # Cleanup
        if not keep_files:
            print("Cleaning up...")
            shutil.rmtree(work_dir)
            shutil.rmtree(upload_dir)
            print("Cleanup complete.")
        else:
            print("Keeping all files as requested.")
            print(f"Files are in {work_dir} and {upload_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error during quantization process: {e}")
        return False


# Command line interface
if __name__ == "__main__":
    import argparse
    from datasets import load_dataset
    
    parser = argparse.ArgumentParser(description="Quantize a Hugging Face model with imatrix calibration")
    parser.add_argument("input_model", help="Input Hugging Face model ID")
    parser.add_argument("output_model", help="Output Hugging Face model ID")
    parser.add_argument("dataset", help="Hugging Face dataset ID for calibration")
    parser.add_argument("--keep-files", action="store_true", help="Keep intermediate files after upload")
    parser.add_argument("--text-column", default="text", help="Name of text column in dataset (default: text)")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples for calibration (default: 1000)")
    parser.add_argument("--dataset-split", default="train", help="Dataset split to use (default: train)")
    
    args = parser.parse_args()
    
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    
    success = quantize_model_with_imatrix(
        input_model=args.input_model,
        output_model=args.output_model,
        dataset=dataset,
        keep_files=args.keep_files,
        text_column=args.text_column,
        max_samples=args.max_samples
    )
    
    if success:
        print("Quantization completed successfully!")
    else:
        print("Quantization failed!")
        exit(1)
