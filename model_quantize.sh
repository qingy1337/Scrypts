#!/bin/sh

usage() {
  echo "Usage: $0 <input-hf-model> <output-hf-model> [--keep-files]"
  echo "Example: $0 qingy2024/GRMR-V3-G4B qingy2024/GRMR-V3-G4B-GGUF"
  echo "Options:"
  echo "  --keep-files    Keep downloaded and generated files after upload (default: delete)"
  exit 1
}

if [ $# -lt 2 ]; then
  usage
fi

INPUT_MODEL=$1
OUTPUT_MODEL=$2
KEEP_FILES=false

if [ $# -eq 3 ] && [ "$3" = "--keep-files" ]; then
  KEEP_FILES=true
fi

MODEL_NAME=$(basename $INPUT_MODEL)
WORK_DIR="$(pwd)/${MODEL_NAME}"
FP16_GGUF="${WORK_DIR}/${MODEL_NAME}-FP16.gguf"

mkdir -p "$WORK_DIR"
echo "Working directory: $WORK_DIR"

chmod +x ./llama-quantize

echo "Downloading model ${INPUT_MODEL}..."
huggingface-cli download "${INPUT_MODEL}" --local-dir "${WORK_DIR}"
if [ $? -ne 0 ]; then
  echo "Error downloading model. Exiting."
  exit 1
fi

echo "Converting model to GGUF format..."
python3 convert_hf_to_gguf.py "${WORK_DIR}" --outfile "${FP16_GGUF}"
if [ $? -ne 0 ]; then
  echo "Error converting model to GGUF. Exiting."
  exit 1
fi

QUANT_TYPES="Q2_K Q3_K_L Q3_K_M Q3_K_S Q4_K_M Q4_K_S Q5_K_M Q5_K_S Q6_K Q8_0"
QUANTIZED_FILES=""

for QUANT in $QUANT_TYPES; do
  QUANT_OUTPUT="${WORK_DIR}/${MODEL_NAME}-${QUANT}.gguf"
  echo "Quantizing model to ${QUANT}..."
  ./llama-quantize "${FP16_GGUF}" "${QUANT_OUTPUT}" "${QUANT}"
  if [ $? -ne 0 ]; then
    echo "Warning: Failed to quantize model to ${QUANT}. Continuing."
    continue
  fi
  QUANTIZED_FILES="$QUANTIZED_FILES ${QUANT_OUTPUT}"
done

# Final upload directory: <model>_upload
UPLOAD_DIR="${MODEL_NAME}_upload"
mkdir -p "$UPLOAD_DIR"
cp "${FP16_GGUF}" "${UPLOAD_DIR}/$(basename ${FP16_GGUF})"
for FILE in $QUANTIZED_FILES; do
  cp "${FILE}" "${UPLOAD_DIR}/$(basename ${FILE})"
done

cat > "${UPLOAD_DIR}/README.md" << EOL
# Quantized GGUF models for ${MODEL_NAME}

This repository contains GGUF quantized versions of [${INPUT_MODEL}](https://huggingface.co/${INPUT_MODEL}).

## Available quantizations:
- FP16 (full precision)
EOL

for QUANT in $QUANT_TYPES; do
  echo "- ${QUANT}" >> "${UPLOAD_DIR}/README.md"
done

cat >> "${UPLOAD_DIR}/README.md" << EOL

## Original model
This is a quantized version of [${INPUT_MODEL}](https://huggingface.co/${INPUT_MODEL}).

## Generated on
$(date)
EOL

echo "Uploading all files to Hugging Face..."
huggingface-cli upload "${OUTPUT_MODEL}" "${UPLOAD_DIR}/" .

if [ $? -ne 0 ]; then
  echo "Error uploading files to Hugging Face. Files remain in ${UPLOAD_DIR}"
else
  echo "Successfully uploaded quantized models to ${OUTPUT_MODEL}"
  if [ "$KEEP_FILES" = false ]; then
    echo "Cleaning up..."
    rm -rf "${WORK_DIR}"
    rm -rf "${UPLOAD_DIR}"
    echo "Cleanup complete."
  else
    echo "Keeping all files as requested."
    echo "Files are in ${WORK_DIR} and ${UPLOAD_DIR}"
  fi
fi

echo "Done!"
