#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="baby-calls"
DATASET_PATH="$(pwd)/UNS dataset"

echo -e "${YELLOW}Starting Docker build and run process...${NC}\n"

# Step 1: Build Docker image
echo -e "${YELLOW}Step 1: Building Docker image '${IMAGE_NAME}'...${NC}"
if sudo docker build -t ${IMAGE_NAME} .; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}\n"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

# Step 2: Run generate_keywords.py
echo -e "${YELLOW}Step 2: Running generate_keywords.py...${NC}"
if sudo docker run --rm --network host \
    --env-file .env \
    -v "${DATASET_PATH}":/app/src/UNS\ dataset \
    ${IMAGE_NAME} generate_keywords.py; then
    echo -e "${GREEN}✓ generate_keywords.py completed${NC}\n"
else
    echo -e "${RED}✗ generate_keywords.py failed${NC}"
    exit 1
fi

# Step 3: Run generate_summary.py
echo -e "${YELLOW}Step 3: Running generate_summary.py...${NC}"
if sudo docker run --rm --network host \
    --env-file .env \
    -v "${DATASET_PATH}":/app/src/UNS\ dataset \
    ${IMAGE_NAME} generate_summary.py; then
    echo -e "${GREEN}✓ generate_summary.py completed${NC}\n"
else
    echo -e "${RED}✗ generate_summary.py failed${NC}"
    exit 1
fi

# Step 4: Run generate_transcription.py
echo -e "${YELLOW}Step 4: Running generate_transcription.py...${NC}"
if sudo docker run --rm --network host \
    --env-file .env \
    -v "${DATASET_PATH}":/app/src/UNS\ dataset \
    ${IMAGE_NAME} generate_transcription.py; then
    echo -e "${GREEN}✓ generate_transcription.py completed${NC}\n"
else
    echo -e "${RED}✗ generate_transcription.py failed${NC}"
    exit 1
fi

echo -e "${GREEN}All tasks completed successfully!${NC}"