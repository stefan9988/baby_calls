#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="baby-calls"
DATASET_PATH="$(pwd)/UNS dataset"

# Display menu and get user choice
echo -e "${BLUE}=== Baby Calls Docker Runner ===${NC}\n"
echo "Select which pipeline to run:"
echo "  1) Standard pipeline (keywords → summary → transcription)"
echo "  2) LangChain pipeline (keywords → summary → transcription) - Recommended"
echo "  3) SDialog transcription"
echo ""
read -p "Enter your choice (1, 2, or 3): " choice

echo -e "\n${YELLOW}Starting Docker build and run process...${NC}\n"

# Step 1: Build Docker image
echo -e "${YELLOW}Step 1: Building Docker image '${IMAGE_NAME}'...${NC}"
if sudo docker build -t ${IMAGE_NAME} .; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}\n"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

# Run selected pipeline
case $choice in
    1)
        echo -e "${BLUE}Running Standard Pipeline${NC}\n"

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
        ;;

    2)
        echo -e "${BLUE}Running LangChain Pipeline${NC}\n"

        # Step 2: Run generate_keywords_langchain.py
        echo -e "${YELLOW}Step 2: Running generate_keywords_langchain.py...${NC}"
        if sudo docker run --rm --network host \
            --env-file .env \
            -v "${DATASET_PATH}":/app/src/UNS\ dataset \
            ${IMAGE_NAME} generate_keywords_langchain.py; then
            echo -e "${GREEN}✓ generate_keywords_langchain.py completed${NC}\n"
        else
            echo -e "${RED}✗ generate_keywords_langchain.py failed${NC}"
            exit 1
        fi

        # Step 3: Run generate_summary_langchain.py
        echo -e "${YELLOW}Step 3: Running generate_summary_langchain.py...${NC}"
        if sudo docker run --rm --network host \
            --env-file .env \
            -v "${DATASET_PATH}":/app/src/UNS\ dataset \
            ${IMAGE_NAME} generate_summary_langchain.py; then
            echo -e "${GREEN}✓ generate_summary_langchain.py completed${NC}\n"
        else
            echo -e "${RED}✗ generate_summary_langchain.py failed${NC}"
            exit 1
        fi

        # Step 4: Run generate_transcription_langchain.py
        echo -e "${YELLOW}Step 4: Running generate_transcription_langchain.py...${NC}"
        if sudo docker run --rm --network host \
            --env-file .env \
            -v "${DATASET_PATH}":/app/src/UNS\ dataset \
            ${IMAGE_NAME} generate_transcription_langchain.py; then
            echo -e "${GREEN}✓ generate_transcription_langchain.py completed${NC}\n"
        else
            echo -e "${RED}✗ generate_transcription_langchain.py failed${NC}"
            exit 1
        fi
        ;;

    3)
        echo -e "${BLUE}Running SDialog Pipeline${NC}\n"

        # Run sdialog_generate_transcription.py
        echo -e "${YELLOW}Step 2: Running sdialog_generate_transcription.py...${NC}"
        if sudo docker run --rm --network host \
            --env-file .env \
            -v "${DATASET_PATH}":/app/src/UNS\ dataset \
            ${IMAGE_NAME} sdialog_generate_transcription.py; then
            echo -e "${GREEN}✓ sdialog_generate_transcription.py completed${NC}\n"
        else
            echo -e "${RED}✗ sdialog_generate_transcription.py failed${NC}"
            exit 1
        fi
        ;;

    *)
        echo -e "${RED}✗ Invalid choice. Please select 1, 2, or 3.${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}All tasks completed successfully!${NC}"