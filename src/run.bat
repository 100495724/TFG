#!/bin/bash
SEEDS="42,123,456"

CONDITIONS=(
  "homo_llama__level_0_neutral"
  "homo_llama__level_1_professional"
  "homo_llama__level_2_identity"
  "homo_qwen__level_0_neutral"
  "homo_qwen__level_1_professional"
  "homo_qwen__level_2_identity"
  "homo_mistral__level_0_neutral"
  "homo_mistral__level_1_professional"
  "homo_mistral__level_2_identity"
  "homo_gpt__level_0_neutral"
  "homo_gpt__level_1_professional"
  "homo_gpt__level_2_identity"
  "homo_claude__level_0_neutral"
  "homo_claude__level_1_professional"
  "homo_claude__level_2_identity"
  "hetero_local__level_0_neutral"
  "baseline"
  "hetero_local__level_2_identity"
  "hetero_api_claude__level_0_neutral"
  "hetero_api_claude__level_1_professional"
  "hetero_api_claude__level_2_identity"
  "hetero_api_gpt__level_0_neutral"
  "hetero_api_gpt__level_1_professional"
  "hetero_api_gpt__level_2_identity"
  "hetero_local__level_1_professional__cooperative"
)

echo "=== Running ${#CONDITIONS[@]} conditions ==="

for i in "${!CONDITIONS[@]}"; do
  COND="${CONDITIONS[$i]}"
  NUM=$((i + 1))
  echo ""
  echo ">>> [$NUM/${#CONDITIONS[@]}] $COND"
  python experiment_runner.py --act 1 --condition "$COND" --seeds "$SEEDS"
  if [ $? -ne 0 ]; then
    echo "!!! FAILED: $COND"
  fi
done

echo ""
echo "=== Done ==="