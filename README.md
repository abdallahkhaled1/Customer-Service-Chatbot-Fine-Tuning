# Customer-Service-Chatbot-Fine-Tuning


Fine-tuning a small language model (**Qwen2.5-1.5B-Instruct**) into a polite and concise customer service assistant, using **Unsloth** and **QLoRA**. The notebook runs on a free Google Colab T4 GPU.

## Overview

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Method:** QLoRA (4-bit quantization + LoRA adapters)
- **Framework:** [Unsloth](https://github.com/unslothai/unsloth), TRL (`SFTTrainer`), Hugging Face Transformers
- **Dataset:** [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

## What the Notebook Does

1. Installs dependencies and logs in to Hugging Face.
2. Loads the base model in 4-bit and applies the Qwen 2.5 chat template.
3. Loads the dataset and formats each example as a chat (system / user / assistant).
4. Tests the base model on a sample customer question.
5. Attaches LoRA adapters and fine-tunes the model.
6. Tests the fine-tuned model on the same question to compare the outputs.

## Training Setup

| Setting | Value |
|---|---|
| Max sequence length | 2048 |
| LoRA rank / alpha | 16 / 16 |
| Target modules | q, k, v, o, gate, up, down projections |
| Batch size | 2 (gradient accumulation 4) |
| Learning rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Training steps | 60 |

> **Note:** 60 steps covers only a small fraction of the dataset, so this setup is a quick demo of the pipeline. For better results, train for at least one full epoch.

## Requirements

- Google Colab with a GPU runtime (T4 or better)
- A Hugging Face account and access token

## Usage

1. Open the notebook in Google Colab.
2. Go to **Secrets** (the key icon) and add your Hugging Face token as `HF_TOKEN`.
3. Select a GPU runtime: **Runtime → Change runtime type → T4 GPU**.
4. Run all cells.

## Example

**Prompt:** `question about cancelling order`

| Before fine-tuning | After fine-tuning |
|---|---|
| Hello! I'm here to help with any questions you might have about canceling an order. Could you please provide me with more details? What specific information do you need from the order or what is your question regarding cancellation?|,
|I'm here to assist you with your inquiry regarding the cancellation of an order. To cancel an order, please visit our website or contact us at {{Customer Support Phone Number}}. Our dedicated team will be more than happy to guide you through the process and ensure that your order is cancelled promptly. Rest assured, we value your satisfaction and appreciate any feedback you may have.|

## Limitations

- The dataset is synthetic and template-based, so responses may contain generic placeholders and may not match a real company's policies.
- The model was evaluated on a single sample prompt, without quantitative metrics.
- Short training run; results are for demonstration purposes.

## Possible Improvements

- Train on assistant responses only.
- Add a validation split and track evaluation loss.
- Evaluate on a larger and more diverse set of prompts.
- Fine-tune on real company data.
- Save the adapters and push them to the Hugging Face Hub.

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth)
- [Bitext](https://huggingface.co/bitext) for the dataset
- [Qwen Team](https://huggingface.co/Qwen) for the base model

## License

Add your license here. Please also check the licenses of the base model and the dataset.
