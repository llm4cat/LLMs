# Leverage Larage Language Models (LLMS) in cataloging subject analysis
# Project Title
LCSH Label Generation with LLMs and Small Model Guidance
## Table of Contents
1. [Introduction](#introduction)
2. [Method](#method)
3. [Installation](#installation)
4. [Tech Stack](#tech-stack)
5. [License](#license)
6. [Contact](#contact)

## Introduction

We proposed a hybrid framework that combines small predictive models, regression models, and embedding models with large language models to further improve the performance of LLMs in our  Library of Congress Subject Headings (LCSH) prediction task. And we added post-processing to handle hallucinations.

We address key limitations in zero-shot learning such as hallucinated labels, domain mismatch by combining small model guidance with LLM reasoning and vocabulary post-editing.


## Method
Small Model for Label Count Prediction
- lightweight model (Randon Forest, Linear regression, XGboost) predicts the number of LCSH labels (N) based on metadata ( title and abstract).
- We explore multiple values of N (N, 2N, 3N) to analyze the trade-off between precision and recall and to accommodate different user needs.

LLM-Based Label Generation
- We test various LLMs' performance in our task- Llama-3.1-8b, GPT-3.5/4, DeepSeek-R1/V3
- We ask LLMs to generate as many LCSH labels as possible to ensure diversity.
- We use the predicted number of labels to guide the LLM to generate an appropriate number of subject headings.
  
Chain-of-Thought (CoT) Reasoning
- We apply CoT via prompt engineering; we designed a multi-round inference prompt and let LLMs generate the final answer step by step.
- LLMs will consider previously generated labels when predicting later answers in CoT reasoning.
- In this way, we can efficiently improve the outputs' diversity and richness.
  
Fine-tuning
- We fine-tune Llama-3.1-8B-Instruct with our training set (76160 samples) and evalute fine-tuned models with our testing set (2100 samples)
- We apply Supervised Fine-Tuning (SFT) combined with Low-Rank Adaptation (LoRA)to fine-tune our LLM.
- SFT ensures that the model learns from high-quality, domain-specific examples.
- LoRA significantly reduces training cost and GPU memory usage by freezing most model parameters and only updating a small set of low-rank matrices.
  
Post-processing
- We add post-processing to handle hallucinations in LLM.
- Non-standard terms are replaced by using embedding similarity.
- We replaced them with official controlled vocabulary.
  
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/llm4cat/LLMs.git
   ```

2. Navigate to the project directory:
   ```bash
   cd LLMs
    ```
3. Create a new virtual environment(Where we run is in Linux with sever and we manage our environment by conda )
   ```bash
      conda create --name bert_env python=3.8
      conda activate bert_env
    ```

4. Install dependencies:
   ```bash
   python --version # make sure you already have install python
   pip install pandas  
   pip install scikit-learn
   pip install torch
   pip install transformers 
   pip install sentence-transformers
   pip install openai
   pip install peft
   pip install faiss
   ```
## Tech Stack
- LLMs: Llama-3.1-8b, GPT-3.5/4 (Openai API), DeepSeek-R1/V3(DeepSeek API)
- Fine-tuning: Framework: peft, transformers
- LoRA Adapter: peft (by HuggingFace)
- Supervised Fine-Tuning (SFT): Trainer API (HuggingFace)
- Small models: Randon Forest, Linear regression, XGboost
- Ebedding models: all-MiniLM-L6-v2 (BERT-base), SciBERT
- Embedding Matching: faiss(NSS), sentence-transformers
- Evaluation: scikit-learn.metrics

  
   
   



## License
This project is licensed under the BSD 3 License. See the `LICENSE` file for details.

## Contact
For questions or issues, please contact:
- **Jinyu Liu** - JinyuLiu@my.unt.edu
- Project Link: [GitHub Repository]https://github.com/llm4cat/LLMs

---

Thank you for using this project! We appreciate your contributions and feedback.
