# Multimodal Medical AI: Vision-Language Alignment, XAI, and RAG

[![Kaggle](https://www.kaggle.com/code/atenazare/chest-ai-assistance)

An end-to-end PyTorch framework for analyzing chest X-rays using multimodal deep learning. This project integrates a dual-encoder vision-language architecture, gradient-based Explainable AI (XAI), and a Retrieval-Augmented Generation (RAG) pipeline to ground radiological images in medical text.

## 📌 Overview
In medical AI, it is crucial not only to process images and text but to link them in an interpretable way. This repository serves as a proof-of-concept for a multimodal clinical assistant. It trains a Vision Transformer (ViT) and a clinical BERT model to share a unified embedding space using Symmetric Contrastive Learning (CLIP loss). 

Once aligned, the model supports **pathology localization** (visual grounding) and **multimodal retrieval** (RAG), enabling generative models to answer complex medical queries based on retrieved clinical evidence.

## 🚀 Key Features

* **Dual-Encoder Architecture (CLIP-style):** 
  * **Vision:** Pre-trained Vision Transformer (`ViT-B/16`).
  * **Text:** `Bio_ClinicalBERT`, optimized for medical domain tokenization.
  * **Projections:** Custom MLP projection heads with GELU and LayerNorm to map distinct modalities into a shared 512-dimensional latent space.
* **Explainable AI (Visual Grounding):** 
  * Implements a custom gradient-based method (Grad-CAM style) that traces the gradient of the image-text similarity score back to the spatial feature maps. This creates heatmaps showing *where* the model is looking given a text prompt (e.g., "pleural effusion").
* **Multimodal RAG Pipeline:** 
  * Integrates **FAISS** vector search to index medical knowledge.
  * Supports multimodal querying: Combines image embeddings and text embeddings to retrieve relevant clinical context.
  * Feeds retrieved evidence into a generative LLM to provide grounded, context-aware radiological insights.

## 📊 Dataset
This project uses the **Indiana University Chest X-ray Dataset** (available via Kaggle/OpenI). It contains frontal and lateral chest radiographs paired with their corresponding diagnostic reports (`findings` and `impressions`).

## ⚙️ Pipeline Architecture

1. **Preprocessing:** DCM/PNG images are resized and normalized; texts are tokenized via HuggingFace.
2. **Training (Contrastive Learning):** The ViT and Bio_ClinicalBERT are trained simultaneously. The InfoNCE/CLIP loss pushes embeddings of matching image-text pairs closer together while pushing non-matching pairs apart.
3. **Retrieval (RAG):** The trained BERT text-projector encodes a database of medical texts into a FAISS index. Novel X-rays are embedded by the ViT and used to search the database for similar historical cases or medical definitions.

## 🛠️ Future Directions & Roadmap
This repository is an ongoing research project. Planned improvements include:
- **Batch Size Simulation:** Implementing Gradient Cache/MoCo to overcome memory limitations and provide better negative sampling for the contrastive loss.
- **Deep Attention Interpretability:** Upgrading the XAI module from projection-layer Grad-CAM to *Attention Rollout* for deeper Transformer interpretability.
- **LLM Upgrades:** Replacing the lightweight generative model (`distilgpt2`) with a specialized medical LLM (e.g., BioMistral or Llama-3 8B) for high-fidelity generation.
- **Database Scaling:** Indexing full MIMIC-CXR report datasets into FAISS for large-scale clinical case retrieval.

## 💻 Usage & Installation

### Run directly on Kaggle (Recommended)
The easiest way to explore this code, view the visual grounding heatmaps, and test the RAG pipeline is to run the fully reproducible Kaggle notebook environment:

👉 **[View and Run the Kaggle Notebook Here](https://www.kaggle.com/code/atenazare/chest-ai-assistance)**
