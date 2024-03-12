
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
   

</head>
<body>
  



<h1>💪 Empowering Mistral7B Instruct with LoRa: 🤖 Specializing in DoD's Fiscal Insights 💰 </h1>

<h2>Introduction</h2><p>
    Fine-tuning Large Language Models (LLMs) 🧠 unlocks their true potential ✨, enabling them to excel 🏆 at specialized tasks and address domain-specific knowledge gaps 🕳️. In our project 🚀, we employed the LoRa method 🛠️ to fine-tune our LLM, Mistral7B Instruct 🌬️. This process empowered 💪 Mistral7B Instruct with the ability to answer intricate questions ❓ about the Department of Defense's (DoD) budget 💰 for fiscal year 2024 📅, a domain where it previously lacked expertise 🚫🤷.
</p>

<h2>🗂 Repository Structure</h2>
<p>
  📝 This repository consists of three key files: two Python scripts and one Jupyter notebook.
</p>

<ul>
  <li>🗃 Database Creation Script:
    <a href="https://github.com/GhaithMag/Fine_tune_LLM_LoRa_Mistral7B_Instruct/blob/main/prepare_dataset_LLM.py">prepare_dataset_LLM.py</a> 
  </li>
  <li>🚀 Model Training Script:
    <a href="https://github.com/GhaithMag/Fine_tune_LLM_LoRa_Mistral7B_Instruct/blob/main/train_llm_lora.py">train_llm_lora.py</a> 
  </li>
  <li>📘 Model Training Guide:
    <a href="https://github.com/GhaithMag/Fine_tune_LLM_LoRa_Mistral7B_Instruct/blob/main/mistral-finetune-own-data.ipynb">mistral-finetune-own-data.ipynb</a> 
  </li>
</ul>



<h2>📋 Prerequisites</h2>

<p>🐧 Ensure you are using a Linux environment because the <code>bitsandbytes</code> library, which is essential for our task, only works on Linux as of now.</p>

<p>📥 The dataset required for this project can be downloaded from <a href="https://www.congress.gov/bill/118th-congress/house-bill/4365/text?format=txt">here</a>.</p>

<p>🔑 Access to a LLM API is necessary for creating your own dataset using the <code>prepare_dataset_LLM.py</code> script. For this project, we used LM Studio. Feel free to use any API you want. 😊</p>

<h2>🚀 Getting Started</h2>

<p>👩‍💻 Once you have cloned the repository, start by running the <code>create-data.py</code> script to generate your dataset. Then, proceed to explore the provided <a href="https://github.com/GhaithMag/Fine_tune_LLM_LoRa_Mistral7B_Instruct/blob/main/mistral-finetune-own-data.ipynb">notebook</a> to understand the training process in detail.</p>

<p>💡 Fine-tuning an LLM has never been easier. This project provides you with the necessary tools and guidance to customize an LLM to your specific needs.</p>

<img src="fine_tune_ill.jpg" alt="Illustration of LLM Fine-Tuning" style="width:100%; max-width:200px; display:block">

