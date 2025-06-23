# Paediatric Transport Deterioration Prediction

This repository contains the codebase associated with the research project:

**"Real-time prediction of cardiorespiratory deterioration during paediatric critical care transport using interpretable machine learning"**

📄 **Authors**: Milan Kapur\*, Kezhi Li\*, Alexander Brown, Zhiqiang Huo, John Booth, Philip Knight, Gwyneth Davies†, Padmanabhan Ramnarayan†  
\*Joint first authors · †Joint senior authors  
📧 Corresponding Author: Milan Kapur (m.kapur@ucl.ac.uk)  
📍 Institutions: UCL GOSH ICH, UCL IHI, GOSH, CATS, Imperial College London

---

## 🧠 Project Overview

This work presents lightweight, interpretable machine learning models capable of predicting **respiratory and cardiovascular deterioration up to 15 minutes in advance** during interhospital transport of critically ill children.

- Trained on high-frequency physiological data from **1,519** transports
- Best models use **transformer architectures** with vector-embedded diagnoses
- Designed for **real-time deployment** on edge devices
- Integrated Gradients used for interpretability

> For full details, see the manuscript (forthcoming).

---

## 📊 Key Results

| Task                    | Model                        | AUROC | AUPRC |
|-------------------------|------------------------------|--------|--------|
| Respiratory Deterioration | Transformer + Embedding       | 0.851  | 0.200  |
| Cardiovascular Deterioration | Transformer + Embedding       | 0.792  | 0.183  |

---

## 🔐 Data Access

Due to patient sensitivity and ethical governance, the underlying physiological and EHR data are **not publicly available**.

To request access for replication or extension under appropriate ethical approvals, please contact:

- **Children’s Acute Transport Service (CATS)** – [CATS contact page](https://www.cats.nhs.uk/)
- **Great Ormond Street Hospital R&D** – [research.gosh.nhs.uk](https://www.gosh.nhs.uk/Research)

---

## 📜 License

This code is licensed under the [MIT License](LICENSE).

> Note: This repository **does not include the original patient data** used for model training and evaluation. Due to ethical and governance restrictions, access to the data must be requested separately from the Children's Acute Transport Service (CATS) and Great Ormond Street Hospital.


---

## 🧩 Citation

If using this codebase in your research, please cite:



---

## 🙏 Acknowledgements

This work was supported by:
- NIHR Academic Clinical Fellowship (MK)
- UKRI CDT in AI-enabled healthcare (KL)
- NIHR GOSH BRC
- Kinseed, DRIVE @ GOSH, BioClinicalBERT

All experiments conducted on the GOSH Digital Research Environment.

---
