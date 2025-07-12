> **IT** → La relazione completa in italiano, che descrive l’intero progetto di Deep Learning per la rilevazione della malattia di Chagas da segnali ECG, è disponibile nel notebook:  
> **ENG** → The complete report written in Italian, detailing the entire Deep Learning pipeline for Chagas disease detection from ECG signals, is available in the following notebook:  
> [`notebooks/chagas_disease_detection.ipynb`](notebooks/chagas_disease_detection.ipynb)



# Recurrent Convolutional Neural Network for Real World ECG Classification

This project leverages advanced Deep Learning techniques to automatically detect **Chagas disease** from multilead ECG recordings. 

> Academic project developed at the University of Cagliari during the 2024/2025 academic year, as part of the Deep Learning course.



## Project Objectives

The main goal of this project is to develop a **binary classification model** that, starting from **multi-lead ECG signals**, can:

- **Automatically predict** the presence of **Chagas disease**, a parasitic infection that can cause severe cardiac complications.
- Output both a **binary label** (*positive* / *negative*) and an associated **probability score**, useful for **clinical risk assessment**.
- Implement a complete pipeline including **signal preprocessing**, **deep learning model design and training**, **performance evaluation**, and **result visualization**.
- Investigate the influence of **clinical features** (e.g., age) on model performance and assess the robustness of different neural architectures.
- Compare multiple modeling approaches to identify the most effective architecture for detecting Chagas disease from ECG data.

This work aims to combine methodological rigor with potential biomedical applicability, providing a generalizable and interpretable model that can serve as a foundation for future research in medical AI.



## Repository Structure

```
chagas/
┣ data/                                 → placeholder directory (currently empty, tracked via `.gitkeep`)
┣ notebooks/                            → model development notebooks, each containing full experiments: data loading, training, metric evaluation, and visualization
    ┣ assets/                           → images and figures used in the notebook report or documentation
    ┣ Bidirectional_RCNN.ipynb
    ┣ CNN_GRU..ipynb
    ┣ CNN_LSTM.ipynb
    ┣ Simple_CNN.ipynb
    ┣ Simple_GRU.ipynb
    ┣ Simple_LSTM.ipynb
    ┣ Simple_RCNN.ipynb
    ┗ chagas_disease_detection.ipynb    → complete report written in Italian, detailing the entire Deep Learning pipeline
┣ src/                                  → Main source code for the project      
    ┣ models/                           → Model support code (not full model definitions)
        ┣ analysis.py                   → Functions for model performance evaluation and interpretation
        ┣ layers.py                     → Custom Keras layers  
        ┗ utils.py                      → Utility functions
    ┣ preprocessing/                    → Data preprocessing scripts for filtering, loading, and preparing ECG datasets
        ┣ filters.py
        ┣ helper_code.py
        ┣ preprocessing_CODE-15.py
        ┣ preprocessing_SaMi-Trop.py
        ┗ tf_dataset_loader.py
┣ test/
┣ .gitignore                            → Git ignore rules 
┣ LICENSE                               → Project license file
┣ README.md                             → Project documentation (this file)
┣ environment.yml                       → Conda environment file with all required dependencies
┣ main.py                               → Main entry point script (currently a placeholder)
┗ requirements.txt                      → Python packages and specific versions required via pip
```



## Further Insights

For a complete and in-depth overview of the project, please refer to the final report: [`notebooks/chagas_disease_detection.ipynb`](notebooks/chagas_disease_detection.ipynb).


## Contributors

 [@dokunoale](https://github.com/dokunoale)        
 [@giooooia](https://github.com/giooooia)                       
 [@vittoriapala](https://github.com/vittoriapala)  
  