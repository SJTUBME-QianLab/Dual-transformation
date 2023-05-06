# Abstract
Pancreatic cancer is a malignant tumor, and its high recurrence rate after surgery is related to the lymph node metastasis status. In clinical practice, a preoperative imaging prediction method is necessary for prognosis assessment and treatment decision; however, there are two major challenges: insufficient data and difficulty in discriminative feature extraction. This paper proposes a dual-transformation with self-supervised learning framework to predict lymph node metastasis in pancreatic cancer using multiphase CT images. Specifically, we designed a novel dynamic surface projection and combined it with the well-established spiral transformation to establish a dual-transformation method that can fully exploit 3D tumor information while reducing computational resources. The dual-transformation can be used to transform 3D data into 2D images, preserving the spatial correlation of the original texture information. Moreover, a dual-transformation-based data augmentation algorithm was developed to produce numerous 2D-transformed images to enhance the diversity and complementarity of the dataset, thereby alleviating the effect of insufficient samples. We designed a self-supervised learning scheme based on intra-space-transformation consistency and inter-class specificity to mine additional supervised information and obtain more discriminative features. Extensive experiments have shown that this model exhibits a promising performance with an accuracy of 74.4%, providing a potential approach for predicting lymph node metastasis in pancreatic cancer. An external evaluation was performed on the H&N1 dataset, further confirming the stability and generality of this model. The proposed methodologies represent a novel paradigm for the fine-grained classification of oncological images with small sample sizes. 
# Quick train and test
python btach_train.py
# Citation
Please cite the following paper if you use this repository in your research.
```
@inproceedings{
  title     = { A Dual-transformation with Contrastive Learning Framework for Lymph Node Metastasis Prediction in Pancreatic Cancer},
  author    = {X Chen, W Wang, Y Jiang, X Qian*},
  journal   = {Medical Image Analysis},
  month     = {April}，
  year      = {2023},
}
```

# Contact
For any question, feel free to contact
```
Xiahan Chen : chenxiahan@sjtu.edu.cn
```
