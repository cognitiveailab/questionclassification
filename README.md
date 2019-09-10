## Question Classification with the Aristo Reasoning Challenge (ARC) and the BERT-QC Model.

This is the github repository for the paper ["Multi-class Hierarchical Question Classification for Multiple Choice Science Exams" (Xu et al., 2019)](https://arxiv.org/abs/1908.05441).  

The contributions of this work are:
1. A new, large, detailed question classification dataset providing question classification labels for the [7,787 Aristo Reasoning Challenge (ARC) questions](http://data.allenai.org/arc/), which are drawn from elementary and middle school (3rd through 9th grade) science exams in the United States.
2. A hierarchical taxonomy of over 400 detailed question classes and problem domains for these science exam questions, developed based on exam syllabi, study guides, and a detailed data-driven analysis of the ARC questions. 
3. A BERT-based question classification model that performs well across 4 benchmark question classification datasets, matching or exceeding state-of-the-art performance on each dataset.
4. A BERT-based QA method that incorporates question classification information using query expansion, which is fast and easy to implement. 
5. The paper also includes a section on performing fast, detailed, and automated error analyses using these detailed question classification labels. 


## Data

### Question Classification Labels and Taxonomy
The question classification labels for the 7,787 ARC questions, as well as the 6-level hierarchical question classification taxonomy is available here:
http://cognitiveai.org/dist/arc-questionclassificationdata-2019.zip (2MB)

### Pre-compued Question Classification Predictions
Pre-computed predictions for our best question classification model, BERT-QC, across all datasets investigated in the paper (ARC, TREC, GARD, and MLBioMedLAT), that allow one to make use of the results of the BERT-QC model without having to run it yourself.  The pre-computed predictions are available here:
http://cognitiveai.org/dist/bertqc-precomputed-predictions.zip (49MB)

### Fine-tuned BERT-QC Question Classifier for ARC
An example fine-tuned BERT-QC model, trained using ARC at it's highest level of question classification granularity (L6).  This is useful if you would like to make QC predictions for new questions, but don't want to train the BERT-QC ARC model yourself.  Please note that this file is large (~1GB).
http://cognitiveai.org/dist/ARC-BERTQC-Base-Uncased-model-L6.zip  (1GB)

### License 
Please note that the ARC questions are subject to the terms set forth in the included license `EULA AI2 Mercury Dataset 01012018.docx`.

## Code

### Question Classification (BERT-QC)
Code to replicate the question BERT-QC question classification model experiments (Section 4, Tables 3, 4, 5, 6) is available here:
https://github.com/cognitiveailab/questionclassification/tree/master/qc

### Question Answering using Question Classification (QC+QA)
Code to replicate the QC+QA experiments (Section 5, Figure 2) is available here: 
https://github.com/cognitiveailab/questionclassification/tree/master/qc+qa

## Paper

### Paper
The paper is available at: https://arxiv.org/abs/1908.05441

### Citation
If you use this work, please cite as (BibTeX):
```
@article{xu2019multi,
  title={Multi-class Hierarchical Question Classification for Multiple Choice Science Exams},
  author={Xu, Dongfang and Jansen, Peter and Martin, Jaycie and Xie, Zhengnan and Yadav, Vikas and Madabushi, Harish Tayyar and Tafjord, Oyvind and Clark, Peter},
  journal={arXiv preprint arXiv:1908.05441},
  year={2019}
}
```


## Related Links

#### TREC Question Classification: 
https://cogcomp.seas.upenn.edu/Data/QA/QC/

#### GARD Question Classification
https://lhncbc.nlm.nih.gov/project/consumer-health-question-answering

#### MLBioMedLat Question Classification
https://github.com/wasimbhalli/Multi-label-Biomedical-QC-Corpus

## Related Papers
One of the central themes of the initial experimental section of this paper is that most popular existing methods for question classification developed on the TREC questions do not transfer to the significantly more complex science exam questions.  The observation that question classification models developed on TREC fail to transfer to other domains was detailed by Roberts et al. (2014) in the medical domain:
https://www.ncbi.nlm.nih.gov/pubmed/25954411


To the best of our knowledge, the current top-performance on the TREC dataset is "High Accuracy Rule-based Question Classification using Question Syntax and Semantics" (Madabushi and Lee, COLING 2016).  This paper uses an alternate approach of hand-crafted syntactic rules:
https://www.aclweb.org/anthology/C16-1116

## Contact
If you have any questions, comments, or issues, please feel free to contact Peter Jansen (pajansen@email.arizona.edu).
