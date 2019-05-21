### Bio
I am Gates Kempenaar and I am a 3rd year Computer Science student at the University of Lethbridge. During this term I will be using machine learning models to make phenotypic predictions for Shiga-toxin producing _Escherichia coli_ (STEC) using whole genome sequences (WGS) and omnilog phenotypic microarray data.

### Introduction
STEC vary in the frequency and severity of disease that they cause among humans. STEC are responsible for sporadic cases of illness as well as large scale outbreaks. Previous work has shown that although these organisms all produce Shiga-toxin, they have considerable differences in their genomic content. 

WGS is used for routine identification, characterization, and surveillance of STEC. While there is a wealth of genomic data there have been few large-scale studies linking phenotypic traits to specific differences in genome sequence. As well, the specific link between genotype and phenotype among factors that influence human illness, bacterial survival, and virulence is still largely unknown. There have been other studies focusing on well-known virulence factor differences, or broad differences between bacterial groups, but an in-depth genomic analysis of phenotypic differences among STEC has largely been absent. 

Phenotypes such as resistance to antimicrobials or presence of toxin gene can be attributed to the presence of absence of a single gene, while other phenotypes are more complex and not as easily attributed to specific single changes. As a result of this complexity we must look at the contributions of thousands of factors to individual phenotypes. TODO: omnilog information

Machine Learning (ML) models allows us to examine these complex relationships. Over this term I will analyze 192 genomes from 106 STEC serotypes in a comprehensive analysis of phenotype and genotype using Omnilog phenotypic microrrays and whole-genome sequencing. 

### Implementation
WGS data will be reformatted for use in the supervised machine learning models using Jellyfish.

The 192 bacterial isolates are from 106 serotypes, with 114 isolated from human hosts, 44 from bovine hosts, 27 from environmental smaples, and 5 from ovine hosts. 

DNA was extracted from cultured bacteria using the Epicentre MasterPure DNA Purification Kit and stored at -20C until it was 
needed. The DNA was then sequenced using the Illumina MiSeq platform at either the Core Services centre of the National 
Microbiology Laboratory (Winnipeg, Manitoba), or the McGill University and Genome Quebec Innovation Centre (Montreal, Quebec).

Experimental procedure for E. coli as detailed in the Omnilog Biolog manual (Biolog, California) was followed. The resulting kinetic bacterial growth curves were subjected to cubic spline fitting using the R package OPM, and the area under this curve was used in downstream analyses.

Predictions will be made using three separate machine learning models: Gradient Boosted Decision Trees (XGBoost), Artificial Neural Networks (ANN) and Support Vector Machines (SVM). The XGBoost model is implemented using the XGBoost Python package - the ANN and SVM are implemented using Keras, TensorFlow, and scikit-learn.

### Conclusion
Successful completion of this project will result in the rapid prediction of STEC phenotypes including host, serotype, and carbohydrate utilization, based on both WGS and omnilog microarray data. 

These predictions will allow us to:
- rapidly identify STEC more frequently associated with human illness, reducing time required for surveillance and outbreak investigations
- the identified omnilog substrates could be used for new differential and selective media
- identified genomic regions could elucidate fundamental genomic elements that contribute to complex phenotypes such as human disease and environmental persistence


