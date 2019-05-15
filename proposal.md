### Bio
I am Gates Kempenaar and I am a 3rd year Computer Science student at the University of Lethbridge. During this term I will be 
developing tools to predict differnt serotypes of shiga toxin producing _Escherichia coli_ (STEC) using the relationship between 
genotype, phenotype and whole genome sequences (WGS). I will be focusing on both O157:H7 and non-O157:H7 serotypes of STEC. 

### Introduction
Some serotypes of _E. coli_ can produce shiga toxin while the frequency and severity of the disease caused by organisms within 
this group varies widely, as does the genomic composition of its members. This group of STEC is globally distributed and 
responsible for occasional human illness as well as large scale outbreaks. 

As WGS has become more affordable and easier to access it has since become the standard for STEC analysis. WGS is used for 
routine identification, characterization, and surveillance of STEC. 

High levels of phenotypic variation can be observed in clonal populations of both O157:H7 and non-O157:H7, this includes traits 
associated with virulence in humans. Quantitatively measuring phenotypes to facilitate accurate predictive genomics wherein 
phenotypes can be accurately predicted from the genome sequence. Phenotypes such as resistance to antimicrobials or presence of 
toxin gene can be attributed to a single gene presence or absence while other phenotypes are more complex and not as easily 
attributed to specific single changes. As a result of this complexity we cannot look at single changes but instead need to look 
at relationships across thousands of factors.

Machine Learning (ML) is a tool we can use to look at these complex relationships. Over this term I will be examining 187 genomes 
from X STEC serotypes in a comprehensive analysis of phenotype and genotype using Omnilog phenotypic microrrays and whole-genome 
sequencing. 

### Implementation
WGS data will be reformatted for use in the supervised machine learning models that will be used to predict serotypes.

Bacterial strains from 187 E. coli genomes were used in this study made up of X different serotypes. Of this data X were isolated 
from human hosts, X from bovine hosts, X from environmental water hosts and an additional X isolates from ovine hosts. We can 
then use ML models to make predictions on serotype. 

DNA was extracted from cultured bacteria using the Epicentre MasterPure DNA Purification Kit and stored at -20C until it was 
needed. The DNA was then sequenced using the Illumina MiSeq platform at either the Core Services centre of the National 
Microbiology Laboratory (Winnipeg, Manitoba), or the McGill University and Genome Quebec Innovation Centre (Montreal, Quebec).

Predictions will be made using three seperate machine learning models: Gradient Boosted Decision Trees (XGBoost), artificial 
neural networks (ANN) and Support Vector Machines (SVM). The XGBoost model will be used as implemented using the XGBoost Python 
package - the ANN and SVM are implemented using Keras, TensorFlow, and scikit-learn.

