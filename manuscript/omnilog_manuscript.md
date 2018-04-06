---
title: Predictive marker discovery, and phenotypic classification of Shiga-toxin producting _Escherichia coli_ using machine learning models

author:
- Chad Laing
- Rylan Boothman
- Matthew Whiteside
- Akiff Manji
- Vic Gannon
date: \today{}
geometry: margin=2cm
header-includes:
- \usepackage{setspace}
- \doublespacing
- \usepackage{lineno}
- \linenumbers
---

<!---
To compile the PDF with pandoc, use:
  pandoc omnilog_manuscript.md tables/strain_data.md figure_captions.md -o omnilog_manuscript.pdf --csl=biomed-central.csl --filter pandoc-fignos --bibliography=omnilog_references.bib --pdf-engine=xelatex

  This requires having installed pandoc-fignos (eg. pip install pandoc-fignos)

  Also note that to produce a .docx, first convert to .tex, then convert the .tex to .docx. Otherwise the macros won't be expanded.

  pandoc omnilog_manuscript.md tables/strain_data.md figure_captions.md -o omnilog_manuscript.tex --csl=biomed-central.csl --filter pandoc-fignos --bibliography=omnilog_references.bib; pandoc omnilog_manuscript.tex -o omnilog_manuscript.docx; rm omnilog_manuscript.tex
-->

\include{values}

# Abstract
Among Shiga-toxin producing Escherichia coli (STEC) certain serotypes, and lineages within these serotypes, are more frequently associated with human disease. Despite the importance of STEC, the link between clade-specific genes and phenotypes that influence bacterial survival and virulence is still largely unknown. In this study we examined 143 STEC from 36 serotypes, using whole-genome sequencing, phenotypic microarray (PM) analyses, and machine learning (ML) models to establish these linkages.

PMs were analyzed using the Omnilog system (Biolog, CA). Genomes were sequenced using an Illumina MiSeq at Genome Quebec or the National Microbiology Laboratory in Winnipeg, Canada. Sequences were assembled using Spades v3.2.
Comparative genomics and statistical analyses were performed using methods from the SuperPhy platform.
Artificial neural network (ANN) models were implemented using Keras with a TensorFlow back end, composed of a single one-dimensional convolutional layer followed by a single fully connected layer; hyperparameters were optimized using hyperas. The linear support vector machine (SVM) model was implemented using scikit-learn.

The phylogeny based on single nucleotide polymorphisms (SNPs) among the 143 genomes was highly concordant with that based on the PM data. Phenotypic clades were largely divided among O- and H-type specific subgroups in both.
ML models trained on the PM data correctly predicted serotype 98.8% of the time using the ANN, and 77.67% using the SVM. Host classification as human / non-human using the PM data correctly predicted host source 68.4% of the time with the ANN model, and 65.4% of the time using the SVM.
The same models using kmer analyses of the corresponding WGS gave serotype prediction accuracy of 99.6% for the ANN, and 89.9% for the SVM. Host classification accuracy was 59.9% for the ANN, and 58.6% for the SVM.
Predictive markers, both phenotypic and genomic, were identified for all of the major phylogenetic clades, and for serotype-specific groups.

PM and WGS data were found to produce highly concordant phylogenies, and to both be useful as input for ML models . ANN in particular shows promise for the rapid and accurate predictive classification of STEC.  Sets of markers, both phenotypic and genomic, were identified for the predictive classification of STEC into phylogenetic and serotype groups. Potential implications of this work include the development of selective media for specific serotypes or lineages, and the rapid classification of bacteria into subgroups most frequently associated with human disease.

# Introduction

Shiga-toxin producing _Escherichia coli_ (STEC) are globally distributed and cause sporadic cases of human illness, as well as large-scale outbreaks [@Majowicz2014]. All members of this group are by definition capable of producing Shiga-toxin (Stx); however, the frequency and severity of disease caused by organisms within this gorup varies widely, as does the genomic composition of its members [@haugum_comparative_2014, @hazen_refining_2013, @ogura_extensive_2007]. Most studies of human illness caused by STEC have focused on serotype O157:H7, even though recent estimates suggest that at least half of all human infections are due to non-O157 STEC [@hughes_emerging_2006], and that over half of all patients with illness caused by STEC may go undiagnosed when selective STEC screening is implemented [@Majowicz2014]. Additionally, the United States Department of Agriculture has deemed the 6 serogroups that are responsible for 71% of non-O157 STEC infections as adulterants in beef products [@bosilevac_prevalence_2011, @USDA2012].

Despite the importance of STEC, the specific link between genotype and phenotype relating to factors that influence human illness, such as bacterial survival and virulence, is still largely unknown. While some genetic factors known to be associated with human disease have been identified within STEC, such as the presence of Shiga-toxin iteself, and other effector proteins [@Croxen2013], fine-grained analyses of phenotypic differences among STEC has largely been absent. Previous studies have focused mainly on well-known virulence factor differences [@Caceres2017, @Naseer2017, @Feng2017], or broad differences between bacterial groups, such as those associated with human illness and those not [@Dallman2015, @Norman2015], or those from different geographic regions or environmental niches [@Strachan2015, @Singh2015, @Ferdous2016].

The advent of cheap and easy whole-genome sequencing has allowed WGS to become part of the standard analyses for STEC, with national public health programs in the United States, the United Kingdom, and Canada having implemented WGS-based protocols for the routine identification, characterization, and surveillance of STEC [@Lindsey2016, @Chattaway2016, @Nadon2017].

Even though this wealth of genomic data exists, there have been comparatively few corresponding large-scale studies that link observed phenotypic traits to their corresponding genome. In the absence of phenotypic testing, phelogenetic clustering of bacterial strains has provided a reasonable proxy for phenotypic similarity [@Diodati2015, @Touchon2009]; however, studies of both O157 and non-O157 STEC have found that even in clonal populations, high levels of phenotypic variation can be observed, including traits known to be associated with virulence in humans [@Carter2016, @Dallman2015].

It is therefore important that phenotypes be quantitatively measured, to facilitate accurate predictive genomics wherein phenotypes can be accurately predicted from the genome sequence. Previous work linking genome to phenotype within STEC includes: the identification of genomic changes associated with increased adherence to human epithelial cells, as a proxy for human virulence [@Pielaat2015]; the accurate inference of antimicrobial resistance based on WGS [@Tyson2015]; and the development of microbiological risk assessment frameworks based on previously measured phenotypic traits and WGS [@Rantsiou2017].

The current state-of-the art for high-throughput, quantitative phenotypic assessment of bacteria, is the omnilog system by Biolog [Biolog (Biolog, Hayward, CA)]. This phenotypic microarray tests nutrient utlization and chemical sensitivity of microorganisms in a high-throughput manner. Previous studies using the Biolog system within STEC are primarily focused on STEC O157:H7, and include those that: have shown the interplay of glycolytic and gluconeogenic nutrients among probiotic _E. coli_ and the O157:H7 outbreak strain EDL933 [@schinner_escherichia_2015]; identified genomic and phenotypic markers of O157:H7 strains associated with the bovine reservoir [@eppinger_genome_2011]; identified high-oxidizing strains of O157:H7 as those that are longest survived in cattle manure [@franz_variability_2011]; and determined the phenotypic differences among O157:H7 outbreak strain Sakai, with and without the plasmid pO157, where it was found that lack of the plasmid was associated with enhanced survival in synthetic gastric fluid and poorer colonization of cattle [@lim_phenotypic_2010]. A single recent study looked at carbohydrate utilization differences among 37 STEC from 10 different serogroups, and identified three major groups based on their carbohydrate utilization, with specific utilization profiles for each group [@Kerangart2018].

While certain phenotypes are the result of single gene presence or absence, or of single nucleotide polymorphisms, such as the resistance to certain antimicrobials [@Hunt2017, @Jia2016], or the presence of toxin genes, for example Shiga-toxin or heat-stable enterotoxin [@Croxen2013], other phenotypes are less easily attributable to single changes, and likely involve the complex interplay of many biological systems. Within STEC this includes traits like the propensity to cause human disease, or the ability to survive in specific environmental niches. Such complex traits do not lend themselves to simple association studies of single changes, but instead require methods that are able to examine the relationships among thousands of factors [@Knights2011]. Supervised machine learning (ML) models are one such group of tools, and include random forest classifiers (RFCs), support vector machines (SVMs), and neural networks (NNs).

RFCs have been used to correctly classify _E. coli_ into the six major pathogroups using a minimal set of SNPs [@Roychowdhury2018], to predict the probability of association of STEC and other _E. coli_ with fresh produce [@Martinez-Garcia2016], and to identify unique groups of alleles that act as fingerprints for individual antibiotics within _E. coli_ [@Weiss2016]; SVM models have been used within STEC to classify O157:H7 strains according to host-source specificity [@Lupolova2016, @Lupolova2017], and to predict subcellular localization of proteins [@Yu2004]; NNs have been used to identify novel antimicrobial peptides [@Fjell2009], and to aid in bacterial colony counting [@Ferrari2017].

In this study, we examined 140 genomes from 36 STEC serotypes, including all of the "top 7", in a comprehensive comparison of both genotype and phenotype using Omnilog phenotypic microrrays and whole-genome sequencing. We identified a strong correlation between phylogeny and phenotypic profile, as well as similarities and differences among serotypes and phylogenetic clades. Unique sets of predictive markers, both the presence / absence of genomic regions, and SNPs, for these groups were identified and may be useful in creating selective media for STEC of particular phenotypes, or creating diagnostic tests based on sets of biomarkers.


# Methods

## Bacterial strains
One-hundred forty-three _E. coli_ bacterial strains, comprised of X serotypes were used in this study (Table 1).  Sixty-five strains were isolated from human hosts, 44 from bovine hosts, and 27 from environmental water samples. Additionally there were five isolates from ovine hosts, and one each from retail meat and a goose fecal sample.

## Omnilog Phenotypic Microarray Analyses
The experimental procedure for _E. coli_ as detailed in the Omnilog Biolog manual (Biolog, California) was followed. Briefly, frozen cultures were streak-plated for single colonies on blooad agar and grown overnight at 37C. A single colony from these initial plates were again sub-cultured and grown overnight as before. Single colonies from these second subcultures were used to inoculate the appropriate fluids for each PM plate. Three experimental replicates of each plate were conducted, and the colorimetric value for each of the 96 wells recorded every 15 minutes for 24hours. To normalize each well to a potentially different number of starting cells, despite the uniformity of the Biolog protocol, the colorimetric reading at time point 0 was subtracted from subsequent readings. The resulting kinetic bacterial growth curves were subjected to cubic spline fitting using the R package OPM [@Vaas2013], and the area under this curve was used in downstream analyses.

## Machine learning
K-mers were counted using Jellyfish [@Marcais2011]. To prepare the data for use in machine learning models, the Jellyfish output was processed to remove all k-mers that did not appear in every genome. Additionally, the input data was scaled so that all of the values lay within the range -1 and 1 using Scikit-learn's MinMaxScaler [@Pedregosa2012].

Feature selection was performed using SelectKBest, implemented in Scikit-learn (v0.19.0), with the ranking of features determined using the f-test [@Pedregosa2012].

Support vector machine and random forest classifiers were implemented in Scikit-learn [@Pedregosa2012]. The neural network was implemented in Keras (v2.0.5) [@chollet2015keras] using a (TensorFlow v1.3.0) backend [@Schapiro2013]. The hyperparameters of all three models were tuned using a combination of gridsearch and hyperas [@pumperla2016hyperas]. The support vector machine used a linear kernel with default parameters. The neural network consisted of a one-dimensional convolutional layer, followed by a flatten layer, and one dense layer. The random forest classifier contained 50 trees with a maximum tree depth of 100. The complete random forest and neural network parameters can be found at \url{https://github.com/superphy/kmer/blob/master/kmerprediction/models.py}.

Determination of features most predictive of the SVM model was accomplished by extracting the absolute value of the feature coefficients; for the random forest classifier, the built-in feature importance attribute was used. Identification of predictive features was calculated from 100 independent runs using the same model and data. Because the output of the models is non-deterministic, the results of each run were sorted from most predictive to least predictive and each feature was given a score equal to 2^-i, where i is index of a feature after sorting. The most important features were then determined by averaging the scores from all 100 runs  \url{https://github.com/superphy/kmer/blob/master/scripts/important_features.py}.


## Whole-Genome Sequencing
Bacteria were isolated from frozen glycerol stocks, streaked for single colonies on Blood Agar and incubated overnight at 37C. A single isolated colony was used to inoculate 5ml of Brain Heart Infusion liquid medium, which was incubated overnight at 37C. DNA was extracted from these cultures using the Epicentre MasterPure DNA Purification Kit (Epicentre, Madison, Wisconsin), and stored at -20C until needed. Sequencing of this DNA was performed using the Illumina MiSeq platform at either the Core Services centre of the National Microbiology Laboratory (Winnipeg, Manitoba), or the McGill University and Genome Quebec Innovation Centre (Montreal, Quebec).

## Genomic sequence analyses
Raw reads were assembled using Spades v2.5 [@bankevich_spades_2012]. Pan-genome analyses were conducted using Panseq with default settings [@laing_pan-genome_2010]. The phylogenetic tree was constructed using FastTree2 [@price_fasttree_2010]. Hierarchical clustering of omnilog data was performed using R. Tree similarity was computed using the Robinson-Foulds distance measure, with the ape package for R.

# Results
## Phylogeny
### Phylogeny based on WGS data
Using maximum likelihood analyses of SNP variation among shared genomic regions, the strains of this study clustered into phylogenetic groups that were partitioned according to serotype, as shown in Figure @fig:snp_tree. Larger clades that consisted of more than one serotype were frequently grouped by H-antigen type, as was the case for the H11 cluster that contained strains of serotypes O26:H11, O103:H11, and the H2 cluster that contained O103:H2 strains and O45:H2 strains. Conversely, a small cluster of O104 strains grouped by O-type, containing strains of O104:H2 and O104:H21. Strains of serogroup O104 were the most distributed among the tree, with O104:H7, O104:H4, O104:H21 and  O104:H2 all present in separate clades. Forty-six O157:H7 genomes were analyzed; these formed a separate cluster and additionally clustered according to the three known O157:H7 lineages.

### Phylogeny based on omnilog phenotypic microarray data



## Validation and selection of machine learning parameters

### Validation with previously published data
To validate the models used in this study, the tests performed in the Lupolova et al. (2016) [@Lupolova2016] paper were recreated. The Lupolova et al. study used protein variance cluster data as input to a support vector machine to predict isolate host. Their data set consisted of 185 _E. coli_ O157:H7 isolates from the UK and an additional 88 _E. coli_ O157:H7 isolates from the US. 91 of the UK samples came from a human host and 94 came from a bovine host, while 44 of the US samples came from a human host and the remaining 44 came from a bovine host. When using the UK isolates as the training set and the US isolates as the test set, Lupolova et al accurately predicted host 82% of the time. When training and testing on a random split of the combined UK and US data sets, Lupolva et al. accurately predicted host 78% of the time. To perform each test their "datasets were divided by 6 and cross-validated over 100 runs."

When using k-mer count data as input to an artificial neural network we received \ValidationNNSplit % accuracy with the data sets split and \ValidationNNMixed % accuracy with the data sets mixed. When using k-mer count data as input to a support vector machine we received \ValidationSVMSplit % accuracy with the data sets split and \ValidationSVMMixed % accuracy with the data sets mixed. Model accuracy was found by averaging results over 100 runs. When training and testing with the data sets mixed, a stratified shuffle split was used to determine the train and test splits for each run.

### Selection of kmer length and cutoff parameters
The optimal kmer length and kmer frequency cutoffs were determined by using the previously published dataset of Lupolova et al. (2016) [@Lupolova2016], where correct prediction of _E. coli_ host-source was examined. In this study, Kmer lengths from 3 - 31 were tested in combination with frequency cutoffs from 1 - 15. As can be seen in Figure @fig:kmer, increasing kmer lengths from 3 - 7 resulted in an increase in predictive accuracy for all kmer frequency cutoff values, culminating in a maximum predictive value of 87.1% at kmer frequency cutoff 13 and kmer length 7. Kmer lengths from 8 - 10 remained relatively constant at approximately 86% predictive ability for all kmer frequency cutoff lengths, following which a decline in predictive accuracy was observed for all combinations of kmer length and kmer frequency cutoff.

### Feature selection

The feature selection method used in this study was chosen after testing the SelectKBest using the f-test, SelectKBest using the chi-squared test, and Recursive Feature Elimination methods from Scikit-learn. All three methods perform feature selection by removing all but the k most predictive features. Values of k were tested from 10 to 1000, with accuracies at each k value determined by averaging the results of an SVM using kmer data with both the split and mixed Lupolava et al (2016) [@Lupolova2016] data sets. Each combination of data and feature selection method was tested 10 times. As can be seen in @fig:featureselection, all three feature selection methods show a general increase in predictive ability as k increases from 10 to 270 where SelectKBest by f-test and SelectKBest by chi-squared test both reach maximum values of 83.5% and 83.3% respectively. As k increases beyond 270, the SelectKBest by f-test starts a general down trend, the Recursive Feature Elimination method levels off, and the SelectKBest by chi-squared test has a sudden drop followed by another gradual increase in predictive ability.

### Data Augmentation

While designing the models, various data augmentation methods (artificially creating more training samples based on the current training data) were also tested. Since none of the methods provided a significant increase in model accuracy, data augmentation was not used in the final version of the models.


## Classification of genomes
The predictive ability of the three ML models using data from both the omnilog microarray and WGS was tested for the correct classification of isolate host and serotype. Both binary (category vs. not-category) and multiclass (correct assignment to any of the possible categories) predictions were tested.

### Serotype
The ability to classify O-group, H-group, and serotype was tested for all groups of bacteria that contained five or more members. As can be seen in Figure @fig:serotype when classifying an isolate as either belonging to or not belonging to a serotype of interest the models had a mean accuracy of \WGSSerotypeAcc % when using WGS data and \OmniSerotypeAcc % when using phenotypic data. Choice of model and serotype being predicted had little effect on the predictive ability of the models as the standard deviation of all predictions using WGS and phenotypic data was \WGSSerotypeStdAcc % and \OmniSerotypeStdAcc % respectively. As can be seen in Figure @fig:multiclass when classifying isolates as belonging to any of the possible serotypes the predictive ability of the WGS data continued to outperform the phenotypic data, but by a larger margin, with the average difference in accuracy between WGS and phenotypic data being \DiffMultiWGSMultiOmniSerotypeAcc % for multiclass predictions and \DiffWGSOmniSerotypeAcc % for binary predictions. When switching from making binary predictions to making multiclass predictions, the mean accuracy of the random forest and SVM models decreased by \DiffBinMultiSerotypeRFAcc % and \DiffBinMultiSerotypeSVMAcc % respectively, while the mean accuracy of the ANN increased to \MultiWGSSerotypeNNAcc % when using WGS data and \MultiOmniSerotypeNNAcc % when using phenotypic data.

Similar trends as when predicting serotype were also found when predicting O-group and H-group. As seen in Figures @fig:otype and @fig:htype, WGS data provided \DiffWGSOmniOtypeAcc % better accuracy than phenotypic data when predicting O-group and \DiffWGSOmniHtypeAcc % better accuracy when predicting H-group. As well, similar changes were found when switching from binary predictions to multiclass predictions, with the accuracy of the ANN increasing to \MultiOtypeNNAcc % and \MultiHtypeNNAcc % for O-group and H-group respectively,  and the accuracy of the other models decreasing by between \MinMultiBinOHtypeRFSVMAcc % and \MaxMultiBinOHtypeRFSVMAcc %.

### Host
The ability to classify host-source of bacteria was tested for all hosts with more than five representatives. As can be seen in Figure @fig:hosts when classifying an isolate as either orginating from or not orginating from a host-source of interest the models had a mean accuracy of \WGSHostAcc % when using WGS data and \OmniHostAcc % when using phenotypic data. This is in contrast to predicting serotype, where the models received higher accuracies when using WGS data than when using phenotypic data. When predicting serotype, the models received approximately equal accuracy for each serotype but, when predicting host-source, the models received \MinDiffSpecificHost % - \MaxDiffSpecificHost % higher accuracy when classifying ovine and water hosts as compared to classifying bovine and human hosts.

## Predictive features

### Serotype

### Host

# Tables
 \renewcommand{\arraystretch}{0.5}


