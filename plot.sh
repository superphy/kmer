for f in ampicillin chloramphenicol gentamicin kanamycin nalidixic\ acid spectinomycin streptomycin sulphonamides tetracycline trimethoprim
do
	python plot.py "-f" /home/rboothman/Data/lower_limits/results_salmonella/$f/ "-t" kmer Length and Cutoff "for" $f -xl kmer Length -yl Accuracy -xr 0 32 1 -yr 0.65 0.97 0.02 -ey 1
done

