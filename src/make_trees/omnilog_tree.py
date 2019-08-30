from skbio.tree import nj
from skbio import DistanceMatrix
from collections import Counter
from scipy.spatial import distance_matrix

temp = []
dict = {}
count_list = []
substrate_list = ["Negative Control","L-Arabinose","N-Acetyl-D-Glucosamine","D-Saccharic Acid","Succinic Acid","D-Galactose","L-Aspartic Acid","L-Proline""D-Alanine","D-Trehalose","D-Mannose","Dulcitol","D-Serine","D-Sorbitol","Glycerol","L-Fucose","D-Glucuronic Acid","D-Gluconic Acid","D,L-Glycerol-Phosphate","D-Xylose","L-Lactic Acid","Formic Acid","D-Mannitol","L-Glutamic Acid","D-Glucose-6-Phosphate","D-Galactonic Acid-Lactone","D,L-Malic Acid","D-Ribose","Tween 20","L-Rhamnose","D-Fructose","Acetic Acid","D-Glucose","Maltose","D-Melibiose","ThymidineD-1L-Asparagine","D-Aspartic Acid","D-Glucosaminic Acid","1,2-Propanediol","Tween 40","Keto-Glutaric Acid","Keto-Butyric Acid","Methyl-D-Galactoside","D-Lactose","Lactulose","Sucrose","Uridine","L-Glutamine","m-Tartaric Acid","D-Glucose-1-Phosphate","D-Fructose-6-Phosphate","Tween 80","Hydroxy Glutaric Acid-Lactone","Hydroxy Butyric Acid","Methyl-D-Glucoside","Adonitol","Maltotriose","2-Deoxy Adenosine","Adenosine","Glycyl-L-Aspartic Acid","Citric Acid","m-Inositol","D-Threonine","Fumaric Acid","Bromo Succinic Acid","Propionic Acid","Mucic Acid","Glycolic Acid","Glyoxylic Acid","1D-Cellobiose","Inosine","Glycyl-L-Glutamic Acid","Tricarballylic Acid","L-Serine","L-Threonine","L-Alanine","L-Alanyl-Glycine","Acetoacetic Acid","N-Acetyl-D-Mannosamine","Mono Methyl Succinate","Methyl Pyruvate","D-Malic Acid","L-Malic Acid","Glycyl-L-Proline","p-Hydroxy Phenyl Acetic Acid","m-Hydroxy Phenyl Acetic Acid","Tyramine","D-Psicose","L-Lyxose","Glucuronamide","Pyruvic Acid","L-Galactonic Acid-Lactone","D-Galacturonic Acid","Phenylethyl-amine","2-Aminoethanol"]

for substrate in substrate_list:
    print(substrate)
with open("data/omnilog_data_summary.txt", 'r') as read:
    for line in read:
        temp.append(line.split('\t'))

for list in temp:
    dict[list[0]] = []

for list in temp:
    num = list[2].split('\n')
    num = num[0]
    count_list.append(list[1])
    dict[list[0]].append(float(num))

df = DataFrame(data = dict)
print(df)

#dm = DistanceMatrix(reduced, index)
#dm = distance_matrix(reduced, reduced)
#tree = nj(dm)
#print(tree.ascii_art())
