import pandas as pd
import csv
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

temp = []
importance = [['L-Threonine', 4197.460037074031],
['Sodium Formate', 4019.52882565846],
['D-Tartaric Acid', 3592.216583095755],
['L-Pyroglutamic Acid', 3529.707137539738],
['β-Methyl-D-Xylopyranoside', 3278.614946800014],
['D,L-Octopamine' ,2856.678997833786],
['Oxalic Acid', 2759.1350390323205],
['D,L-Carnitine', 2325.198816863778],
['Sebacic Acid', 2079.9387230236316],
['β-Cyclodextrin' ,1978.9688133240393],
['Stachyose', 1974.037750018293],
['D-Lactic Acid Methyl Ester', 1920.7690754993346],
['Butyric Acid', 1679.1494447622256],
['Xylitol', 1668.5307998062733],
['α-Methyl-D-Mannoside', 1573.819339552497],
['α-Keto-Valeric Acid', 1497.9749180212873],
['Quinic Acid', 1434.3885428132064],
['Acetamide', 1357.635100114074],
['β-Hydroxy-Butyric Acid', 1349.9007397987057],
['L-Tartaric Acid', 1325.3424937301588],
['D-Ribono-1,4-Lactone', 1315.05384572482],
['Succinamic Acid', 1217.03434739752],
['Itaconic Acid', 1192.181556031168],
['L-Glutamic Acid', 1187.5479736831135],
['2,3-Butanediol', 1186.1814750332123],
['Melibionic Acid', 1097.4829432395736],
['Dextrin', 1082.8736402989052],
['4-Hydroxy-Benzoic Acid', 1029.1189459494442],
['Sedoheptulosan', 1025.6923553772276],
['2-Hydroxy-Benzoic Acid', 1010.718933212456],
['Butylamine [sec]', 1007.2162912235567],
['D-Melezitose', 833.2251231714736],
['Maltitol', 830.1925827209133],
['a-Hydroxy-Butyric Acid', 780.2104555682417],
['L-Glutamine', 762.8782638741579],
['Dihydroxy-Acetone', 756.7686520568432],
['a-Keto-Butyric Acid', 746.1842286966446],
['L-Valine', 713.5027809666223],
['L-Lysine', 710.0694747706405],
['γ-Hydroxy-Butyric Acid', 708.6124108822335],
['D-Aspartic Acid', 681.4610421006619],
['D-Arabinose', 669.8047724337234],
['Acetoacetic Acid', 597.7528648980511],
['3-O-Methyl-D-Glucose', 592.9562152159053],
['N-Acetyl-L-Glutamic Acid', 537.4184685485509],
['b-Phenylethylamine', 512.5178885880812],
['Glycogen', 509.93750571428495],
['Glucuronamide' ,495.1323666084003],
['D-Cellobiose', 473.67943618829634],
['D-Threonine', 436.80238806152454],
['β-D-Allose', 430.7858969791384],
['2,3-Butanedione', 426.56802793012037],
['Turanose', 421.2629786602512],
['Tyramine', 411.5616745025459],
['Chondroitin Sulfate C', 405.01207994080346],
['Mucic Acid', 397.75386713403014],
['a-Hydroxy-Glutaric Acid-g-Lactone', 393.9186790467535],
['Oxalomalic Acid', 391.9485912972045],
['4-Hydroxy-L-Proline [trans]', 375.88303029514486],
['L-Ornithine', 371.9660509517198],
['D-Psicose', 359.18714093678886],
['a-Keto-Glutaric Acid', 347.4943275422504],
['D-Tagatose', 342.0959204586218],
['α-Cyclodextrin', 336.754170366597],
['D-Salicin', 332.35269118416846],
['L-Lyxose', 330.33321894126254],
['Adonitol', 315.2554058490746],
['m-Tartaric Acid', 300.89551952373256],
['D-Lactitol', 297.72927190203984],
['δ-Amino-Valeric Acid' ,287.35645826255234],
['L-Proline', 284.9914401919701],
['Ethanolamine', 276.92675583558287],
['D-Glucosamine', 270.0957170702328],
['Tween 20', 268.66723158482625],
['Sorbic Acid', 268.3195599704912],
['L-Galactonic Acid-g-Lactone', 257.93122291675576],
['Lactulose', 244.49951582788083],
['D-Arabitol', 241.73853163742626],
['Bromo-Succinic Acid', 240.86714753450678],
['5-Keto-D-Gluconic Acid', 231.92551011629584],
['Methyl Pyruvate', 229.6348191163459],
['D-Saccharic Acid' ,228.39969327650437],
['Amygdalin', 226.743474924805],
['Fumaric Acid', 206.91077786664383],
['Malonic Acid', 203.14550790865297],
['L-Methionine' ,199.23892479690892],
['1,2-Propanediol', 191.99088270826158],
['Sucrose', 190.26227365404293],
['Glyoxylic Acid' ,180.01592873348946],
['Inulin', 176.85433500139362],
['myo-Inositol', 169.92631884249795],
['L-Alaninamide', 164.8396616150654],
['L-Homoserine', 162.42848820543136],
['D-Raffinose', 152.37559741503088],
['β-Methyl-D-Glucuronic Acid' ,147.88407649863072],
['Tween 80', 144.13605644918283],
['Propionic Acid', 141.9530301058087],
['L-Phenylalanine', 135.06244126380852],
['Dulcitol', 134.63788357034167],
['β-Methyl-D-Galactoside', 134.08936433425083],
['Putrescine', 130.02141250941176],
['Citric Acid', 126.46117711414954],
['Mannan', 121.46581873743779],
['D-Glucosaminic Acid', 116.18037639779902],
['L-Arginine', 96.92853424299365],
['D-Citramalic Acid', 93.63492830936855],
['L-Glucose', 91.99272237082559],
['D-Sorbitol', 89.50489854923181],
['α-Methyl-D-Glucoside', 83.84331817794383],
['D-Fructose', 83.13345454087485],
['D-Mannitol', 78.48409721232551],
['L-Leucine', 73.45357768887364],
['β-Gentiobiose', 73.40696813221055],
['L-Sorbose', 67.19940974398503],
['D-Mannose', 66.85238022647381],
['Tween 40', 65.76184419158903],
['L-Asparagine', 64.34031465229654],
['Mono-Methyl Succinate', 62.85329342206515],
['L-Isoleucine', 60.2343701915803],
['D,L-Malic Acid', 56.09411359197428],
['Citraconic Acid', 54.359642674346105],
['D-Xylose', 51.61754596481275],
['Inosine', 49.27907696462902],
['L-Malic Acid', 49.03405767123482],
['Negative Control', 46.52026097078519],
['b-Methyl-D-Glucoside', 45.82247905907819],
['D-Galactonic Acid-g-Lactone', 41.91046024907624],
['N-Acetyl-D-Glucosaminitol', 38.68237078704628],
['Caproic Acid', 35.342846808324225],
['D-Fucose', 34.31449411693012],
['Gly-Asp', 32.90005958524125],
['Maltotriose', 31.145740478373618],
['D-Trehalose', 30.84366083888625],
['p-Hydroxy-Phenylacetic Acid', 28.536355179657924],
['Glycolic Acid', 27.316476741310915],
['D-Maltose', 24.653666673390703],
['Palatinose', 24.20499976676108],
['m-Hydroxy-Phenylacetic Acid', 24.036040945602785],
['D-Malic Acid', 23.354080077749366],
['L-Arabitol', 23.237441605841468],
['D,L-a-Glycerol-Phosphate', 21.34379973534535],
['N-Acetyl-Neuraminic Acid', 19.765523686177495],
['L-Serine', 18.911132802911993],
['Pyruvic Acid', 18.587934368390556],
['Gly-Glu', 17.354360401879134],
['a-D-Glucose-1-Phosphate', 16.909015927040137],
['L-Histidine', 16.744193103669335],
['N-Acetyl-b-D-Mannosamine', 16.359067889444468],
['Thymidine', 16.344682477277043],
['D-Galacturonic Acid', 15.389088458568114],
['N-Acetyl-D-Galactosamine', 13.756799309286926],
['Ala-Gly', 12.410125435692276],
['Pectin', 12.238489685576422],
['L-Alanine', 11.243927347607082],
['γ-Amino-n-Butyric Acid', 9.64833533785539],
['Arbutin', 8.135093281343547],
['γ-Cyclodextrin', 7.680895336710361],
['L-Aspartic Acid', 7.539485547600854],
['Uridine', 6.407407094673941],
['L-Lactic Acid', 6.153795463678996],
['D-Glucuronic Acid', 5.0658696164652115],
['Glycine', 5.01656583439825],
['m-Erythritol', 4.8097077631677205],
['2-Deoxy-Adenosine', 4.776226955109646],
['2-Deoxy-D-Ribose', 4.6872727907927905],
['L-Rhamnose', 4.2338754335285635],
['Capric Acid', 4.1310723439562045],
['D-Alanine', 2.091196186611201],
['Glycerol', 2.0106492378266334],
['D-Glucose-6-Phosphate', 1.8572148897099956],
['L-Fucose', 1.849871032521639],
['3-O-β-D-Galactopyranosyl-D-Arabinose', 1.650467906454785],
['Gly-Pro', 1.4787058463517866],
['Adenosine', 1.439849849759443],
['D-Serine', 1.3237863667511647],
['D-Glucose', 1.014397864212206],
['a-Methyl-D-Galactoside', 0.9614876181914584],
['Succinic Acid', 0.7514484256133867],
['Tricarballylic Acid', 0.5553081188396095],
['D-Melibiose', 0.5507016834582892],
['Acetic Acid', 0.47171746096941525],
['D-Gluconic Acid', 0.40923829306344495],
['Laminarin', 0.22368047952597062],
['Gelatin', 0.20477619134254826],
['N-Acetyl-D-Glucosamine', 0.13881532562256305],
['D-Fructose-6-Phosphate', 0.1067672813525599],
['D-Galactose', 0.09442295598077874],
['D-Ribose', 0.05755410673702899],
['a-D-Lactose', 0.025340601242035795],
['L-Arabinose', 4.6194353422919394e-05]]

with open("data/omnilog_data_summary.txt", 'r') as read:
    for line in read:
        temp.append(line.split('\t'))

for line in temp:
    num = line[2].split('\n')
    num = float(num[0])
    line[2] = num

for i in temp:
    for j in importance:
        if i[1] == j[0]:
            i.append(j[1])

with open('data/final_omnilog_metadata.csv', 'r') as f:
    reader = csv.reader(f)
    omnilog = list(reader)

for line in temp:
    for j in omnilog:
        if line[0] in j[0]:
            if j[4] == "Human":
                line.append(0.0)
            else:
                line.append(1.0)
final = []
for line in temp:
    if len(line) == 5:
        final.append(line)

for line in final:
    if line[4] == 'Human':
        print(line)

df = pd.DataFrame(data = final, columns = ['Strain', 'Substrate', 'Area Under the Curve', 'Importance to Prediction', 'Substrate Type'])
substrate = df.pivot_table(index=['Area Under the Curve'], aggfunc='size')
print(substrate)
ax = sns.scatterplot(x = "Area Under the Curve", y = "Importance to Prediction", hue = "Substrate Type", data = df)
plt.tight_layout(pad = 1)
l = plt.legend()
l.get_texts()[1].set_text('Human')
l.get_texts()[2].set_text('Non-human')
plt.savefig('figures/auc_host.png')
