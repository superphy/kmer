import pandas as pd
import csv
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

temp = []
importance = [['Glycolic Acid', 22156.355548975826],
['D-Galactonic Acid-g-Lactone', 19990.352679944568],
['m-Hydroxy-Phenylacetic Acid', 18640.28591874295],
['D-Sorbitol', 16851.856548938726],
['3-O-Methyl-D-Glucose', 16046.018727480541],
['p-Hydroxy-Phenylacetic Acid', 14998.78120934073],
['m-Tartaric Acid', 13249.731835272496],
['D-Saccharic Acid', 13129.516446219455],
['5-Keto-D-Gluconic Acid', 11446.581591599912],
['D-Glucosaminic Acid', 9564.452791673059],
['Glyoxylic Acid', 9441.884731836797],
['L-Rhamnose', 7598.346410967137],
['Mucic Acid', 7003.1845963191145],
['Stachyose', 5820.2574064630435],
['L-Threonine', 5584.840315790427],
['b-Phenylethylamine', 5460.194636515886],
['Glucuronamide', 5394.167010201307],
['L-Lyxose', 5096.427569578933],
['Sedoheptulosan', 4896.777523217502],
['Butyric Acid', 4700.58746720985],
['myo-Inositol', 4476.627557359257],
['N-Acetyl-Neuraminic Acid', 4349.707837222181],
['Acetamide', 4296.9693398776235],
['D-Tartaric Acid', 4245.3023349351],
['Tween 20', 3748.7761556873966],
['L-Glutamine', 3679.6139522325393],
['Glycogen', 3545.3918532433563],
['4-Hydroxy-Benzoic Acid', 3457.8673899098726],
['Sorbic Acid', 3278.0128454700566],
['L-Pyroglutamic Acid', 3227.674198685204],
['β-Methyl-D-Xylopyranoside', 3124.4492973480087],
['Gly-Asp', 2913.6707879725327],
['Citric Acid', 2892.3372376494763],
['Mono-Methyl Succinate', 2861.3353625774057],
['Sodium Formate', 2672.641405836464],
['Pectin', 2464.6683942355576],
['D,L-Octopamine', 2387.0379801768986],
['L-Glutamic Acid', 2145.2651148895525],
['Dextrin', 2134.977686845702],
['Acetoacetic Acid', 2123.2534844465736],
['D-Glucosamine', 2069.245327339499],
['Tricarballylic Acid', 2045.074956084465],
['Bromo-Succinic Acid', 1977.4635310508966],
['Tween 40', 1941.8185399705012],
['2,3-Butanediol', 1774.163848746758],
['Putrescine', 1689.4398876252367],
['Ethanolamine', 1578.7958209114513],
['D-Ribono-1,4-Lactone', 1523.6558828668162],
['Tyramine', 1511.5415742648631],
['Oxalomalic Acid', 1506.3083518237113],
['Sebacic Acid', 1498.6995823891211],
['3-O-β-D-Galactopyranosyl-D-Arabinose', 1429.0668748016374],
['Tween 80', 1331.190867010458],
['L-Arabitol', 1267.0117465264439],
['L-Tartaric Acid', 1228.1743104068983],
['Glycine', 1206.9071666547156],
['α-Keto-Valeric Acid', 1165.2944054946145],
['m-Erythritol', 1102.692396529936],
['Succinamic Acid', 1063.1017008035292],
['D-Fucose', 1051.6292171279615],
['Sucrose', 1044.7273825180473],
['L-Leucine', 999.3309737610248],
['Arbutin', 952.6898491519943],
['Fumaric Acid', 910.0183974294969],
['Methyl Pyruvate', 908.3706513713947],
['Lactulose', 902.4419299813422],
['Inulin', 899.990530353159],
['Melibionic Acid', 882.767732049037],
['β-Cyclodextrin', 874.0059866465563],
['D-Serine', 870.2310739218676],
['Xylitol', 861.5015402803762],
['Citraconic Acid', 838.3795718794447],
['β-Hydroxy-Butyric Acid', 818.8684688856365],
['b-Methyl-D-Glucoside', 798.0253406131836],
['Quinic Acid', 789.4797237317013],
['Laminarin', 754.0554956220803],
['D,L-a-Glycerol-Phosphate', 723.6575929958412],
['L-Malic Acid', 638.4961382602519],
['γ-Hydroxy-Butyric Acid', 610.3026814984158],
['α-Cyclodextrin', 604.8178205551883],
['L-Arginine', 596.5201870940568],
['L-Asparagine', 571.2058919235005],
['Dihydroxy-Acetone', 545.4435542278607],
['D-Aspartic Acid', 520.0946566825938],
['N-Acetyl-b-D-Mannosamine', 498.8201950783652],
['Gly-Glu', 486.52836757843124],
['Dulcitol', 484.84366343792067],
['D,L-Malic Acid', 465.6660314319481],
['Chondroitin Sulfate C', 458.0636258348884],
['α-Methyl-D-Glucoside', 431.43500549444747],
['L-Ornithine', 394.88525300887045],
['Adonitol', 390.63589721945084],
['α-Methyl-D-Mannoside', 380.333680770673],
['D-Lactitol', 364.18564779853773],
['D-Malic Acid', 355.6920648804617],
['a-Keto-Butyric Acid', 329.7599035728367],
['Negative Control', 301.6178912245524],
['γ-Amino-n-Butyric Acid', 299.3678399679802],
['D-Raffinose', 291.9604724769443],
['L-Alanine', 284.86458496445624],
['D-Threonine', 278.12434573598534],
['1,2-Propanediol', 267.1385740582007],
['L-Proline', 256.34391045515883],
['D,L-Carnitine', 254.59590093094943],
['2-Hydroxy-Benzoic Acid', 252.05171146784198],
['L-Lysine', 242.22375512984667],
['a-D-Lactose', 221.4429609654526],
['Maltitol', 215.77910215563458],
['L-Sorbose', 205.87069075758382],
['L-Histidine', 201.0916314545267],
['a-Keto-Glutaric Acid', 195.81078269070807],
['D-Cellobiose', 190.68069597163117],
['Pyruvic Acid', 162.50401009995073],
['D-Citramalic Acid', 152.70274749583848],
['D-Melezitose', 152.6459522985532],
['δ-Amino-Valeric Acid', 151.67107856391289],
['Gly-Pro', 144.58062977882742],
['D-Melibiose', 143.00715820458205],
['Maltotriose', 141.21646950920655],
['a-D-Glucose-1-Phosphate', 135.30544774445139],
['a-Hydroxy-Glutaric Acid-g-Lactone', 130.14948620461013],
['D-Galacturonic Acid', 123.0718735646805],
['D-Psicose', 120.3623151315818],
['Ala-Gly', 119.41544999262271],
['D-Alanine', 97.26462100641257],
['2,3-Butanedione', 96.2400534171807],
['D-Fructose-6-Phosphate', 91.42129704934814],
['D-Lactic Acid Methyl Ester', 87.36640274981143],
['L-Valine', 82.31596137275173],
['D-Maltose', 81.19594037463642],
['D-Salicin', 78.68387819806559],
['Succinic Acid', 78.58392350306329],
['Amygdalin', 77.29470382681563],
['Capric Acid', 74.3808733358309],
['D-Arabitol', 74.2201835496581],
['Caproic Acid', 74.13188862125217],
['a-Methyl-D-Galactoside', 74.00630971415735],
['Oxalic Acid', 72.78249236069803],
['D-Galactose', 71.85373654059636],
['4-Hydroxy-L-Proline [trans]', 64.41358575752653],
['Turanose', 59.56397283895487],
['Inosine', 58.568056221211904],
['N-Acetyl-D-Glucosaminitol', 57.05350546009116],
['D-Arabinose', 53.546045808570895],
['D-Trehalose', 52.618183547000584],
['Thymidine', 52.53524236290423],
['D-Glucuronic Acid', 51.88683244352403],
['D-Mannitol', 49.925304688899004],
['Butylamine [sec]', 49.118877743969236],
['L-Isoleucine', 48.97606387825453],
['D-Gluconic Acid', 47.42712358421946],
['L-Homoserine', 47.30020248704923],
['L-Serine', 47.200267553243734],
['β-Methyl-D-Galactoside', 46.18516271526522],
['Propionic Acid', 41.78671706720089],
['Adenosine', 40.318138583154635],
['Itaconic Acid', 34.664516535448115],
['N-Acetyl-D-Glucosamine', 34.380007744193165],
['L-Glucose', 32.30123593250902],
['2-Deoxy-Adenosine', 31.803729643389666],
['D-Glucose-6-Phosphate', 29.90533788863329],
['a-Hydroxy-Butyric Acid', 29.66723870987049],
['2-Deoxy-D-Ribose', 29.460220583799995],
['N-Acetyl-D-Galactosamine', 28.65233441605635],
['L-Alaninamide', 24.456122828480847],
['L-Galactonic Acid-g-Lactone', 19.118029125889006],
['β-Gentiobiose', 18.909718496947544],
['Glycerol', 17.816537449180657],
['L-Methionine', 16.982705336269262],
['Palatinose', 15.498956603159012],
['β-D-Allose', 14.806821397188152],
['L-Lactic Acid', 14.295228929607838],
['D-Xylose', 12.85293458436835],
['Acetic Acid', 11.865291453550032],
['D-Fructose', 11.778584853412891],
['γ-Cyclodextrin', 11.136442259088124],
['D-Mannose', 8.937200941308985],
['Malonic Acid', 6.162492857925606],
['L-Phenylalanine', 5.8056477476403465],
['L-Arabinose', 3.8360904069156785],
['D-Ribose', 2.0474416054551807],
['L-Aspartic Acid', 1.483579395467729],
['Gelatin', 1.4223608980471565],
['Mannan', 1.1471830483262104],
['D-Tagatose', 0.9668373756377208],
['D-Glucose', 0.6118450907991742],
['N-Acetyl-L-Glutamic Acid', 0.29024574829026456],
['Uridine', 0.14342748031271188],
['L-Fucose', 0.0022194962067714017]]
'''
,
['1,2-Propanediol', 0],
['2,3-Butanediol', 0],
['2,3-Butanedione', 0],
['2-Deoxy-Adenosine', 0],
['2-Deoxy-D-Ribose', 0],
['2-Hydroxy-Benzoic Acid', 0],
['3-O-Methyl-D-Glucose', 0],
['3-O-β-D-Galactopyranosyl-D-Arabinose', 0],
['4-Hydroxy-Benzoic Acid', 0],
['4-Hydroxy-L-Proline [trans]', 0],
['5-Keto-D-Gluconic Acid', 0]]
'''
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
            if j[3] == "O157:H7":
                line.append(0.0)
            else:
                line.append(1.0)
final = []
for line in temp:
    if len(line) == 5:
        final.append(line)

for line in final:
    if line[4] == 'O157:H7':
        print(line)

df = pd.DataFrame(data = final, columns = ['Strain', 'Substrate', 'Area Under the Curve', 'Importance to Prediction', 'Substrate Type'])

ax = sns.scatterplot(x = "Area Under the Curve", y = "Importance to Prediction", hue = "Substrate Type", data = df)
plt.tight_layout(pad = 1)
l = plt.legend()
l.get_texts()[1].set_text('157:H7')
l.get_texts()[2].set_text('Non-O157:H7')
plt.savefig('figures/auc.png')
