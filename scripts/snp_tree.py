from ete3 import Tree, TreeStyle, TextFace
import csv


def get_metadata():
    # metadata file
    md_dict = {}
    with open("../data/omnilog_metadata.csv", newline='') as mdfile:
        md_reader = csv.DictReader(mdfile, delimiter=',')
        for row in md_reader:
            md_dict[row['Strain']] = {
                'O': row['O type'],
                'H': row['H type'],
                'serotype': row['Serotype'],
                'host': row['Host'],
                'wgs': row['WGS']
            }
    return md_dict


def get_name_conversion():
    # Panseq output name conversion file
    nc_dict = {}
    with open("../data/phylip_rosetta.txt", newline='') as ncfile:
        nc_reader = csv.reader(ncfile, delimiter='\t')
        for row in nc_reader:
            nc_dict[row[0]] = row[2]
    return nc_dict


def get_tree(md, nc):
        # newick file
    t = Tree("../analyses/panseq/RAxML_bestTree.raxml_snp")
    ts = TreeStyle()
    ts.show_leaf_name = False
    # ts.mode = "c"
    # ts.arc_start = -180
    # ts.arc_span = 180

    for node in t.traverse():
        if node.is_leaf():
            nn = nc.get(node.name)
            serotype = md[nn]['serotype']
            if serotype == "O157:H7":
                nface = TextFace(serotype, fgcolor="blue", fsize=10)
                node.add_face(nface, column=0, position="branch-right")
            else:
                nface = TextFace(serotype, fgcolor = "red", fsize=10)
                node.add_face(nface, column=0, position="branch-right")

    return t, ts


if __name__ == "__main__":
    md = get_metadata()
    nc = get_name_conversion()
    t, ts = get_tree(md, nc)
    t.render("../manuscript/images/snp_tree.png", tree_style = ts, w = 1000)
