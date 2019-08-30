from ete3 import Tree, TreeStyle, TextFace
import csv

if __name__ == "__main__":
    t = Tree("data/kSNP/tree.parsimony.tre")
    ts = TreeStyle()
    ts.show_leaf_name = False

    with open('data/final_omnilog_metadata.csv', 'r') as f:
        reader = csv.reader(f)
        omnilog = list(reader)

    for node in t.traverse():
        if node.is_leaf():
            for list in omnilog:
                if node.name in list[0]:
                    if list[4] == "Human":
                        nface = TextFace(list[3], fgcolor="blue", fsize=10)
                        node.add_face(nface, column=0, position="branch-right")
                    else:
                        nface = TextFace(list[3], fgcolor = "red", fsize=10)
                        node.add_face(nface, column=0, position="branch-right")

    ts.rotation = 90
    t.render("figures/snp_tree.png", w = 3000, tree_style = ts)
