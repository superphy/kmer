from utils import parse_metadata_2_


x_train,y_train,x_test,y_test = parse_metadata_2_('temp.csv', 'Fasta', 'Labels', 'TrainTest', 'Train', 'Test', 'Prefix ', ' Suffix', None)

print [str(x) for x in list(x_train)]
print y_train
print [str(x) for x in list(x_test)]
print y_test
