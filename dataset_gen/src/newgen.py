#Read in existing corpus and re-generate meta/testfiles.csv, trainfiles.csv, valfiles.csv
#If existing file exists, move them to testfiles.csv -> testfiles_<current_time>.csv

#Process each of the non-empty dirs under dialogs.
#Create a list L of (<filename>.tsv, dirname)
#Split L into 80-10-10 and write values into trainfiles.csv, valfiles.csv, testfiles.csv

'''
import os
import glob
directory = "/home/mpandey/ubuntu_dialogue/ubuntu-ranking-dataset-creator/src"
files = list(glob.glob(os.path.join(directory,'dialogs')))
print("files=", files)
'''

train_pct = 0.8
val_pct   = 0.1
test_pct  = 0.1

from os import listdir
directory = "/home/mpandey/ubuntu_dialogue/ubuntu-ranking-dataset-creator/src/dialogs"
files_dir =  listdir(directory)
#print("fd len=", len(files_dir))
datalist = []
for names in files_dir:
    #if names.endswith(".txt"):
    subdir_list = listdir(directory+"/"+names)
    #print("%s size = %d" % (names, len(subdir_list)))
    if(len(subdir_list) != 0):
        limit = 0
        for leafdir in subdir_list:
            entry = "%s,%s" % (leafdir, names)
            datalist.append(entry)
            if(limit < 10) and False:
                print("   Leaf = %s" % (directory+"##"+names+"%%"+leafdir))
                limit = limit+1

datalist_len = len(datalist)                
print "len=%d" % len(datalist)
trainlist  = datalist[:int(datalist_len*train_pct)]
vallist   = datalist[int(datalist_len*train_pct): int(datalist_len*(train_pct+val_pct))]
testlist  = datalist[int(datalist_len*(train_pct+val_pct)):]

print len(trainlist), len(vallist), len(testlist)

def write_dataset(filename, dataset):
    file = open(filename, "w")
    for dname in dataset:
        file.write(dname+"\n")
    file.close()
    
write_dataset("trainfiles.csv",trainlist)
write_dataset("testfiles.csv", testlist)
write_dataset("valfiles.csv",  vallist)
