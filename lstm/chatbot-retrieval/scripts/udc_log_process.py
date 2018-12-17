#!/usr/bin/python3
#Madhulima Pandey -- postprocess output of udc_train.py into a CSV format
###!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3


import re
import sys
import datetime

#Look for strings of the form
#                                    1                      2                                3                  4                                   5                        6
#INFO:tensorflow:Validation (step 136000): recall_at_1 = 0.39754601226993863, recall_at_10 = 1.0, recall_at_2 = 0.5896728016359918, recall_at_5 = 0.8580265848670756, loss = 1.3154824, global_step = 135347

if (len(sys.argv) != 2):
    sys.exit("Expect 1 argument: Usage udc_process.py <log file>")
    
#Group1 - Step #
#Group2 - recall_1
#Group3 - recall_10
#Group4 - recall_2
#Group5 - recall_5
#Group6 - loss

match_record = re.compile(b"^INFO:tensorflow:Validation\s\(step[\s]+([\d]+)\)\:[\s]+recall_at_1[\s]+\=\s([\.\d]+),[\s]+recall_at_10[\s]=[\s]([\.\d]+),[\s]+recall_at_2[\s]=[\s]([\.\d]+),[\s]+recall_at_5[\s]+=[\s]+([\.\d]+)+,[\s]+loss[\s]+=[\s]+([\.\d]+),\sglobal_step\s=\s([\d]+)[\s]*").match

f = open(sys.argv[1], "rb")

print("Step_number, Recall_1, Recall_2, Recall_5, Recall_10, Loss")
for line in f:
    match = match_record(line)
    if match is not None:
        #print(line)
        #print("**** Matched = ", match.groups(), "len=%d"%(len(match.groups())))
        print("%s, %s, %s, %s, %s, %s" % ((match.group(1).decode('utf-8')),
                                          (match.group(2)).decode('utf-8'),
                                          (match.group(4)).decode('utf-8'),
                                          (match.group(5)).decode('utf-8'),
                                          (match.group(3)).decode('utf-8'),
                                          (match.group(6)).decode('utf-8')))
        
