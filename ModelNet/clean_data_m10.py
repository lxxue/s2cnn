import os
import glob
import shutil

file_lists = glob.glob("ModelNet10/*/*/*.off")
#print(file_lists)

total = 0
wrong = 0
for file_name in file_lists:
    total += 1
    from_file = open(file_name)
    header = from_file.readline()
    if len(header) > 4:
        to_file = open("temp.txt", "w")
        #print(header)
        wrong += 1
        to_file.write(header[:3]+'\n')
        to_file.write(header[3:])
        shutil.copyfileobj(from_file, to_file)
        to_file.close()

    from_file.close()
    if len(header) > 4:
        os.system("mv temp.txt {}".format(file_name))

print(total)
print(wrong)

