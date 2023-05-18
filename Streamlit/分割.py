import re
with open('requestments.txt','r') as file:


    for line in file:

        result = re.search(r'^[^@]+(?=@)', line )

        if result:
            text = result.group(0)
            print(text)
        else :
            print(line)




# list = []
# with open('requestments.txt','r') as file:
#
#     for line in file:
#         result = re.search(r'^[^@]+(?=@)', line)
#         if result:
#             text = result.group(0)
#             # list.append(text)
#             print(text)


# with open(gps_path, 'w+')as f_out:
#     for i in list:
#         s = i[0].zfill(4)
#         c = float(i[2])
#         d = float(i[3])
#         e = float(i[4])
#         print(c)
#         if c != -1:
#             f_out.write(s+' '+str(c)+' '+str(d)+' '+str(e)+'\n')