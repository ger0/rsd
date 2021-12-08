# do uproszczenia
def dataDictionary (filePath):
    tempList = []
    dataDict = {}
    i = 0

    file = open(filePath, 'r')
    for line in file:
        tempList.append(line.strip().split())

        if (len(tempList[i]) == (1+1+2)):
            dataDict[str(tempList[i][0])] = [tempList[i][1:]]

        elif (len(tempList[i]) == (1+2+4)):
            dataDict[str(tempList[i][0])] = [tempList[i][1:4], tempList[i][4:7]]

        elif (len(tempList[i]) == (1+3+6)):
            dataDict[str(tempList[i][0])] = [tempList[i][1:4], tempList[i][4:7], tempList[i][7:9]]

        elif (len(tempList[i]) == (1+4+8)):
            dataDict[str(tempList[i][0])] = [tempList[i][1:4], tempList[i][4:7], tempList[i][7:9], tempList[i][9:11]]

        i += 1
    return dataDict
