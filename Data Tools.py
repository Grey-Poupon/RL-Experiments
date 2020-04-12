import numpy as np
import matplotlib
import os

def load_logs(LogDir):
    TD_ERROR = []
    LOSS = []
    FRAME_NUM=[]

    for file in sorted(os.listdir(LogDir)):
        logs = np.load(LogDir+'/'+file, allow_pickle=True)
        FRAME_NUM.extend([int(file[5:-4])] * len(logs['arr_0']))
        LOSS.extend(logs['arr_0'])
        TD_ERROR.extend(logs['arr_1'])


    indx = np.argsort(FRAME_NUM)

    LOSS, TD_ERROR, FRAME_NUM = np.array(LOSS), np.array(TD_ERROR), np.array(FRAME_NUM)
    print(LOSS.shape)
    print(TD_ERROR.shape)
    print(FRAME_NUM.shape)
    return LOSS[indx], TD_ERROR[indx], FRAME_NUM[indx]

def write_csv(LOSS, TD_ERROR, FRAME_NUM, WriteDir):

    top_line="Frame Number, Loss, TD Error\n"
    lines = [top_line]
    AVG_INCR = 10000
    sum_loss=0
    sum_error=0
    count=0
    avg_loss=[]
    avg_error=[]


    for i in range(len(LOSS)):
        lines.append(str(i*4)+", "+str(LOSS[i])+", "+str(np.average(TD_ERROR[i]))+"\n")
        sum_loss += LOSS[i]
        sum_error += np.average(TD_ERROR[i])
        count+=1
        if count % AVG_INCR == 0:
            avg_loss.append(sum_loss/count)
            avg_error.append(sum_error / count)
            sum_loss = 0
            sum_error = 0
    avg_loss.append(sum_loss / count)
    avg_error.append(sum_error / count)


    # with open(WriteDir+".csv", "w+") as writer:
    #     writer.writelines(lines)

    lines = ["Loss, Error, Incr\n"]
    for i in range(len(avg_loss)):
        lines.append(str(avg_loss[i])+', '+str(avg_error[i])+', '+str(AVG_INCR*i)+"\n")

    with open(WriteDir+"Avg.csv", "w+") as writer:
        writer.writelines(lines)





LOSS, TD_ERROR, FRAME_NUM = load_logs("D:\Computer Science\Dissertation\dqns\BasePongLogs")
write_csv(LOSS, TD_ERROR, FRAME_NUM, "D:\Computer Science\Dissertation\dqns\BasePong")