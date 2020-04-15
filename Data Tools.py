import numpy as np
import matplotlib
import os

def load_logs(LogDir):
    QVAL = []
    LOSS = []
    FRAME_NUM = 0
    REWARD = []

    for file in sorted(os.listdir(LogDir), key=lambda x: int(x[5: -4]), reverse=False):
        if int(file[5: -4]) < FRAME_NUM:
            raise Exception
        FRAME_NUM = int(file[5: -4])
        logs = np.load(LogDir+'/'+file, allow_pickle=True)
        LOSS.extend(logs['arr_0'])
        REWARD.extend(logs['arr_1'])
        QVAL.extend(logs['arr_2'])



    LOSS, QVAL, REWARD = np.array(LOSS), np.array(QVAL), np.array(REWARD)
    print(LOSS.shape)
    print(QVAL.shape)
    print(REWARD.shape)

    return LOSS, QVAL, REWARD

def write_csv(LOSS, QVAL, REWARD, WriteDir):

    AVG_INCR = 10000
    sum_loss=0
    sum_Q=0
    count=0
    avg_loss=[]
    avg_Q=[]


    for i in range(len(LOSS)):
        sum_loss += LOSS[i]
        count+=1
        if count % AVG_INCR == 0:
            avg_loss.append(sum_loss / count)
            sum_loss = 0
            count=0
    avg_loss.append(sum_loss / count)

    for i in range(len(QVAL)):
        sum_Q += np.average(QVAL[i])
        count+=1
        if count % AVG_INCR == 0:
            avg_Q.append(sum_Q / count)
            sum_Q = 0
            count=0
    avg_Q.append(sum_Q / count)


    # with open(WriteDir+".csv", "w+") as writer:
    #     writer.writelines(lines)
    print(len(avg_loss))
    print(len(avg_Q))
    print(len(REWARD))

    lines = ["Loss, Q-Value, Reward, Incr\n"]
    for i in range(len(REWARD)):
        if i < len(avg_loss):
            lines.append(str(avg_loss[i])+', '+str(avg_Q[i])+', '+str(REWARD[i])+', '+str(AVG_INCR*i)+"\n")
        elif i < len(avg_Q):
            lines.append(' , '+str(avg_Q[i])+', '+str(REWARD[i])+', '+str(AVG_INCR*i)+"\n")
        else:
            lines.append(' ,  , '+str(REWARD[i])+', '+str(AVG_INCR*i)+"\n")

    with open(WriteDir+"Avg.csv", "w+") as writer:
        writer.writelines(lines)





LOSS, TD_ERROR, REWARD= load_logs("D:\Computer Science\Dissertation\dqns\Duelling")
write_csv(LOSS, TD_ERROR, REWARD, "D:\Computer Science\Dissertation\dqns\Duelling")