#coding=utf-8

import xlrd
import json
import numpy as np
class excel_read:
    def __init__(self, excel_path=r'C:\\Users\\Nancy\\Desktop\\action.xlsx',encoding='utf-8',index=0):

      self.data=xlrd.open_workbook(excel_path)  #获取文本对象
      self.table=self.data.sheets()[index]     #根据index获取某个sheet
      self.rows=self.table.nrows   #3获取当前sheet页面的总行数,把每一行数据作为list放到 list


    def get_data(self):
        result=[]
        for i in range(self.rows):
            col=self.table.row_values(i)  #获取每一列数据
            result.append(col)
            #print(result)
        return result

if __name__ == '__main__':
    action = excel_read().get_data()  #动作list
    state = excel_read(r'C:\\Users\\Nancy\\Desktop\\state.xlsx').get_data() #测试数据 状态list
    filename='state_cal.json'
    with open(filename) as file_obj:
        state_cal = json.load(file_obj)   #状态list
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    x = []
    y = []
    theta = []
    u = []
    v = []
    r = []
    x_sd = []
    y_sd = []
    theta_sd = []
    u_sd = []
    v_sd = []
    r_sd = []
    for i in range(len(action)):
        f1.append(action[i][0])
        f2.append(action[i][1])
        f3.append(action[i][2])
        f4.append(action[i][3])
    for i in range(len(state_cal)):
        x.append(state_cal[i][0])
        y.append(state_cal[i][1])
        theta.append(state_cal[i][2])
        u.append(state_cal[i][3])
        v.append(state_cal[i][4])
        r.append(state_cal[i][5])
    for i in range(len(state)):
        x_sd.append(state[i][0])
        y_sd.append(state[i][1])
        theta_sd.append(state[i][2])
        u_sd.append(state[i][3])
        v_sd.append(state[i][4])
        r_sd.append(state[i][5])

    #画图
    import matplotlib.pyplot as plt
    #action 图
    # time = []
    # for i in range(len(action)):
    #     time.append(i*0.2)
    # plt.ylim(-2,2)
    # plt.plot(time,f1,label='f1',color='blue')
    # plt.plot(time,f2,label='f2',color='red')
    # plt.plot(time,f3,label='f3',color='yellow')
    # plt.plot(time,f4,label='f4',color='purple')
    # plt.title('action')
    # plt.legend()
    # plt.show()

    # x,y图
    # plt.xlim(-1, 9)
    # my_x_ticks = np.arange(-1, 9, 1)
    # plt.xticks(my_x_ticks)
    # plt.ylim(2, 5)
    # plt.plot(x,y,label='calculate data')
    # plt.plot(x_sd,y_sd,label='given test data')
    # plt.legend()
    # plt.show()

    # # theta图
    # time = []
    # for i in range(len(state_cal)):
    #     time.append(i * 0.2)
    # # plt.ylim(-60, 60)
    # plt.plot(time, theta, label='calculate')
    # plt.plot(time, theta_sd, label='given')
    # plt.legend()
    # plt.show()

    # u,v,r图
    time = []
    for i in range(len(state_cal)):
        time.append(i * 0.2)
    plt.subplot(3,1,1)
    plt.plot(time, u, label='calculate')
    plt.plot(time, u_sd, label='given')
    plt.title('u')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(time, v, label='calculate')
    plt.plot(time, v_sd, label='given')
    plt.title('v')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(time, r, label='calculate')
    plt.plot(time, r_sd, label='given')
    plt.title('r')
    plt.legend()
    plt.show()