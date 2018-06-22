# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""绘制饼图"""
def draw_pie(path):
    import matplotlib.pyplot as plt
    import pandas as pd

    data = open(path)
    data = pd.read_csv(data)

    # 女     
    female = data[data.gender == u'Female']
    # 男
    male = data[data.gender == u'Male']    

    # 调节图形大小，宽，高
    plt.figure(figsize=(5, 4))
    # 定义饼状图的标签，标签是列表
    labels = [u'Female(女)',u'Male(男)']
    # 每个标签占多大，会自动去算百分比
    sizes = [female.shape[0], male.shape[0]]
    colors = ['orange', 'lightskyblue']
    # 将某部分爆炸出来， 使用括号，将第一块分割出来，数值的大小是分割出来的与其他两块的间隙
    explode = (0.05, 0)
    
    patches, l_text, p_text = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                    labeldistance=1.1, autopct='%3.1f%%', shadow=True,
                                    startangle=90, pctdistance=0.6)

    # labeldistance，文本的位置离远点有多远，1.1指1.1倍半径的位置
    # autopct，圆里面的文本格式，%3.1f%%表示小数有三位，整数有一位的浮点数
    # shadow，饼是否有阴影
    # startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看
    # pctdistance，百分比的text离圆心的距离
    # patches, l_texts, p_texts，为了得到饼图的返回值，p_texts饼图内部文本的，l_texts饼图外label的文本
    for t in l_text:
        t.set_size = 15

    for t in p_text:
        t.set_size = 15

    # 设置x，y轴刻度一致，这样饼图才能是圆的
    plt.axis('equal')
    plt.title(u'性别比', size=30)
    plt.rc('font', family='STXihei', size=8)
    plt.legend()
    plt.savefig(u'图片\性别比.png', font='png')
    plt.show()
    plt.close()


'''绘制年龄直方图'''
def draw_bar(path):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    data = open(path)
    data = pd.read_csv(data)


    all_data = {'[0-10)': 0,'[10-20)': 0,'[20-30)': 0, '[30-40)': 0, '[40-50)': 0,
                '[50-60)': 0, '[60-70)': 0,'[70-80)': 0, '[80-90)': 0, '[90-100)': 0}

    for key, value in all_data.items():
        all_data[key] = data[data.age == key].shape[0]

    label = [key for key, value in all_data.items()]

    # 第一步，取出一张白纸
    fig = plt.figure(1)

    # 第二步，确定绘图范围，由于只需要画一张图，所以我们将整张白纸作为绘图的范围
    ax1 = plt.subplot(111)

    # 第三步，整理我们准备绘制的数据
    data = np.array([value for key, value in all_data.items()])

    # 第四步，准备绘制条形图，思考绘制条形图需要确定那些要素
    # 1、绘制的条形宽度
    # 2、绘制的条形位置(中心)
    # 3、条形图的高度（数据值）
    width = 0.5
    x_bar = np.arange(len(data))

    # 第五步，绘制条形图的主体，条形图实质上就是一系列的矩形元素，我们通过plt.bar函数来绘制条形图
    rect = ax1.bar(left=x_bar, height=data, width=width, color="lightblue")


    # 第六步，向各条形上添加数据标签
    for rec in rect:
        x = rec.get_x()
        height = rec.get_height()
        ax1.text(x, 1.02*height, str(height))

    # 第七步，绘制x，y坐标轴刻度及标签，标题
    ax1.set_xticks(x_bar)
    ax1.set_xticklabels(label)
    ax1.set_xlabel(u"糖尿病患者年龄分布")
    ax1.set_ylabel("频数")
    # x轴标签自动倾斜排放
    fig.autofmt_xdate()
    plt.legend()
    plt.savefig(u'图片\年龄分布.png', font='png')
    plt.show()
    plt.close()
    
    
'''绘制记录分布直方图'''
def draw_log_bar(all_data):
    import matplotlib.pyplot as plt
    import numpy as np
    import collections


    # 按照key排序
    all_data = collections.OrderedDict(sorted(all_data.items(), key=lambda t: t[0]))
    
    # 第二步，确定绘图范围，由于只需要画一张图，所以我们将整张白纸作为绘图的范围
    ax1 = plt.subplot(111)

    # 第三步，整理我们准备绘制的数据
    data = np.array([value for key, value in all_data.items()])

    # 第四步，准备绘制条形图，思考绘制条形图需要确定那些要素
    # 1、绘制的条形宽度
    # 2、绘制的条形位置(中心)
    # 3、条形图的高度（数据值）
    width = 0.8
    x_bar = np.arange(len(data))

    # 第五步，绘制条形图的主体，条形图实质上就是一系列的矩形元素，我们通过plt.bar函数来绘制条形图
    rect = ax1.bar(left=x_bar, height=data, width=width, color="blue")


    # 第六步，向各条形上添加数据标签
    for rec in rect:
        x = rec.get_x()
        height = rec.get_height()
        ax1.text(x+0.15, height+600, str(height), fontsize = 7, color='r')

    # 第七步，绘制x，y坐标轴刻度及标签，标题
    ax1.set_xticks(x_bar)
    ax1.set_xticklabels(list(all_data.keys()))
    ax1.set_xlabel(u"治疗频数")
    ax1.set_ylabel("治疗人数")
    plt.legend()
    plt.savefig(u'图片\治疗频数-人数分布.png', font='png')
    plt.show()
    plt.close()

    
"""问题一"""
def question_one(path):
    import pandas as pd

    data = open(path)
    data = pd.read_csv(data)

    data = data[[u'诊断号', u'病人号', u'出院去处', u'一级诊断', u'二级诊断', u'三级诊断', u'是否服用药物', u'再次入院']]

    data = data.replace({u'是否服用药物':'No'}, 0)
    data = data.replace({u'是否服用药物':'Yes'}, 1)
    
    # 选取有服用药物的记录    
    result = data[data[u'是否服用药物'] == 1]    
    # 以诊断号为index
    result.index = list(result[u'诊断号'])[0:]
    # 删除诊断号列
    del result[u'诊断号']
    # 根据诊断号进行排序
    result.sort_index(inplace=True)
    
    before = {u'一级诊断': 3, u'二级诊断': 2, u'三级诊断': 1, u'其他': 0}
        
    result.to_csv('数据\question_1_1.csv')
    
    # 处理再次入院列 (治疗后)
    for i in result.index:
        tmp = result.at[i, u'再次入院']
        if tmp == u'<30':
            result.at[i, u'再次入院'] = 2
        elif tmp == u'>30':
            result.at[i, u'再次入院'] = 1
        else:
            # 死亡
            if result.at[i, u'出院去处'] in [11, 13, 14, 19, 20, 21]:
                result.at[i, u'再次入院'] = 3
            # 非死亡
            else:
                result.at[i, u'再次入院'] = 0
    
    # 处理诊断列（治疗前）
    result[u'诊断'] = 0
    for i in result.index:
        for j in [u'一级诊断', u'二级诊断', u'三级诊断']:
            tmp = result.at[i, j]
            if len(tmp) >= 3 and (tmp[0:3] == '249' or tmp[0:3] == '250'):
                result.at[i, u'诊断'] = before[j]
                break;

    # 评价体系
    result[u'good'] = 0
    result[u'bad'] = 0
    
    final_result = 0
    good = 0
    bad = 0
    
    number = {}
    
    for name, group in result.groupby(u'病人号'):
        if group.shape[0] not in number.keys():
            number[group.shape[0]] = 1
        else:
            number[group.shape[0]] += 1
        
        if group.shape[0] == 1:
            if group.at[group.index[0], u'再次入院'] < group.at[group.index[0], u'诊断']:
                good += 1
                group.at[group.index[0], u'good'] += 1
            elif group.at[group.index[0], u'再次入院'] > group.at[group.index[0], u'诊断']:
                bad += 1
                group.at[group.index[0], u'bad'] += 1

            if type(final_result) == int:
                final_result = group
            else:
                final_result = pd.concat([final_result, group], axis=0)
        else:
            good1 = 0
            bad1 = 0

            good2 = 0
            bad2 = 0
            # 内部比较
            for i in group.index:
                if group.at[i, u'再次入院'] < group.at[i, u'诊断']:
                    good1 += 1
                elif group.at[i, u'再次入院'] > group.at[i, u'诊断']:
                    bad1 += 1

            weight1 = good1 + bad1
            if weight1 != 0:
                good1 = good1 / weight1
                bad1 = bad1 / weight1

            # 外部比较
            for i in range(group.shape[0])[1:]:
                if group.at[group.index[i], u'再次入院'] < group.at[group.index[i-1], u'再次入院'] and group.at[group.index[i], u'诊断'] < group.at[group.index[i-1], u'诊断']:
                    good2 += 1
                elif group.at[group.index[i], u'再次入院'] > group.at[group.index[i-1], u'再次入院'] and group.at[group.index[i], u'诊断'] > group.at[group.index[i-1], u'诊断']:
                    bad2 += 1

            weight2 = good2 + bad2
            if weight2 != 0:
                good2 = good2 / weight2
                bad2 = bad2 / weight2

            group[u'good'] = good1*0.5 + good2*0.5
            good += (good1*0.5 + good2*0.5)
            group[u'bad'] = bad1*0.5 + bad2*0.5
            bad += (bad1*0.5 + bad2*0.5)

            if type(final_result) == int:
                final_result = group
            else:
                final_result = pd.concat([final_result, group], axis=0)
    
    draw_log_bar(number)

    result.to_csv(u'数据\question_1_2.csv')
    final_result.sort_index(inplace=True)
    final_result.to_csv(u'数据\question_1_3.csv')
    # 诊断效果 0.6073947334607515
    print(good / (good + bad))


'''数据预处理'''
def deal_data(path):
    data = open(path)
    data = pd.read_csv(data)    

    
    # 体重 / 纳税人代码 / 诊疗医师专业 / 
    attribute = [u'诊断号', u'病人号', u'种族', u'性别', u'年龄', u'入院类型', u'出院去处',	u'入院来源',
                  u'入院天数', u'诊疗医师专业', u'实验室测试次数', u'其它测试数目', u'药物治疗次数', u'门诊次数', u'急诊次数',
                  u'住院次数',u'一级诊断', u'二级诊断', u'三级诊断', u'系统诊断次数', u'葡萄糖血清检测', u'HbA1C检测', u'二甲双胍', u'瑞格列奈', 
                  u'那格列胺', u'氯磺丙脲', u'格列美脲', u'乙酰苯磺酰环己脲', u'格列甲嗪',
                  u'优降糖', u'甲糖宁', u'匹格列酮', u'罗格列酮',u'阿卡波糖', u'米格列醇', u'曲格列酮', u'甲磺吖庚脲',
                  u'甲酰胺', u'西他列汀', u'胰岛素', u'优降糖，二甲双胍', u'格列甲嗪，二甲双胍', u'格列美脲，匹格列酮',
                  u'二甲双胍，罗格列酮', u'二甲双胍，匹格列酮', u'药物类型是否改变', u'是否服用药物', u'再次入院']
    
    # 选取重要的属性
    data = data[attribute]
    
    # 去除种族值为'?'
    data  = data[data[u'种族'] != u'?']

    # 剔除性别为'Unknown/Invalid'
    data = data[data[u'性别'] != u'Unknown/Invalid']

    # 去除出院去处为'18' NULL
    data = data[data[u'出院去处'] != 18]

    # 去除一级诊断为'?'
    data = data[data[u'一级诊断'] != u'?']

    # 去除二级诊断为'?'
    data = data[data[u'二级诊断'] != u'?']
    
    # 去除三级诊断为'?'
    data = data[data[u'三级诊断'] != u'?']
    
    # 去处一级诊断/二级诊断/三级诊断中有为V/E的值
    result = []
    for i in range(data.shape[0]):
        if data.at[data.index[i], u'一级诊断'][0] == 'V' or data.at[data.index[i], u'二级诊断'][0] == 'V' or data.at[data.index[i], u'三级诊断'][0] == 'V' or \
        data.at[data.index[i], u'一级诊断'][0] == 'E' or data.at[data.index[i], u'二级诊断'][0] == 'E' or data.at[data.index[i], u'三级诊断'][0] == 'E' :
            result.append(data.index[i])
    
    data.drop(result,inplace=True)

    data.to_csv(u'数据\question_2_0.csv', index=None)

    # 以诊断号为index
    data.index = list(data[u'诊断号'])[0:]
    # 删除诊断号列
    del data[u'诊断号']
    # 根据诊断号进行排序
    data.sort_index(inplace=True)    
    
    # 种族
#    race = {'Asian': 0, 'Hispanic': 1, 'Caucasian': 2, 'AfricanAmerican': 3, 'Other': 4}
    data[u'种族'].replace(['Asian', 'Hispanic', 'Caucasian', 'AfricanAmerican', 'Other'], [0, 1, 2, 3, 4], inplace=True)
    
    # 性别
#    gender = {'Male': 0, 'Female': 1}
    data[u'性别'].replace(['Male', 'Female'], [0, 1], inplace=True)
    
    # 年龄
#    age = {'[0-10)': 0,'[10-20)': 1,'[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
#           '[50-60)': 5, '[60-70)': 6,'[70-80)': 7, '[80-90)': 8, '[90-100)': 9}
    data[u'年龄'].replace(['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
           '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
    
    # 葡萄糖血清检测    
#    max_glu_serum = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
    data[u'葡萄糖血清检测'].replace(['None', 'Norm', '>200', '>300'], [0, 1, 2, 3], inplace=True) 
    
    # HbA1C检测
#    A1Cresult = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
    data[u'HbA1C检测'].replace(['None', 'Norm', '>7', '>8'], [0, 1, 2, 3], inplace=True)
    
    # 药物类型是否改变
#    change = {'No': 0, 'Ch': 1}
    data[u'药物类型是否改变'].replace(['No', 'Ch'], [0, 1], inplace=True)        
    
    # 是否服用药物
#    diabetesMed = {'No': 0, 'Yes': 1}
    data[u'是否服用药物'].replace(['No', 'Yes'], [0, 1], inplace=True)
    
    # 再次入院
#    readmitted = {'NO':0, '<30': 1, '>30':1}
    data[u'再次入院'].replace(['NO', '<30', '>30'], [0, 1, 0], inplace=True)

    # 药物
    data = data.replace('No', 0)
    data = data.replace('Down', 1)
    data = data.replace('Steady', 2)
    data = data.replace('Up', 3)

    data.to_csv(u'数据\question_2_1.csv', index=None)


'''问题二'''    
def question_two(X): 
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn import metrics

    kmean = KMeans(n_clusters=4, random_state=9)
    y_pred = kmean.fit_predict(X)
    
    data = X.copy()
    data[u'标签'] = kmean.labels_
    for i in range(4):
        tmp = data[data[u'标签'] == i]
        print('类簇人数占人数比重',tmp.shape[0] / data.shape[0])
        print(u'一级诊断 icd9编码范围:', min(tmp[u'一级诊断']), max(tmp[u'一级诊断']))
        print(u'二级诊断 icd9编码范围:', min(tmp[u'二级诊断']), max(tmp[u'三级诊断']))
        print(u'三级诊断 icd9编码范围:', min(tmp[u'三级诊断']), max(tmp[u'三级诊断']))
    
    # 绘制三维图
    x, y, z = list(map(eval, list(X[u'一级诊断']))), list(map(eval, list(X[u'二级诊断']))), list(map(eval, list(X[u'三级诊断'])))
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #  将数据点分成三部分画，在颜色上有区分度
    ax.scatter(x, y, z, c=y_pred, s=0.1)  # 绘制数据点
    
    ax.set_zlabel('diag_3')  # 坐标轴
    ax.set_ylabel('diag_2')
    ax.set_xlabel('diag_1')
    plt.show()
    print(metrics.calinski_harabaz_score(X, y_pred))    
    
    
    # 寻找最优的k值结果
    result = []
    for i in range(100)[3:]:    
        y_pred = KMeans(n_clusters=i, random_state=9).fit_predict(X)
        tmp = metrics.calinski_harabaz_score(X, y_pred)
        result.append(tmp)
        print("Calinski-Harabasz Score", tmp)
        
    plt.scatter([i for i in range(100)[3:]], result, alpha=0.5)
    plt.show()


'''问题三 1'''
def question_three(path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    file = open(path)
    data = pd.read_csv(file)
    # 剔除u'诊疗医师专业'列

    attribute = [u'种族', u'性别', u'年龄', u'出院去处', u'入院来源', u'入院天数', u'诊疗医师专业', u'HbA1C检测', u'葡萄糖血清检测', u'一级诊断']
    xlabels = [u'race', u'gender', u'age', u'discharge_disposition_id', u'admission_source_id', u'time_in_hospital',
               u'medical_specialty', u'A1Cresult', u'max_glu_serum', u'diag_1']

    data = data[attribute]

    for i in range(len(attribute)):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        group_labels = list(set(data[attribute[i]]))
        number = []
        for j in group_labels:
            number.append(data[data[attribute[i]] == j].shape[0] / data.shape[0])

        x = [j for j in range(len(group_labels))]

        # 阶数为5阶
        if len(x) >= 5:
            order = 6
        else:
            order = len(x)

        # 计算多项式 拟合多项式的系数存储在数组c中
        c = np.polyfit(x, number, order)
        print(xlabels[i], c)
        # 进行曲线绘制
        x_new = np.linspace(0, len(x) - 1, 2000)
        f_liner = np.polyval(c, x_new)
        ax.plot(x_new, f_liner, label=u'拟合多项式曲线', color='g', linestyle='-', marker='')

        plt.legend(loc=1)
        plt.plot(x, number, 'orange', marker='o', label=u'数据分布曲线')
        plt.xticks(x, group_labels, rotation=0)
        plt.ylabel('probability')
        plt.xlabel(xlabels[i])
        plt.savefig(u'图片\概率分布' + str(i) + u'.png', font='png')
        plt.show()
        plt.close()


'''问题三 2'''
def question_three_one(path):
    from sklearn.cluster import DBSCAN
    import collections
    import matplotlib.pyplot as plt 
    import numpy as np
    from scipy import stats
    
    file = open(path)
    data = pd.read_csv(file)

    # 绘制皮尔逊相关系数热图
    corMat = pd.DataFrame(data[[u'种族', u'性别', u'年龄', u'出院去处', u'入院来源', u'入院天数', u'HbA1C检测', u'葡萄糖血清检测', u'一级诊断', u'再次入院']].corr())
    print(corMat)
    # 绘制皮尔逊相关系数热图
    plt.pcolor(corMat)
    plt.savefig(u'图片\\皮尔逊相关系数热图.png', font='png')
    plt.show()
    plt.close()
    
    data = data[data[u'是否服用药物'] == 1]
    tmp = data[[u'种族', u'性别', u'年龄', u'出院去处', u'入院来源', u'入院天数', u'HbA1C检测', u'葡萄糖血清检测']]
    dascan = DBSCAN()
    y_pred = dascan.fit_predict(tmp)
    label = set(y_pred)
    
    data['label'] = y_pred
    
    result = {}
    for i in label:
        tmp = data[data['label'] == i]
        rate = round(tmp[tmp['再次入院'] == 1].shape[0] / tmp.shape[0], 4)
        if rate not in result.keys():
            result[rate] = 1
        else:
            result[rate] += 1
    
    
    # 按照key排序
    all_data = collections.OrderedDict(sorted(result.items(), key=lambda t: t[0]))
    
    x = list(all_data.keys())
    y = list(all_data.values())
    
    plt.figure(figsize=(10, 6))
    plt.legend(loc=1)
    plt.plot(x, y, 'purple', marker='o',mec='black',label=u'数据分布曲线')
    plt.xlabel(u'再入院率')
    plt.ylabel('频数')
    plt.savefig(u'图片\\再入院率-频数.png', font='png')
    plt.show()
    plt.close()
    
    print(all_data)
            
    print(len(set(dascan.labels_)))
    
    # 数组
    x=np.array(x)
    # 平均值
    mean=x.mean()
    # 标准差
    std=x.std()
    
    # bata双侧置信区间
    for i in [0.5, 0.9, 0.95]:
        interval = stats.t.interval(i, len(x)-1, mean,std)
        print(interval)    
        

'''问题四'''
def question_four(path):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    file = open(path)
    data = pd.read_csv(file)
    attribute = [u'种族', u'性别', u'年龄', u'出院去处', u'入院来源', u'入院天数', u'HbA1C检测', u'葡萄糖血清检测']

    data = data[data[u'是否服用药物'] == 'Yes']
    print(data.shape[0])
    
    # HbA1C检测
    for i in (data.index):
        if data.at[i, u'药物类型是否改变'] == 'Ch' and (data.at[i, u'HbA1C检测'] == '>7' or data.at[i, u'HbA1C检测'] == '>8'):
            data.at[i, u'HbA1C检测'] = u'high,med.changed'
        elif data.at[i, u'药物类型是否改变'] == 'No' and (data.at[i, u'HbA1C检测'] == '>7' or data.at[i, u'HbA1C检测'] == '>8'):
            data.at[i, u'HbA1C检测'] = u'high,med.not changed'    
    
    # 年龄
    data[u'年龄'].replace(['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
           '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], 
    [u'30 years old or younger', u'30 years old or younger',
    u'30 years old or younger', u'30–60 years old', u'30–60 years old', u'30–60 years old',
    u'Older than 60', u'Older than 60', u'Older than 60', u'Older than 60'], inplace=True)
    
    # 出院去处
    data[u'出院去处'].replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        [u' Discharged to home', 'Otherwise', 'Otherwise', 'Otherwise', 'Otherwise'
         , 'Otherwise', 'Otherwise', 'Otherwise', 'Otherwise', 'Otherwise', 'Otherwise'
         , 'Otherwise', 'Otherwise', 'Otherwise', 'Otherwise', 'Otherwise', 'Otherwise'
         , 'Otherwise', 'Otherwise', 'Otherwise', 'Otherwise', 'Otherwise', 'Otherwise'
         , 'Otherwise', 'Otherwise', 'Otherwise', 'Otherwise', 'Otherwise', 'Otherwise'], inplace=True)
 
    # 入院来源
    data[u'入院来源'].replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26],[u'Admitted because of physician/clinic referral',
        u'Admitted because of physician/clinic referral',u'Otherwise',u'Otherwise',u'Otherwise',
        u'Otherwise',u'Admitted from emergency room',u'Otherwise',u'Otherwise',u'Otherwise',u'Otherwise'
        ,u'Otherwise',u'Otherwise',u'Otherwise',u'Otherwise',u'Otherwise',u'Otherwise',u'Otherwise',
        u'Otherwise',u'Otherwise',u'Otherwise',u'Otherwise',u'Otherwise',u'Otherwise',u'Otherwise',u'Otherwise'
        ], inplace=True)
    
    # 入院天数
    data[u'入院天数'].replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],[u'Within five days', u'Within five days',
         u'Within five days', u'Within five days', u'Within five days', u'Within ten days', u'Within ten days',
          u'Within ten days', u'Within ten days', u'Within ten days', u'Within fifteen days', u'Within fifteen days',
           u'Within fifteen days', u'Within fifteen days'], inplace=True)
 
    # 计算再入院率比例
    for value in attribute:
        print('------------------', value,'------------------')
        key = data[[value, u'再次入院']]
        x = []
        y = []
        for i in set(key[value]):
            tmp = key[key[value] == i]
            print(i, '：Number of encounters', tmp.shape[0], 'rate', tmp.shape[0] / data.shape[0])
            pmt = tmp[tmp[u'再次入院'] == u'<30']
            print(i, '：Number of encounters', pmt.shape[0], 'rate', pmt.shape[0] / tmp.shape[0])
            x.append(i)
            y.append(pmt.shape[0] / tmp.shape[0])
        
        plt.figure()
        plt.legend(loc=1)
        plt.plot([i for i in range(len(x))], y, 'purple', marker='o',mec='black',label=u'数据分布曲线')
        plt.xticks([i for i in range(len(x))], x, rotation=0)
        plt.ylabel('probability')
        plt.xlabel(value)
        plt.ylim(0.05, 0.15)
        plt.savefig(u'图片\\属性-再入院率' + value + u'.png', font='png')
        plt.show()
        plt.close()


if __name__ == '__main__':
    import time
    import pandas as pd
    
    start = time.clock()
        
    # path = u'数据\diabetic_data.csv'
    # # 绘制饼图
    # draw_pie(path)
    # # 绘制年龄直方图
    # draw_bar(path)

    
    # 问题一
    # path = u'数据\diabetic_data_翻译2.csv'
    # question_one(path)
    
    
    # 问题二    
    # path = u'数据\diabetic_data_翻译2.csv'
    # 数据清洗
    # deal_data(path)
    # file = open(u'数据\question_2_1.csv')
    # data = pd.read_csv(file, encoding='utf-8')
    # question_two(data[[u'一级诊断', u'二级诊断', u'三级诊断']])


    # 问题三 1
    # path = u'数据\question_2_0.csv'
    # question_three(path)

    # 问题三 2
    path = u'数据/question_2_1.csv'
    question_three_one(path)

    # 问题四
    # path = u'数据/question_2_0.csv'    
    # question_four(path)

    end = time.clock()
    print(end - start)
