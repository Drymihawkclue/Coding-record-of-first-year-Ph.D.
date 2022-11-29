import pandas as pd
import os
#print(os.system('c:\\windows\\system32\\mspaint.exe'))
os.chdir(r'D:\DataAnalysis\final')
retval=os.getcwd()
print ("当前工作目录为 %s" % retval)

def loadData():
    traindata=pd.read_csv('weibo_train_data.txt',header=None,sep='\t')
    traindata.columns=['uid','mid','date','forward','comment','like','content']

    testdata=pd.read_csv('weibo_predict_data.txt',header=None,sep='\t')
    testdata.columns=['uid','mid','date','content']

    return traindata, testdata

#统计转赞评
def genUidStat():
    traindata, _ =loadData()
    train_stat = traindata[['uid','forward','comment','like']].groupby('uid').agg(['min','max','median','mean'])
    train_stat.columns = ['forward_min','forward_max','forward_median','forward_mean',
                          'comment_min','comment_max','comment_median','comment_mean',
                          'like_min','like_max','like_median','like_mean']
    train_stat = train_stat.apply(pd.Series.round)
    # 存储到字典,线性时间搜索
    stat_dic = {}
    for uid, stats in train_stat.iterrows():
        stat_dic[uid] = stats
    return stat_dic

def predict_with_stat(stat='median'):
    stat_dic = genUidStat()
    traindata, testdata = loadData()

    #获取每个uid的统计信息
    forward, comment, like = [], [], []
    for uid in traindata['uid']:
        if uid in stat_dic.keys():
            forward.append(int(stat_dic[uid]['forward_' + stat]))
            comment.append(int(stat_dic[uid]['comment_' + stat]))
            like.append(int(stat_dic[uid]['like_' + stat]))
        else:
            forward.append(0)
            comment.append(0)
            like.append(0)
    train_real_pred = traindata[['forward','comment','like']]
    train_real_pred['fp'], train_real_pred['cp'], train_real_pred['lp'] = forward, comment, like

    #对固定用户的测试数据进行预测
    test_pred = testdata[['uid','mid']]
    forward, comment, like = [], [], []
    for uid in testdata['uid']:
        if uid in stat_dic.keys():
            forward.append(int(stat_dic[uid]['forward_' + stat]))
            comment.append(int(stat_dic[uid]['comment_' + stat]))
            like.append(int(stat_dic[uid]['like_' + stat]))
        else:
            forward.append(0)
            comment.append(0)
            like.append(0)

    test_pred['fp'], test_pred['cp'], test_pred['lp'] = forward, comment, like

    result = []
    filename='weibo_predict_{}.txt'.format(stat)
    for _, row in test_pred.iterrows():
        result.append('{0}\t{1}\t{2},{3},{4}\n'.format(row[0], row[1], row[2], row[3], row[4]))
    f = open(filename, 'w')
    f.writelines(result)
    f.close()
if __name__ == '__main__':
    predict_with_stat(stat='median')