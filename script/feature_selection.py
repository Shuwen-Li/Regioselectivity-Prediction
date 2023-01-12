import numpy as np
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr

loo = LeaveOneOut()
def circle(des,lable,model,des_len,tem_des_sel):
    all_pearsr = []
    all_mae = []
    des_rank=[]
    all_r2=[]
    try_index=list(set(list(range(des_len)))-set(tem_des_sel))
    for tem_des_index in try_index:
        tem_des_sel_=tem_des_sel+[tem_des_index]
        tem_des=des[:,tem_des_sel_]
        desc=tem_des
        lable=lable                                            
        repeat_pred = []
        repeat_test = []
        for i in range(10):
            all_pred = []
            all_test = []
            for train_index_tep,test_index_tep in loo.split(desc):
                train_x,test_x = desc[train_index_tep],desc[test_index_tep]
                train_y,test_y = lable[train_index_tep],lable[test_index_tep]
                model.fit(train_x,train_y)
                test_pred = model.predict(test_x)
                all_pred.append(test_pred)
                all_test.append(test_y)
            all_pred = np.concatenate(all_pred)
            all_test = np.concatenate(all_test)
            repeat_pred.append(all_pred)
            repeat_test.append(all_test)
        mean_pred = np.mean(repeat_pred,axis=0)
        mean_test = np.mean(repeat_test,axis=0)
        
        r2 = r2_score(mean_test,mean_pred)                            
        pearsr=pearsonr(mean_test,mean_pred)[0]
        mae=mean_absolute_error(mean_test,mean_pred)
        all_r2.append([r2])
        all_pearsr.append([pearsr])
        all_mae.append(mae)
    all_r2=np.array(all_r2)    
    max_r2=all_r2[np.argmax(all_pearsr)]
        
    all_pearsr=np.array(all_pearsr)    
    max_pear=all_pearsr[np.argmax(all_pearsr)]
    
    all_mae=np.array(all_mae)    
    max_mae=all_mae[np.argmax(all_pearsr)]    
    tem_des_sel_max=tem_des_sel+[try_index[np.argmax(all_pearsr)]]
    return max_pear,max_r2,max_mae,tem_des_sel_max