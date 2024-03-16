from sklearn import feature_selection as fs
from sklearn import linear_model as lm
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)




def read_file():
    output_file_path = "./output.txt"
    c_file_path = ''
    output_stop_line = "dielectric:"
    features = []
    feature_names = []
    capacitance = []
    cap_names = []
    dielectrics = []
    die_names = []
    die_repeat_times = []
    t = ''
    with open(output_file_path, 'r') as file:
        lines = file.readlines()
        line0 = lines[0].strip().split(None)
        line1 = lines[1].strip().split(None)
        t = line0[0]
        c_file_path = '../data/' + t + ".txt"
        IsTargetName = False
        line_num = 1    

        for name in line1:
            if IsTargetName:
                feature_names.append(name)
            if name == '|':
                IsTargetName = True
        for line in lines[2:]:
            line_num += 1
            IsTargetData = False
            data = []
            values = line.strip().split(None)
            if output_stop_line in values:
                line_num += 1
                break
            for value in values:
                if IsTargetData:
                    data.append(float(value))
                if value == '|':
                    IsTargetData = True
            if len(data) != 0:
                features.append(data)
        # ��ȡdielectric����
        while 'file_end' not in lines[line_num]:
            begin_line = line_num
            one_die_data = []
            for line in lines[begin_line:]:
                values = line.strip().split(None)
                data = []
                if "END" in line:
                    line_num += 1
                    break

                if line_num == begin_line:
                    die_names.append(values[0])
                    die_repeat_times.append(int(values[1]))
                else:
                    for value in values:
                        data.append(float(value))
                    if len(data) != 0:
                        one_die_data.append(data)
                line_num += 1
            dielectrics.append(one_die_data)

    with open(c_file_path, 'r') as file:
        IsCName = False
        lines = file.readlines()
        line0 = lines[0].strip().split(None)
        for name in line0:
            if IsCName:
                cap_names.append(name)
            if name == '|':
                IsCName = True

        for line in lines[1:]:
            IsC = False
            data = []
            values = line.strip().split(None)
            for value in values:
                if IsC:
                    data.append(float(value))
                if value == '|':
                    IsC = True
            if len(data) != 0:
                capacitance.append(data)

    return features, feature_names, capacitance, cap_names, t, dielectrics, die_names, die_repeat_times


def processVariance(feature, fea_names, TYPE, title, threshold):
    selector = fs.VarianceThreshold(threshold=threshold)

    feature_selected = selector.fit_transform(feature)
    selected_feature_index = selector.get_support(indices=True)
    selected_names = [fea_names[i] for i in selected_feature_index]
    print(title, end=' ')
    print(selected_names)


    variance = list(np.var(feature, axis=0))
    var_name_dic = {}
    for i in range(len(fea_names)):
        var_name_dic[variance[i]] = fea_names[i]
    sorted_var = sorted(variance)  # ��������������
    sorted_names = []  # ��Ӧ����
    for var in sorted_var:
        sorted_names.append(var_name_dic[var])


    fig = plt.figure()
    plt.stem(sorted_var)
    plt.xticks(range(0, len(sorted_names)), sorted_names, rotation=45)
    plt.xlabel("Features")
    plt.ylabel("Variance")
    plt.title(title)
    plt.show()
    # ����ɸѡ������Ϣ
    return feature_selected


def processCoeff(feature, fea_names, capacitance, cap_names, TYPE, title, func, k, die_rep=0):
    if TYPE != title:
        print(TYPE)
    P = None
    selector = None
    if func == 'f_regression':
        selector = fs.SelectKBest(score_func=fs.f_regression, k=k)
        P = True
    if func == 'mutual_info_regression':
        selector = fs.SelectKBest(score_func=fs.mutual_info_regression, k=k)
        P = True
    if func == 'f_classif':
        selector = fs.SelectKBest(score_func=fs.f_classif, k=k)
        P = False
    feature_selected = []
    print(title)
    if die_rep != 0:
        if die_rep != 1:
            cap_temp = []
            for i in range(len(capacitance)):
                for j in range(die_rep):
                    cap_temp.append(capacitance[i])
            capacitance = cap_temp
    for i in range(len(capacitance[0])):
        c = []
        cname = cap_names[i]
        if 't' in cname:
            continue
        for data in capacitance:
            c.append(data[i])
        feature_selected_list = selector.fit_transform(feature, c)
        feature_selected.append(feature_selected_list)
        feature_selected_index = selector.get_support(indices=True)
        name_selected = [fea_names[j] for j in feature_selected_index]
        print(cname + ": ", end='')
        print(name_selected)


        if P:
            fig = plt.figure()
            plt.stem(selector.scores_)
            plt.xticks(range(0, len(fea_names)), fea_names, rotation=45)
            plt.xlabel('Features')
            plt.ylabel('Scores')
            plt.title(title+' & '+cname)
            plt.show()


    return feature_selected


def processREFCV(feature, fea_names, capacitance, cap_names, TYPE, title, die_rep=0):
    if TYPE != title:
        print(TYPE)
    estimator = lm.LinearRegression()
    selector = fs.RFECV(estimator=estimator)
    feature_selected = []
    print(title)
    if die_rep != 0:
        if die_rep != 1:
            cap_temp = []
            for i in range(len(capacitance)):
                for j in range(die_rep):
                    cap_temp.append(capacitance[i])
            capacitance = cap_temp
    for i in range(len(capacitance[0])):
        cname = cap_names[i]
        if 't' in cname:
            continue
        c = []
        for data in capacitance:
            c.append(data[i])
        feature_selected_list = selector.fit_transform(feature, c)
        feature_selected.append(feature_selected_list)
        feature_selected_index = selector.get_support(indices=True)
        name_selected = [fea_names[j] for j in feature_selected_index]
        print(cap_names[i], end=': ')
        print(name_selected)


def processL(feature, fea_names, capacitance, cap_names, TYPE, title, func, die_rep=0):
    if TYPE != title:
        print(TYPE)
    estimator = None
    if func == 'Linear':
        estimator = lm.LinearRegression()
    if func == 'GBR':
        estimator = GradientBoostingRegressor()
    selector = fs.SelectFromModel(estimator=estimator)
    feature_selected = []
    print(title)
    if die_rep != 0:
        if die_rep != 1:
            cap_temp = []
            for i in range(len(capacitance)):
                for j in range(die_rep):
                    cap_temp.append(capacitance[i])
            capacitance = cap_temp
    for i in range(len(capacitance[0])):
        cname = cap_names[i]
        if 't' in cname:
            continue
        c = []
        for data in capacitance:
            c.append(data[i])
        feature_selected_list = selector.fit_transform(feature, c)
        feature_selected.append(feature_selected_list)
        feature_selected_index = selector.get_support(indices=True)
        name_selected = [fea_names[j] for j in feature_selected_index]
        print(cap_names[i], end=': ')
        print(name_selected)


def processEFS(feature, fea_names, capacitance, cap_names, TYPE, title, die_rep=0):
    if TYPE != title:
        print(TYPE)
    x = pd.DataFrame(feature, columns=fea_names)
    estimator = lm.LinearRegression()
    selector = ExhaustiveFeatureSelector(estimator,
                                         min_features=1,
                                         max_features=2,
                                         scoring='r2',
                                         print_progress=True,
                                         cv=2)

    print(title)
    if die_rep != 0:
        if die_rep != 1:
            cap_temp = []
            for i in range(len(capacitance)):
                for j in range(die_rep):
                    cap_temp.append(capacitance[i])
            capacitance = cap_temp
    for i in range(len(capacitance[0])):
        cname = cap_names[i]
        if 't' in cname:
            continue
        c = []
        for data in capacitance:
            c.append(data[i])
        feature_selected = selector.fit(x, c)
        print(cap_names[i], end=': ')
        print(feature_selected.best_feature_names_)



def checkVariance(feature,
                  fea_names,
                  TYPE,
                  dielectric,
                  die_names,
                  die_fea_names,
                  threshold=0.1,
                  ana_die=False,
                  d_name = 'DIE1'
                  ):

    if not ana_die:
        processVariance(feature, fea_names, TYPE, TYPE, threshold)
    else:
        print(TYPE)
        if d_name == '':
            for i in range(len(die_names)):
                title = die_names[i]
                die_features = dielectric[i]
                processVariance(die_features, die_fea_names, TYPE, title, threshold)
        else:
            if d_name not in die_names:
                sys.exit("dielectric name error!")
            else:
                die_index = die_names.index(d_name)
                die_features = dielectric[die_index]
                processVariance(die_features, die_fea_names, TYPE, d_name, threshold)




def checkCoefficient(feature,
                     fea_names,
                     capacitance,
                     cap_names,
                     TYPE,
                     dielectric,
                     die_names,
                     die_fea_names,
                     die_repeat_times,
                     func='f_regression',
                     k = 3,
                     ana_die=False,
                     d_name='DIE1'):
    assert func == 'f_regression' or func == 'mutual_info_regression' or func == 'f_classif', "func parameter error!"

    if not ana_die:
        processCoeff(feature, fea_names, capacitance, cap_names, TYPE, TYPE, func, k)
    else:
        if d_name == '':
            for i in range(len(die_names)):
                title = die_names[i]
                die_features = dielectric[i]
                die_rep = die_repeat_times[i]
                processCoeff(die_features, die_fea_names, capacitance, cap_names, TYPE, title, func, k, die_rep)
        else:
            if d_name not in die_names:
                sys.exit("dielectric name error!")
            else:
                die_index = die_names.index(d_name)
                die_features = dielectric[die_index]
                die_rep = die_repeat_times[die_index]
                processCoeff(die_features, die_fea_names, capacitance, cap_names, TYPE, d_name, func, k, die_rep)



def checkREFCV(feature,
               fea_names,
               capacitance,
               cap_names,
               dielectric,
               die_names,
               die_fea_names,
               die_repeat_times,
               TYPE,
               ana_die=False,
               d_name='DIE1'):
    if not ana_die:
        processREFCV(feature, fea_names, capacitance, cap_names, TYPE, TYPE)
    else:
        if d_name == '':
            for i in range(len(die_names)):
                title = die_names[i]
                die_features = dielectric[i]
                die_rep = die_repeat_times[i]
                processREFCV(die_features, die_fea_names, capacitance, cap_names, TYPE, title, die_rep)
        else:
            if d_name not in die_names:
                sys.exit("dielectric name error!")
            else:
                die_index = die_names.index(d_name)
                die_features = dielectric[die_index]
                die_rep = die_repeat_times[die_index]
                processREFCV(die_features, die_fea_names, capacitance, cap_names, TYPE, d_name, die_rep)




def checkL(feature,
           fea_names,
           capacitance,
           cap_names,
           dielectric,
           die_names,
           die_fea_names,
           die_repeat_times,
           TYPE,
           func = 'Linear',
           ana_die=False,
           d_name='DIE1'):
    assert func == 'Linear' or func == 'GBR', 'func parameter error!'

    if not ana_die:
        processL(feature, fea_names, capacitance, cap_names, TYPE, TYPE, func)
    else:
        if d_name == '':
            for i in range(len(die_names)):
                title = die_names[i]
                die_features = dielectric[i]
                die_rep = die_repeat_times[i]
                processL(die_features, die_fea_names, capacitance, cap_names, TYPE, title, func, die_rep)
        else:
            if d_name not in die_names:
                sys.exit("dielectric name error!")
            else:
                die_index = die_names.index(d_name)
                die_features = dielectric[die_index]
                die_rep = die_repeat_times[die_index]
                processL(die_features, die_fea_names, capacitance, cap_names, TYPE, d_name, func, die_rep)



def checkEFS(feature,
             fea_names,
             capacitance,
             cap_names,
             dielectric,
             die_names,
             die_fea_names,
             die_repeat_times,
             TYPE,
             ana_die=False,
             d_name='DIE1'):
    if not ana_die:
        processEFS(feature, fea_names, capacitance, cap_names, TYPE, TYPE)
    else:
        if d_name == '':
            for i in range(len(die_names)):
                title = die_names[i]
                die_features = dielectric[i]    
                die_rep = die_repeat_times[i]
                processEFS(die_features, die_fea_names, capacitance, cap_names, TYPE, title, die_rep)
        else:
            if d_name not in die_names:
                sys.exit("dielectric name error!")
            else:
                die_index = die_names.index(d_name)
                die_features = dielectric[die_index]
                die_rep = die_repeat_times[die_index]
                processEFS(die_features, die_fea_names, capacitance, cap_names, TYPE, d_name, die_rep)




if __name__=="__main__":
    die_fea_names = ['x', 'y', 'width', 'thickness', 'Er']
    feature_metrix, feature_names, capacitance, cap_names, TYPE, dielectric, die_names, die_repeat_times = read_file()
    # checkVariance(feature_metrix, feature_names, TYPE, dielectric, die_names, die_fea_names, ana_die=True, d_name='')
    # checkCoefficient(feature_metrix, feature_names, capacitance, cap_names, TYPE, dielectric, die_names, die_fea_names, die_repeat_times, func='f_regression', ana_die=True, d_name='')   # func='f_regression' or 'mutual_info_regression' or 'f_classif'
    # checkREFCV(feature_metrix, feature_names, capacitance, cap_names, dielectric, die_names, die_fea_names, die_repeat_times, TYPE, ana_die=True)
    # checkL(feature_metrix, feature_names, capacitance, cap_names, dielectric, die_names, die_fea_names, die_repeat_times, TYPE, func='Linear', ana_die=True) # func='Linear' or 'GBC'
    # checkEFS(feature_metrix, feature_names, capacitance, cap_names, dielectric, die_names, die_fea_names, die_repeat_times, TYPE, ana_die=True, d_name='')

