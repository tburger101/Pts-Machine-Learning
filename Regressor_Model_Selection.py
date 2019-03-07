import numpy as np, pandas as pd, json
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV

def main():
    #Loadigng the data you are dealing with. Replace with location of your CSV
    df=pd.read_csv('Enter File Location', delimiter=None)

    y_variable='pts_above_average'
    scorer='neg_mean_absolute_error'
    final_details=[]
    y = df[y_variable]
    x = df

#Deleting of extra columns not needed for training and testing
    del x[y_variable]
    del x['game_id']
    del x['date']
    del x['player']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    #Determining what columns to scale (skipping binary variables)
    continous_col=[]
    for name in x.columns:
        long = x[name].max()
        small = x[name].min()
        if long == 1 and small==0:
            continue
        else:
            continous_col.append(name)

    # Scaling the data
    scaler=StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled=scaler.transform(x_test)

    #setting up parameter grids we will use for our optimization puropses
    dec_pre_param_grid= {"n_estimators": [50,100,150], "max_depth":[1,3,5]}
    svr_pre_param_grid={'C': [1, 10, 100], 'kernel': ['linear', 'rbf']}
    knn_pre_param_grid={'n_neighbors': [5,10,20,30]}
    pre_grid={'decision': dec_pre_param_grid, 'svr': svr_pre_param_grid, 'knn':knn_pre_param_grid, 'gradient_decision':dec_pre_param_grid}

    dec_param_grid = {"min_samples_split": [2, 5, 10, 20], "n_estimators": [50,100,150], "max_depth": [2, 3,4,5], "min_samples_leaf": [2, 5, 10, 15], "max_leaf_nodes": [2, 5, 10, 15, 20]}
    svr_param_grid={'C': [1, 10, 100, 1000], 'kernel': ['linear', 'poly', 'rbf'], 'gamma':[0.001, 0.01, 0.1, 1]}
    knn_param_grid={'n_neighbors': [2,5,12,15,22,25,30]}
    final_grid={'decision': dec_param_grid, 'svr': svr_param_grid, 'knn':knn_param_grid, 'gradient_decision':dec_param_grid}

    machine_learning=[ [KNeighborsRegressor(), 'knn'], [LinearRegression(), 'lin'], [RandomForestRegressor(), 'decision'],   [GradientBoostingRegressor(),'gradient_decision']]

    for model_list in machine_learning:
        model_details={}
        model=model_list[0]
        type=model_list[1]


        if type=='decision' or type=='gradient_decision':
            #The first optimization starts by looking for what the best features are for the model
            beg_param_grid=pre_grid.get(type)
            final_param_grid=final_grid.get(type)

            clf_grid=GridSearchCV(estimator=model, param_grid=beg_param_grid, scoring=scorer, n_jobs=-1, cv=3)
            clf_grid.fit(x_train_scaled, y_train)
            clf_pre=clf_grid.best_estimator_

            feature_search = RFECV(estimator=clf_pre, cv=3, scoring=scorer, n_jobs=-1)
            feature_search.fit(x_train_scaled, y_train)


            input_names = []
            for x, y in zip(feature_search.support_, x_train.columns.values):
                if x == True:
                    input_names.append(y)

            #After we have figured out the best features we re-optimize our hyperparameters
            x_train_feature=feature_search.transform(x_train_scaled)
            x_test_feature=feature_search.transform(x_test_scaled)

            clf_grid_final = GridSearchCV(estimator=model, param_grid=final_param_grid, scoring=scorer, n_jobs=-1, cv=3)
            clf_grid_final.fit(x_train_feature, y_train)
            clf_final= clf_grid_final.best_estimator_
            final_train_avg_score=clf_grid_final.best_score_

            #testing our final tuned model
            predictions = clf_final.predict(x_test_feature)
            error = mean_absolute_error(y_test, predictions)


            #Dumping the dictionary to file to compare the results of models
            model_details['final_train_avg_scores']=final_train_avg_score
            model_details['features']=input_names
            model_details['final_test_score']=error
            model_details['type']=type
            final_details.append(model_details)
            file_name=type+".text"
            with open(file_name, 'w') as file:
                file.write(json.dumps(model_details))

        elif type=='lin':
            # The first optimization starts by looking for what the best features are for the model
            clf_pre = LinearRegression(normalize=True, fit_intercept=True)
            feature_search = RFECV(estimator=clf_pre, cv=3, scoring=scorer, n_jobs=-1)
            feature_search.fit(x_train_scaled, y_train)
            x_train_feature=feature_search.transform(x_train_scaled)
            x_test_feature=feature_search.transform(x_test_scaled)
            input_names = []

            for x, y in zip(feature_search.support_, x_train.columns.values):
                if x == True:
                    input_names.append(y)

            train_cross_val_scores=cross_val_score(clf_pre, x_train_feature, y_train, scoring=scorer, cv=3)
            final_train_avg_score=np.mean(train_cross_val_scores)

            # After we have figured out the best features we re-optimize our linear model
            clf_pre.fit(x_train_feature, y_train)
            predictions = clf_pre.predict(x_test_feature)
            error = mean_absolute_error(y_test, predictions)

            # Dumping the dictionary to file to compare the results of models
            model_details['final_train_avg_scores'] = final_train_avg_score
            model_details['features'] = input_names
            model_details['final_test_score'] = error
            model_details['type'] = type
            final_details.append(model_details)
            file_name = type + ".text"
            with open(file_name, 'w') as file:
                file.write(json.dumps(model_details))

        elif type=='knn':
            train_cross_val_scores = cross_val_score(model, x_train_scaled, y_train, scoring=scorer, cv=3)
            final_train_avg_score = np.mean(train_cross_val_scores)

            model.fit(x_train_scaled, y_train)
            predictions = model.predict(x_test_scaled)
            error = mean_absolute_error(y_test, predictions)

            model_details['final_train_avg_scores'] = final_train_avg_score
            model_details['features'] = []
            model_details['final_test_score'] = error
            model_details['type'] = type
            final_details.append(model_details)
            file_name = type + ".text"
            with open(file_name, 'w') as file:
                file.write(json.dumps(model_details))

if __name__ == '__main__':
    main()
