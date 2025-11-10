ridge = Ridge(random_state=88, max_iter=10000)
alpha_grid = {'alpha': np.logspace(-3, 3, 13)}
err_scorer = make_scorer(large_prediction_error_count, greater_is_better=False)
cv = KFold(n_splits=10, shuffle=True, random_state=1)
ridge_cv_err = GridSearchCV(ridge, alpha_grid, scoring=err_scorer, cv=cv, n_jobs=-1, verbose=0)
ridge_cv_err.fit(X_train_rr, y_train)
best_alpha_err = ridge_cv_err.best_params_['alpha']
best_mean_err = -ridge_cv_err.best_score_
ridge_best = Ridge(alpha=best_alpha_err, random_state=88, max_iter=10000).fit(X_train_rr, y_train)
y_pred_train_log = ridge_best.predict(X_train_rr)
y_pred_test_log = ridge_best.predict(X_test_rr)
train_err_cnt = large_prediction_error_count(y_train, y_pred_train_log, 2000)
test_err_cnt = large_prediction_error_count(y_test, y_pred_test_log, 2000)
print(best_alpha_err)
print(best_mean_err)
print(train_err_cnt)
print(test_err_cnt)