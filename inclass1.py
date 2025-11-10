lambdavalues = [1e-1, 1, 10, 1e2, 1e3, 1e4]

train_osr2_list = []
test_osr2_list = []

for alpha in lambdavalues:
    ridge = Ridge(alpha=alpha, random_state=88)
    ridge.fit(X_train_lasso, y_train)
    y_pred_train = ridge.predict(X_train_lasso)
    y_pred_test = ridge.predict(X_test_lasso)
    train_osr2 = OSR2(y_train, y_train, y_pred_train)
    test_osr2 = OSR2(y_train, y_test, y_pred_test)
    train_osr2_list.append(train_osr2)
    test_osr2_list.append(test_osr2)
    

plt.plot(candidate_values, train_osr2_list, 'o-', label='Training OSR²')
plt.plot(candidate_values, test_osr2_list, 's--', label='Testing OSR²')

plt.xscale('log')
plt.xlabel('Value of lambda', fontsize=12)
plt.ylabel('OSR^2', fontsize=12)
plt.title('Ridge Regression: OSR^2 vs lamda', fontsize=14)
plt.legend()
plt.grid(True, which='both')
plt.show()