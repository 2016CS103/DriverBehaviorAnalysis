from sklearn.metrics import classification_report, confusion_matrix
#Confution Matrix and Classification Report
Y_pred = classifier.predict_generator(test_set, Num_of_test_Samples // No_of_feature_Detectors+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_set.classes, y_pred))
print('Classification Report')
target_names = ['C0', 'c1', 'c2', 'C3', 'c4', 'c5', 'C6', 'c7', 'c8', 'c9', 'c10']
print(classification_report(test_set.classes, y_pred, target_names=target_names))
