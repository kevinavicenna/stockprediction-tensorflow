import os

cwd = os.getcwd()
print("Current working directory:", cwd)

file_path = os.path.join(cwd, 'lstm_model2.keras')
print("File path:", file_path)
print("File exists:", os.path.exists(file_path))