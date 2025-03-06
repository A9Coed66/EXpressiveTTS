import os

def delete_files_with_underscore(directory):
    for filename in os.listdir(directory):
        if filename.startswith('_'):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
                    print(f"Deleted directory: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

# Example usage
directory_path_1 = './syn_samples'
directory_path_2 = './syn_samples_cp'
delete_files_with_underscore(directory_path_1)    
delete_files_with_underscore(directory_path_2)    