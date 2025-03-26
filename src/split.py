import os
import shutil
import argparse
from get_data import get_data

def train_and_test(config_file):
    config = get_data(config_file)
    root_dir = config['raw_data']['data_src']
    dest = config['load_data'].get('preprocessed_data', 'default/path/to/preprocessed_data')
    num_classes = config['load_data']['num_classes']
    splitr = config['train']['split_ratio']

    class_names = [f'class_{i}' for i in range(num_classes)]  # Generate class names

    for k in range(num_classes):
        class_name = class_names[k]  # Get correct class name
        class_path = os.path.join(root_dir, class_name)

        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} does not exist!")
            continue  # Skip missing classes
        
        per = len(os.listdir(class_path))  # Get total images
        print(f"{k} -> {per} images")

        split_ratio = round((splitr / 100) * per)
        cnt = 0

        for j in os.listdir(class_path):
            src_path = os.path.join(class_path, j)

            if cnt < split_ratio:
                dest_path = os.path.join(dest, 'train', class_name)
            else:
                dest_path = os.path.join(dest, 'test', class_name)
            
            os.makedirs(dest_path, exist_ok=True)  # Ensure directory exists
            shutil.copy(src_path, dest_path)
            cnt += 1

        print(f"Done splitting {class_name}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    passed_args = args.parse_args()
    train_and_test(config_file=passed_args.config)
