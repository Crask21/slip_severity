from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import os
import torch

def import_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        #skip header
        next(reader)
        data = []
        for row in reader:
            data.append([float(i) for i in row])
    data = np.array(data, dtype=np.float32)
    time = data[:, 0]
    data = data[:, 1:]
    return time, data

finger_LUT = {
        "th": 0,
        "ff": 1,
        "mf": 2,
        "rf": 3,
        "lf": 4
    }

class SequenceDataset(Dataset):
    def __init__(self, path):
        
        
        self.sensor = []  # List of (sensors, target) 
        self.target = []  # List of (sensors, target)
        for file_path in os.listdir(path):
            if file_path.endswith('.csv'):
                time, data = import_csv(os.path.join(path, file_path))
                sensor_data = data[:, :255]  # First 5*51 columns are sensor data
                pose_data = data[:, 255:]    # Last 5*7 columns are pose data

                velocity_data = np.diff(pose_data, axis=0) / np.diff(time[:, None], axis=0)
                sensor_data = sensor_data[1:]  # Remove the first row to match velocities

                # squeeze to (T, 5, 51) and (T, 5, 7)
                sensor_data = sensor_data.reshape(-1, 5, 51)
                velocity_data = velocity_data.reshape(-1, 5, 7)
                
                self.sensor.append(sensor_data)
                self.target.append(velocity_data)

    def __len__(self):
        return len(self.sensor)

    def __getitem__(self, idx):
        return self.sensor[idx], self.target[idx]


class SequenceDataloader(DataLoader):
    def __init__(self, data_path, fingers=["th", "ff", "mf", "rf", "lf"], batch_size=32, shuffle=True):
        self.dataset = SequenceDataset(data_path)
        self.fingers = [finger_LUT[ finger ] for finger in fingers]

        def collate_fn(batch):
            '''
            limit to selected fingers and random sequence of length seq_len
            '''
            import random
            sensors, targets = zip(*batch)
            batch_sensors = []
            batch_targets = []

            seq_len = random.randint(20, 50)
            smallest_seq_len = min(sensor.shape[0] for sensor in sensors)
            seq_len = min(seq_len, smallest_seq_len)
            for sensor, target in zip(sensors, targets):
                # limit to selected fingers and random sequence of length seq_len
                sensor = sensor[:, self.fingers]
                target = target[:, self.fingers]
                start = random.randint(0, sensor.shape[0] - seq_len)
                end = start + seq_len
                batch_sensors.append(sensor[start:end])
                batch_targets.append(target[start:end])
            
            batch_sensors = torch.tensor(np.stack(batch_sensors), dtype=torch.float32)
            batch_targets = torch.tensor(np.stack(batch_targets), dtype=torch.float32)

            return batch_sensors, batch_targets
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


if __name__ == "__main__":
    dataloader = SequenceDataloader("G:\\datasets\\tac2Slip\\severity\\salt", fingers=["th", "ff", "mf", "rf"], batch_size=4)
    for sensors, target in dataloader:
        print(sensors.shape)  # (B,T,5,51)
        print(target.shape)   # (B,T,5,7)