import torch
from LSTM import LSTMModel
import torch.nn.functional as F
from dataloader import SequenceDataloader
from tqdm import tqdm
import matplotlib.pyplot as plt


def quaternion_loss(q_pred, q_gt):
    dot = torch.sum(q_pred * q_gt, dim=-1)
    return torch.mean(1 - torch.abs(dot))

def pose_loss(pred, target):
    

    t_pred = pred[:, :3]
    q_pred = pred[:, 3:]

    # Get the last target pose in the sequence
    t_gt = target[:, -1, :3]
    q_gt = target[:, -1, 3:]
    
    l_trans = F.mse_loss(t_pred, t_gt)
    l_rot = quaternion_loss(q_pred, q_gt)

    return l_trans + l_rot

def plot_sensor_target_pair(sensors, target, sensor_idx=0, target_idx=0):
    """
    Plots one sensor and target pair across time.
    sensors: Tensor of shape (T, 51)
    target: Tensor of shape (T, 7)
    sensor_idx: index of sensor feature to plot
    target_idx: index of target feature to plot
    """
    import numpy as np
    sensors_np = sensors.cpu().numpy()
    target_np = target.cpu().numpy()

    #use the first sample in the batch
    sensors_np = sensors_np[0, :, 0, :]  # (T, 51)
    sensors_np_restructured = sensors_np.reshape(-1, 17, 3)  # (T, 5, 51)
    sensors_np_magnitude = np.linalg.norm(sensors_np_restructured, axis=-1)  # (T, 5, 17)
    target_np = target_np[0, :, 0, :]    # (T, 7)


    print(f"sensors_np shape: {sensors_np.shape}, target_np shape: {target_np.shape}, sensors_np_magnitude shape: {sensors_np_magnitude.shape}")

    time = range(sensors_np.shape[0])
    # Plot the magnitude of the first 17 sensors
    fig, axs = plt.subplots(4, 5, figsize=(15, 10))
    for j in range(17):
        ax = axs[j // 5, j % 5]
        ax.plot(time, sensors_np_magnitude[:, j])
        ax.set_title(f'Sensor {j} Magnitude')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Magnitude')
    #Add title to the figure
    fig.suptitle('Magnitude of First 17 Sensors Over Time', fontsize=16)
    fig.legend(['Slip'], loc='upper right')

    # For plots 17,18,19 plot the target x,y,z
    ax = axs[3, 2]
    ax.plot(time, target_np[:, 0], label='Target X')
    ax.plot(time, target_np[:, 1], label='Target Y')
    ax.plot(time, target_np[:, 2], label='Target Z')
    ax.set_title('Target velocity Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.legend(fontsize='small')

    ax = axs[3, 3]
    ax.plot(time, target_np[:, 3], label='Target QX')
    ax.plot(time, target_np[:, 4], label='Target QY')
    ax.plot(time, target_np[:, 5], label='Target QZ')
    ax.plot(time, target_np[:, 6], label='Target QW')
    ax.set_title('Target Quaternion Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Quaternion')
    ax.legend(fontsize='small')

    plt.tight_layout()
    plt.show()
    # plt.figure(figsize=(10, 5))
    # plt.plot(time, sensors_np[:, sensor_idx], label=f'Sensor[{sensor_idx}]')
    # plt.plot(time, target_np[:, target_idx], label=f'Target[{target_idx}]')
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.title('Sensor and Target Pair Across Time')
    # plt.show()

def train(model, dataloader, optimizer, epochs=1000):
    losses = []
    for epoch in range(epochs):
        for sensors, target in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):

            # print(f"Min and max sensor values: {sensors.min().item()}, {sensors.max().item()}")
            if torch.cuda.is_available():
                sensors = sensors.cuda()   # (B,T,num_fingers, 51)
                target = target.cuda()     # (B,T,num_fingers, 7)

            # Scaling
            sensors = sensors/1024


            if model.fc.out_features == 3: # only train on translation
                target = target[:, :, :, :3]  # (B,T,num_fingers, 3)

            # take only the middle finger. Remove the num_fingers dimension
            sensors = sensors[:, :, 0, :]  # (B,T,51)
            target = target[:, :, 0, :]      # (B,T,7)

            pred = model(sensors)
            # if epoch > 500:
            # print(pred[0], target[0, -1])  # print the first sample in the batch and the last target pose

            if pred.shape[-1] == 7:
                loss = pose_loss(pred, target)
            else:
                loss = F.mse_loss(pred, target[:, -1])  # only compare to the last target pose

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0:
            print("epoch", epoch, "loss", loss.item())  

    #plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    # Save the model
    save_path = "lstm_model_scaled.pth"
    torch.save(model.state_dict(), "lstm_model.pth")

def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        for sensors, target in dataloader:
            sensors = sensors.cuda()   # (B,T,num_fingers, 51)
            target = target.cuda()     # (B,T,num_fingers, 7)

            if model.fc.out_features == 3: # only train on translation
                target = target[:, :, :, :3]  # (B,T,num_fingers, 3)

            # take only the middle finger. Remove the num_fingers dimension
            sensors = sensors[:, :, 0, :]  # (B,T,51)
            target = target[:, :, 0, :]      # (B,T,7)

            pred = model(sensors)

            if pred.shape[-1] == 7:
                loss = pose_loss(pred, target)
            else:
                loss = F.mse_loss(pred, target[:, -1])  # only compare to the last target pose

            print("Evaluation loss", loss.item())  
            break

def plot_predictions(model, dataloader):
    model.eval()
    with torch.no_grad():
        for sensors, target in dataloader:
            sensors = sensors.cuda()   # (B,T,num_fingers, 51)
            target = target.cuda()     # (B,T,num_fingers, 7)

            if model.fc.out_features == 3: # only train on translation
                target = target[:, :, :, :3]  # (B,T,num_fingers, 3)

            # take only the middle finger. Remove the num_fingers dimension
            sensors = sensors[:, :, 0, :]  # (B,T,51)
            target = target[:, :, 0, :]      # (B,T,7)

            pred = model(sensors)

            # plot_sensor_target_pair(sensors.cpu(), target.cpu(), sensor_idx=0, target_idx=0)
            break


if __name__ == "__main__":

    model = LSTMModel(output_dim=3).cuda() if torch.cuda.is_available() else LSTMModel(output_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 1000
    classes = ["book", "book_bottom_finger_contact", "chopping-board", "linear_book", "linear_book_bottom_finger_contact", "salt", "wood"]
    
    dataloader = SequenceDataloader("G:\\datasets\\tac2Slip\\severity-03-15", classes=classes, fingers = ["ff"], batch_size=32, min_seq_len=20, max_seq_len=100)
    # Get one batch of data
    sensors, target = next(iter(dataloader))

    plot_sensor_target_pair(sensors, target, sensor_idx=0, target_idx=0)

    train(model, dataloader, optimizer, epochs=epochs)
    evaluate(model, dataloader)
    plot_predictions(model, dataloader)