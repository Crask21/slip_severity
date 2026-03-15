
import torch
from LSTM import LSTMModel
import torch.nn.functional as F
from dataloader import SequenceDataloader
from tqdm import tqdm


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

model = LSTMModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 1000

dataloader = SequenceDataloader("G:\\datasets\\tac2Slip\\severity\\salt", fingers = ["mf"])


for epoch in range(epochs):
    for sensors, target in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        sensors = sensors.cuda()   # (B,T,num_fingers, 51)
        target = target.cuda()     # (B,T,num_fingers, 7)

        # take only the middle finger. Remove the num_fingers dimension
        sensors = sensors[:, :, 0, :]  # (B,T,51)
        target = target[:, :, 0, :]      # (B,T,7)

        pred = model(sensors)

        loss = pose_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print("epoch", epoch, "loss", loss.item())