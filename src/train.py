import torch
from LSTM import LSTMModel
import torch.nn.functional as F
from dataloader import SequenceDataloader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


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

def train(model, dataloader, optimizer, epochs=1000, output_model_path="lstm_model.pth", show_plots=True):
    losses = []
    model.train()
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        for sensors, target in dataloader:
        # for sensors, target in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):

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
        # if epoch % 10 == 0:
        #     print("epoch", epoch, "loss", loss.item())  

    #plot losses
    if show_plots:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
    plt.show()

    # Save the model
    save_path = "lstm_model_scaled.pth"
    torch.save(model.state_dict(), output_model_path)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
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
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print("Average evaluation loss", avg_loss)
    return avg_loss

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

    epochs = 5000
    classes = ["book", "book_bottom_finger_contact", "chopping-board", "linear_book", "linear_book_bottom_finger_contact", "salt", "wood"]
    
    dataloader = SequenceDataloader("G:\\datasets\\tac2Slip\\severity-03-15", classes=classes, fingers = ["ff"], batch_size=32, min_seq_len=5, max_seq_len=20)
    # # Get one batch of data
    # sensors, target = next(iter(dataloader))
    # plot_sensor_target_pair(sensors, target, sensor_idx=0, target_idx=0)

    # output_model_path = "lstm_model_5-20.pth"
    # train(model, dataloader, optimizer, epochs=epochs, output_model_path=output_model_path)
    # evaluate(model, dataloader)
    # plot_predictions(model, dataloader)

    # ------------------------------ parameter sweep ----------------------------- #
    dataloader = SequenceDataloader("G:\\datasets\\tac2Slip\\severity-03-15", classes=classes, fingers = ["ff"], batch_size=32, min_seq_len=1, max_seq_len=20)
    epochs = 1000
    min_seq_len_list = [1,2,4,8,16,24,32]
    max_seq_len_list = [4,5,8,10,15,20,30,40,80]
    loss_dict = {}
    loss_dict2 = {(1, 5): 0.01602193315259435, (1, 10): 0.00183819556100802, (1, 20): 0.0032020777262831953, (1, 40): 0.004137220242145387, (1, 80): 0.002144014155915515, (2, 5): 0.005716218401423909, (2, 10): 0.005209184336391362, (2, 20): 0.0015702461823821068, (2, 40): 0.0021224190925501966, (2, 80): 0.0017859623344107108, (4, 5): 0.0023999367946420202, (4, 10): 0.002662712316536768, (4, 20): 0.0012993481518192725, (4, 40): 0.0016759456008334052, (4, 80): 0.002587770408188755, (16, 20): 0.00420660614459352, (16, 40): 0.0028542684687470846, (16, 80): 0.004215445094318552, (32, 40): 0.0035798147913407197, (32, 80): 0.003839397400786931}
    loss_std_dict = {}
    for min_seq_len in min_seq_len_list:
        for max_seq_len in max_seq_len_list:
            loss_list = []
            for _ in range(5):
                # if (max_seq_len, min_seq_len) in loss_dict2:
                #     print(f"Already have loss for min_seq_len={min_seq_len} and max_seq_len={max_seq_len}, skipping training")
                #     loss_dict[(min_seq_len, max_seq_len)] = loss_dict2[(max_seq_len, min_seq_len)]
                #     continue
                if max_seq_len < min_seq_len:
                    continue
                dataloader.min_seq_len = min_seq_len
                dataloader.max_seq_len = max_seq_len
                print(f"Training with min_seq_len={min_seq_len} and max_seq_len={max_seq_len}")
                output_model_path = f"lstm_model_{min_seq_len}-{max_seq_len}.pth"
                train(model, dataloader, optimizer, epochs=epochs, output_model_path=output_model_path, show_plots=False)
                loss = evaluate(model, dataloader)
                loss_list.append(loss)
            loss_dict[(min_seq_len, max_seq_len)] = loss_list
            loss_std_dict[(min_seq_len, max_seq_len)] = np.std(loss_list)
    print(f"{loss_dict=}")
    print(f"{loss_std_dict=}")
    #Plot the losses for different sequence lengths
    plt.figure(figsize=(10, 5))
    for min_seq_len in min_seq_len_list:
        losses = [loss_dict[(min_seq_len, max_seq_len)] if (min_seq_len, max_seq_len) in loss_dict else None for max_seq_len in max_seq_len_list]
        plt.plot(max_seq_len_list, losses, 'o-', label=f'Min Seq Len={min_seq_len}')
    plt.xlabel('Max Sequence Length')
    plt.ylabel('Loss')
    plt.title('Evaluation Loss vs Max Sequence Length')
    plt.legend()
    plt.show()


loss_dict={(1, 4): [0.05382377112453634, 0.06012919274243442, 0.025783700953153046, 0.0209622879258611, 0.015853937334296377], (1, 5): [0.012402488299730148, 0.011037449775771662, 0.007290823131122373, 0.015491691417992115, 0.012226154265755957], (1, 8): [0.012564135461368343, 0.006868350107900121, 0.009443033952265978, 0.007628244093873284, 0.010832938365638256], (1, 10): [0.007551602553576231, 0.008623278403485363, 0.006958869201215831, 0.00854020273651589, 0.009341414213519205], (1, 15): [0.01100219769234007, 0.011112279935993931, 0.009872523305768316, 0.011805106039074335, 0.012424229644238949], (1, 20): [0.012412223202938383, 0.009516702372242104, 0.007381935476918112, 0.00576931991699067, 0.007450251924720677], (1, 30): [0.008451578630642458, 0.007481678473678502, 0.00882592306218364, 0.005517782071943988, 0.005595925623889674], (1, 40): [0.0056993966414169836, 0.006791027918965979, 0.005861435530029915, 0.00767862508920106, 0.007292749635367231], (1, 80): [0.008176555052738298, 0.009426322553984144, 0.01073530087755485, 0.0089509209448641, 0.007983408063988794], (2, 4): [0.007089019202711907, 0.008903042286295782, 0.007306581024419178, 0.007879220287908207, 0.00981148886917667], (2, 5): [0.01087638409808278, 0.013015159659764984, 0.012986686842685396, 0.012950779023495588, 0.011481558438390493], (2, 8): [0.01311340851878578, 0.013499305414205248, 0.017387617379426956, 0.01478545189919797, 0.016206375661898743], (2, 10): [0.015607457692650232, 0.02092566823756153, 0.01598270863971927, 0.013577417174184864, 0.022730124352330513], (2, 15): [0.01471711898391897, 0.01896878374232487, 0.0227771672335538, 0.022504963404075665, 0.02278446617790244], (2, 20): [0.026845024559985508, 0.029023799706589092, 0.0275676478208466, 0.03630321218886159, 0.03376829200847582], (2, 30): [0.02997898208824071, 0.04296846687793732, 0.03991942768069831, 0.02913442931391976, 0.03295767730609937], (2, 40): [0.03429179012098096, 0.028947828676212917, 0.02897160237824375, 0.027580468153411693, 0.02291209547018463], (2, 80): [0.030619289895350284, 0.02516479790210724, 0.026011911644177002, 0.02582139242440462, 0.02722257189452648], (4, 4): [0.02705547357486053, 0.02928874133662744, 0.03263700871982358, 0.03365945274179632, 0.04546688361601396], (4, 5): [0.0396345950324427, 0.049931886182589966, 0.04580311900512739, 0.05244122411717068, 0.048296387900005684], (4, 8): [0.06425830586390062, 0.061418403278697624, 0.0666001784530553, 0.07769820805300366, 0.07423225722529671], (4, 10): [0.07782179354266687, 0.08356583660299127, 0.08102038875222206, 0.08691102063113992, 0.07944614135406235], (4, 15): [0.08319417929107492, 0.08719175139611418, 0.08278767222707922, 0.06354746798222716, 0.07323481007055803], (4, 20): [0.079818571833047, 0.0798405185341835, 0.08688531206412749, 0.0756061930548061, 0.08238055049018427], (4, 30): [0.09462146596475081, 0.08320696076208894, 0.08812135322527452, 0.10322876680981029, 0.10324404727328908], (4, 40): [0.10227871483022516, 0.10956845703450116, 0.11284229227087715, 0.10311779244379564, 0.11367839100685986], (4, 80): [0.06828419830311429, 0.09823434258049185, 0.07886159894141284, 0.06921468641270291, 0.08736831017515877], (8, 4): [], (8, 5): [], (8, 8): [0.06462949785319241, 0.060453558510000054, 0.07440475509925322, 0.07904491099444302, 0.07932049510153857], (8, 10): [0.07808931612155655, 0.07946635681119832, 0.0811424312943762, 0.08593812449411913, 0.08287334476004947], (8, 15): [0.08536159145561131, 0.08660862425511534, 0.08312218128280206, 0.09374430267648264, 0.09080792760307138], (8, 20): [0.0970741838894107, 0.09612636539069089, 0.10441661355170337, 0.10154559260064905, 0.10116628015583212], (8, 30): [0.11446657099507072, 0.10858175564895976, 0.10667708583853462, 0.10381069169803099, 0.1152138655835932], (8, 40): [0.09711303765123541, 0.0823600021275607, 0.08306659995154901, 0.09470832957462831, 0.08201982995325868], (8, 80): [0.09294790402054787, 0.08242522688074545, 0.09152353351766412, 0.09103055366060951, 0.0973639650778337], (16, 4): [], (16, 5): [], (16, 8): [], (16, 10): [], (16, 15): [], (16, 20): [0.08471590415997939, 0.08641325750134209, 0.08907881785522807, 0.08521410789002072, 0.08590896224433725], (16, 30): [0.07945173233747482, 0.0898131670599634, 0.081165650351481, 0.09320479021830992, 0.08399601856415922], (16, 40): [0.07281260361725633, 0.05930510874498974, 0.06718700717795979, 0.0671413628892465, 0.06739962338046594], (16, 80): [0.0685053061355244, 0.07755559987642548, 0.08261467516422272, 0.0815064778382128, 0.07299261235378006], (24, 4): [], (24, 5): [], (24, 8): [], (24, 10): [], (24, 15): [], (24, 20): [], (24, 30): [0.07817047089338303, 0.07647502998059447, 0.0742047679695216, 0.07086151296442206, 0.07563270865516229], (24, 40): [0.08817941695451736, 0.0721738263964653, 0.08553533425385301, 0.08500507812608372, 0.08439896323464134], (24, 80): [0.08474667878313498, 0.09359465403990312, 0.08625890856439417, 0.09479928829453209, 0.07894961671395735], (32, 4): [], (32, 5): [], (32, 8): [], (32, 10): [], (32, 15): [], (32, 20): [], (32, 30): [], (32, 40): [0.08512841029600664, 0.08662752129814842, 0.08502810956402258, 0.08246044780720364, 0.08377093483101238], (32, 80): [0.08275932920250026, 0.08238381486047398, 0.08344121751460162, 0.07802903381260959, 0.07304305380040949]}
loss_std_dict={(1, 4): np.float64(0.01807704421475727), (1, 5): np.float64(0.002646754130713055), (1, 8): np.float64(0.0020781055806147336), (1, 10): np.float64(0.0008436239202665605), (1, 15): np.float64(0.0008561011821302187), (1, 20): np.float64(0.002286816209764041), (1, 30): np.float64(0.0013920558823419362), (1, 40): np.float64(0.0007765943011444161), (1, 80): np.float64(0.0009891508972955185), (2, 4): np.float64(0.0010224008583698068), (2, 5): np.float64(0.0009050809279241876), (2, 8): np.float64(0.0016132817011230165), (2, 10): np.float64(0.0034643811637468714), (2, 15): np.float64(0.003165398529921432), (2, 20): np.float64(0.0036957408783477434), (2, 30): np.float64(0.005504244161473197), (2, 40): np.float64(0.003633997862802182), (2, 80): np.float64(0.0019430427906510512), (4, 4): np.float64(0.00637446903124998), (4, 5): np.float64(0.004366234733326334), (4, 8): np.float64(0.00614219987779108), (4, 10): np.float64(0.0032001329233877786), (4, 15): np.float64(0.008555371705312809), (4, 20): np.float64(0.003696814198043431), (4, 30): np.float64(0.008011031872202111), (4, 40): np.float64(0.004780786571226388), (4, 80): np.float64(0.011321584860895127), (8, 4): np.float64(nan), (8, 5): np.float64(nan), (8, 8): np.float64(0.007690593534265835), (8, 10): np.float64(0.0027378579598738005), (8, 15): np.float64(0.003835269332303314), (8, 20): np.float64(0.003059389915323256), (8, 30): np.float64(0.004431316925706031), (8, 40): np.float64(0.006631023498021043), (8, 80): np.float64(0.004859612720699199), (16, 4): np.float64(nan), (16, 5): np.float64(nan), (16, 8): np.float64(nan), (16, 10): np.float64(nan), (16, 15): np.float64(nan), (16, 20): np.float64(0.0015212086590137815), (16, 30): np.float64(0.005207568341079779), (16, 40): np.float64(0.0043115168379493345), (16, 80): np.float64(0.005285720870498614), (24, 4): np.float64(nan), (24, 5): np.float64(nan), (24, 8): np.float64(nan), (24, 10): np.float64(nan), (24, 15): np.float64(nan), (24, 20): np.float64(nan), (24, 30): np.float64(0.0024642911176187424), (24, 40): np.float64(0.005593221360059876), (24, 80): np.float64(0.005873846228247211), (32, 4): np.float64(nan), (32, 5): np.float64(nan), (32, 8): np.float64(nan), (32, 10): np.float64(nan), (32, 15): np.float64(nan), (32, 20): np.float64(nan), (32, 30): np.float64(nan), (32, 40): np.float64(0.0014027346802305864), (32, 80): np.float64(0.003934428843522092)}