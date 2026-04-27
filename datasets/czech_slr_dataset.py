import ast
import torch
import pandas as pd
from augmentations import *
from random import randrange
import torch.utils.data as torch_data
from normalization.body_normalization import BODY_IDENTIFIERS
from normalization.hand_normalization import HAND_IDENTIFIERS
from normalization.body_normalization import normalize_single_dict as normalize_single_body_dict
from normalization.hand_normalization import normalize_single_dict as normalize_single_hand_dict

ORI_HAND_IDENTIFIERS = HAND_IDENTIFIERS
HAND_IDENTIFIERS = [id + "_0" for id in HAND_IDENTIFIERS] + [id + "_1" for id in HAND_IDENTIFIERS]


def remove_data(df, num_remove, remove_from):
    length = len(ast.literal_eval(df["indexDIP_left_X"][0]))

    ori_identifiers, extr_identifier = [], []
    if remove_from == 'right_hand':
        ori_identifiers = ORI_HAND_IDENTIFIERS
        extr_identifier = ["_right_X", "_right_Y"]
    elif remove_from == 'left_hand':
        ori_identifiers = ORI_HAND_IDENTIFIERS
        extr_identifier = ["_left_X", "_left_Y"]
    elif remove_from == 'body_face':
        ori_identifiers = BODY_IDENTIFIERS
        extr_identifier = ["_X", "_Y"]

    for i in range(num_remove):
        ori_identifier = ori_identifiers[i]
        identifier_list = [ori_identifier + extr_identifier[0], ori_identifier + extr_identifier[1]]
        for identifier in identifier_list:
            df[identifier] = str([0 for _ in range(length)])

    return df


# return data   | [shape(N , 54=(21*2+12) ,2 )]
#        labels | shape = N
def load_dataset(file_location: str, num_remove=0, remove_from=None):

    # Load the datset csv file
    df = pd.read_csv(file_location, encoding="utf-8")

    if remove_from is not None:
        df = remove_data(df, num_remove, remove_from)

    # TO BE DELETED
    df.columns = [item.replace("_left_", "_0_").replace("_right_", "_1_") for item in list(df.columns)]
    if "neck_X" not in df.columns:
        df["neck_X"] = [0 for _ in range(df.shape[0])]
        df["neck_Y"] = [0 for _ in range(df.shape[0])]

    # TEMP
    labels = df["labels"].to_list()
    # labels = [label + 1 for label in df["labels"].to_list()]
    # print('##############')

    # print(len(labels))
    # print('##############')

    data = []

    for row_index, row in df.iterrows():
        current_row = np.empty(shape=(len(ast.literal_eval(row["leftEar_X"])), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2))
        for index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
            current_row[:, index, 0] = ast.literal_eval(row[identifier + "_X"])
            current_row[:, index, 1] = ast.literal_eval(row[identifier + "_Y"])

        data.append(current_row)
    
    # data.shape = 3200, 204, 21, 2 = hand
    return data, labels


def tensor_to_dictionary(landmarks_tensor: torch.Tensor) -> dict:

    data_array = landmarks_tensor.numpy()
    output = {}

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[identifier] = data_array[:, landmark_index]

    return output


def dictionary_to_tensor(landmarks_dict: dict, identifier_lst: list) -> torch.Tensor:
    '''
INPUT
    landmarks_dict: dict: Một dictionary.
        Keys: Tên của các điểm khớp (ví dụ: 'wrist_0', 'thumbCMC_0', ...).
        Values: Một danh sách các tọa độ [x, y] cho mỗi khung hình của video. 
        Ví dụ: 
            landmarks_dict = {
                                    'wrist_0': [
                                        [0.1, 0.2],   # frame 0
                                        [0.15, 0.25], # frame 1
                                        [0.2, 0.3]    # frame 2
                                    ],
                                    'thumb_0': [
                                        [0.5, 0.6],   # frame 0
                                        [0.55, 0.65], # frame 1
                                        [0.6, 0.7]    # frame 2
                                    ]
                            }




    identifier_lst: list: Một danh sách chứa tên của các điểm khớp theo một thứ tự cố định. 
        Thứ tự này rất quan trọng vì nó sẽ quyết định thứ tự của các điểm khớp trong Tensor đầu ra. 
        Ví dụ: 
            identifier_lst = ['wrist_0', 'thumb_0']

OUTPUT:
    tensor_output = dictionary_to_tensor(landmarks_dict, identifier_lst)
    print(tensor_output)
    print(tensor_output.shape)

    tensor([[[0.1000, 0.2000],
         [0.5000, 0.6000]],

        [[0.1500, 0.2500],
         [0.5500, 0.6500]],

        [[0.2000, 0.3000],
         [0.6000, 0.7000]]], dtype=torch.float64)

    torch.Size([3, 2, 2])

    '''

    sequence_len = len(next(iter(landmarks_dict.values()), None)) # cho N=(số frame)
    output = np.empty(shape=(sequence_len, len(identifier_lst), 2)) # shape=(N, len(identifier_lst), 2)
    for landmark_index, identifier in enumerate(identifier_lst):
        output[:, landmark_index, 0] = [frame[0] for frame in landmarks_dict[identifier]]
        output[:, landmark_index, 1] = [frame[1] for frame in landmarks_dict[identifier]]

    return torch.from_numpy(output)


def isolate_single_body_dit(row: dict):
    body_identifiers = BODY_IDENTIFIERS  # [0:6] on face [6:] on body

    body_dit = {key: row[key] for key in body_identifiers if key in row}
    # print(f'Keys of the extracted body dit: {list(body_dit.keys())} of length of {len(body_dit.keys())}')
    return body_dit, body_identifiers


def isolate_single_hand(row: dict, hand_index: int):
    hand_identifiers = []
    for identifier in HAND_IDENTIFIERS:
        if identifier[-1] == str(hand_index):
            hand_identifiers.append(identifier)

    hand_dit = {key: row[key] for key in hand_identifiers if key in row}
    # which_hand = "left" if hand_index == 0 else "right"
    # print(f'Keys of the extracted {which_hand} hand dit: {list(hand_dit.keys())} of length of {len(hand_dit.keys())}')
    return hand_dit, hand_identifiers
    '''
    input:
    row: dict 
        {
            'wrist_0_X': [0.1, 0.3, 0.4,........] // len=số frame
            'w
            .....
            // 108 ddieemr cuar cow theer
        }
    
    hand_index in [0,1] | 0=left - 1=hand

    OUTPUT:
    hand_dit: 
        {
            điểm của bàn tay trái hoặc phải [0.1, 0.2, 0.4 ....]
        }

    '''

class CzechSLRDataset(torch_data.Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: [np.ndarray]
    labels: [np.ndarray]

    def __init__(self, dataset_filename: str, num_labels=5, transform=None, augmentations=False,
                 augmentations_prob=0.5, normalize=True, num_remove=0, remove_from=None):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        """

        loaded_data = load_dataset(file_location=dataset_filename, num_remove=num_remove, remove_from=remove_from)
        data, labels = loaded_data[0], loaded_data[1]

        self.data = data
        self.labels = labels
        self.targets = list(labels)
        self.num_labels = num_labels
        self.transform = transform

        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.normalize = normalize

    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """

        depth_map = torch.from_numpy(np.copy(self.data[idx]))
        label = torch.Tensor([self.labels[idx]])

        depth_map = tensor_to_dictionary(depth_map)

        # Apply potential augmentations
        if self.augmentations and random.random() < self.augmentations_prob:

            selected_aug = randrange(5)

            if selected_aug == 0:
                depth_map = augment_rotate(depth_map, (-13, 13))

            if selected_aug == 1:
                depth_map = augment_shear(depth_map, "perspective", (0, 0.1))

            if selected_aug == 2:
                depth_map = augment_shear(depth_map, "squeeze", (0, 0.15))

            if selected_aug == 3:
                depth_map = augment_arm_joint_rotate(depth_map, 0.3, (-4, 4))

            if selected_aug == 4:
                depth_map = depth_map

        if self.normalize:
            depth_map = normalize_single_body_dict(depth_map)
            depth_map = normalize_single_hand_dict(depth_map)

        l_hand_depth_map, l_hand_identifiers = isolate_single_hand(depth_map, 0)
        r_hand_depth_map, r_hand_identifiers = isolate_single_hand(depth_map, 1)
        body_depth_map, body_identifiers = isolate_single_body_dit(depth_map)

        l_hand_depth_map = dictionary_to_tensor(l_hand_depth_map, l_hand_identifiers) - 0.5
        r_hand_depth_map = dictionary_to_tensor(r_hand_depth_map, r_hand_identifiers) - 0.5
        body_depth_map = dictionary_to_tensor(body_depth_map, body_identifiers) - 0.5

        if self.transform:
            l_hand_depth_map = self.transform(l_hand_depth_map)  # (B, 204, 21, 2)
            r_hand_depth_map = self.transform(r_hand_depth_map)  # (B, 204, 21, 2)
            body_depth_map = self.transform(body_depth_map)  # (B, 204, 12, 2)

        return l_hand_depth_map, r_hand_depth_map, body_depth_map, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    pass
