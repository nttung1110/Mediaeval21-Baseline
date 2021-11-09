import numpy as np 
import itertools
import torch
import pdb


# Be careful when changing the orders of name_component_to_id
ref_matrix_2020 = [[1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1, 1]]

# ['Backspin', 'Loop', 'Sidespin', 'Topspin', 'Hit', 'Flip', 'Push', 'Block']

ref_matrix_2021 = [[1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1, 1]]

# For ignoring exclusive label in 2021: Serve Backhand Loop, Serve Backhand Sidepin
list_exclusive_comb = [((0, 1), 1), # Serve Backhand Loop
                ((0, 2), 1)] # Serve Backhand Sidepin


def pair_two_component(p1, p3, is_2020):
    label_1 = list(range(len(p1)))
    label_3 = list(range(len(p3)))
    all_pair_combination_label = list(itertools.product(label_1, label_3))

    if is_2020:
        ref_matrix = ref_matrix_2020
    else:
        ref_matrix = ref_matrix_2021 

    predict_pairs = torch.stack((p3, p3, p3))
    mul_pred_ref = np.array(predict_pairs.cpu().data)*np.array(ref_matrix)
    res_pair_prob = []
    for pair_label in all_pair_combination_label:
        pair_label_1 = pair_label[0]
        pair_label_3 = pair_label[1]

        score_pair = mul_pred_ref[pair_label_1][pair_label_3]*p1[pair_label_1].item()
        res_pair_prob.append(score_pair)

    return res_pair_prob, all_pair_combination_label


def find_best_combination_hard(p1, p2, p3, is_2020):

    res_pair_prob, all_pair_combination_label = pair_two_component(p1, p3, is_2020)

    label_2 = list(range(len(p2)))

    # p = [p1, p2, p3] # represent score
    # label = [label_1, label_2, label_3] # represent label
    all_combinations_p = list(itertools.product(res_pair_prob, p2.cpu().data.tolist()))
    all_combinations_label = list(itertools.product(all_pair_combination_label, label_2))

    prod = np.prod(all_combinations_p, axis=1)

    if not is_2020:
        # exclude missing labels in 2021 data
        for idx in range(len(all_combinations_label)):
            comb = all_combinations_label[idx]
            for exclusive_comb in list_exclusive_comb:
                if comb[0][0] == exclusive_comb[0][0] and \
                        comb[0][1] == exclusive_comb[0][1] and \
                            comb[1] == exclusive_comb[1]:
                    prod[idx] = 0.0 # zero out the score 


    max_score, max_comb_idx = np.max(prod), np.argmax(prod)

    max_comb_label = all_combinations_label[max_comb_idx]

    return max_score, max_comb_label

def find_best_combination_soft(p1, p2, p3, is_2020):

    predicted_label_1 = torch.argmax(p1).item()
    predicted_label_2 = torch.argmax(p2).item()

    if is_2020:
        ref_matrix = ref_matrix_2020
    else:
        ref_matrix = ref_matrix_2021

    mul_pred_ref = np.array(p3.cpu().data)*np.array(ref_matrix[predicted_label_1])
    predicted_label_3 = np.argmax(mul_pred_ref)

    return 1.0, ((predicted_label_1, predicted_label_3), predicted_label_2)

if __name__ == "__main__":
    pass