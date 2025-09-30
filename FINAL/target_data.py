import numpy as np

target_joint_pos_1 = [4.622576563079859, 1.4919956782391877, -2.4435069071962348, 2.0936591207981596, 0.9990974803253598]
target_joint_pos_2 = [1.4387632277949232, 0.7054776193039478, -2.512352196859098, 1.3866665704382877, 1.6211613228469561]
target_joint_pos_3 = [4.708484383458099, 1.3752175535418365, -3.214261824370574, 1.7887614596312122, 1.7302036080836274]
target_joint_pos_4 = [4.054210505693871, 0.7992699204278928, -2.288749607899941, 1.2692020100313526, 2.645584347531603]

# The limitations of the joints are as follows:
target_joint_close_to_max = [165 * np.pi / 180, 85 * np.pi / 180, 145 * np.pi / 180, 97.5 * np.pi / 180, 162.5 * np.pi / 180]
target_joint_close_to_min = [-165 * np.pi / 180, -60 * np.pi / 180, -145 * np.pi / 180, -97.5 * np.pi / 180, -162.5 * np.pi / 180]
target_joint_pos_5 = [3.0, 1.0, -2.0, 1.0, 1.0]

# rows are target joint positions, columns are the joints
TARGET_JOINT_POSITIONS = np.array([target_joint_pos_1, target_joint_pos_2, target_joint_pos_3, target_joint_pos_4])
                                    
# # Test 1 for Task 1D
# TARGET_JOINT_POSITIONS = np.array([target_joint_pos_1, target_joint_pos_2, target_joint_pos_3, target_joint_pos_4, target_joint_pos_5])

# # Test 2 for Task 1D
# TARGET_JOINT_POSITIONS = np.array([target_joint_pos_1, target_joint_pos_2, target_joint_pos_3, target_joint_pos_4, \
#                                    target_joint_close_to_max, target_joint_close_to_min])
