import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Callable
import gymnasium
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs import make_env, StretchDrawer
from argparse import ArgumentParser
# import wandb
import time
import pickle

import moviepy.video.io.ImageSequenceClip

from PIL import Image, ImageDraw, ImageFont
import sys
from sb3_contrib import TQC
from envs.utils import get_obs_split
from model_structures.BC import BCNetwork
from model_structures.Discriminator import Discriminator
from train_modules.Trainer import BCD_trainer_drawer
# from reset_bank import ResetBank
import os
from collections import deque

import matplotlib.pyplot as plt
import moviepy.config as mp_config
from envs.utils import get_obs_split, plot_info_on_frame


def val_env(max_ep_length, model, discriminator, run_name, sim, device, num_ep=20, seq_len=10):
    f_name = f"videos/{run_name}"
    if os.path.exists(f_name):
        os.system(f"rm -rf {f_name}")
    #     os.Po
    os.makedirs(f_name, exist_ok=True)
    
    # inp = {'jnt_states': torch.tensor([ 5.1088e-01, -0.0000e+00,  2.5605e-04,  1.6227e-02, -1.6348e-05]), 'goal_pos': torch.tensor([0.0509, 0.2365, 0.1236]), 'gripper_pos': torch.tensor([0.0291, 0.4235, 0.6264])}
    # print(model.predict(inp, deterministic=True)[0])
    mean_ep_rew = 0
    num_suc = 0
    video_freq = 1#max(num_ep//10, 1)
    res = []
    grip_pos = [[],[],[]]
    save_dict = {}


    traj_obs = {o: deque(maxlen=seq_len) for o in sim.observation_space.keys()}
    traj_actions = deque(maxlen=seq_len)
    
    for i in range(num_ep):
        obs, info = sim.reset()
        save_dict = {}

        min_prob = 1
        prob = [min_prob, 0, 0]
        screens  = []
        suc = False
        ep_reward = 0

        action = np.zeros(4)
        for _ in range(seq_len):  # Fill with initial observations
            for o in sim.observation_space.keys():
                traj_obs[o].append(torch.tensor(obs[o], dtype=torch.float32, device=device))  # ✅ Convert to Tensor
            traj_actions.append(torch.tensor(action, dtype=torch.float32, device=device))  # ✅ Ensure consistent dtype & device
        
        for j in range(max_ep_length):
            for o in sim.observation_space.keys():
                traj_obs[o].append(torch.tensor(obs[o], dtype=torch.float32, device=device))  # ✅ Automatically removes old ones
            
            _, student_obs = get_obs_split(traj_obs)
            # teacher_obs, _ = get_obs_split(obs)
            
            obs_seq_batch = torch.cat(
                [torch.stack(list(v)) if isinstance(v, deque) else v for v in student_obs.values()],
                dim=-1
            ).unsqueeze(0)
                
            with torch.no_grad():
                action = model.predict(obs_seq_batch[:,-1,:], deterministic=True)
                # print("action size:", action.size())
                action = action[0]
                # teacher_obs_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in teacher_obs.items()}
                # action = model.predict(teacher_obs_tensor, deterministic=True)[0]    
            # save_input = deepcopy(obs)
            # save_input["action"] = action
            # save_input = {k: v.unsqueeze(0).clone() for k, v in save_input.items()}
            # if len(save_dict.keys()) == 0:
            #     save_dict = save_input
            # else:
            #     for k in save_dict.keys():
            #         save_dict[k] = np.vstack((save_dict[k], save_input[k]))

            obs, r, done, trunc, info = sim.step(action.cpu())#sim.eval_step(a, n_step=frame_skip, get_frames=True)
            
            traj_actions.append(torch.tensor(action, dtype=torch.float32, device=device))
            
            obs_seq_tensors = {
                o: torch.stack([torch.tensor(x, dtype=torch.float32, device=device) if isinstance(x, np.ndarray) else x
                                for x in student_obs[o]], dim=0)
                for o in student_obs.keys()
            }
            obs_seq_batch = torch.cat(list(obs_seq_tensors.values()), dim=-1).unsqueeze(0) 
            
            act_seq_batch = torch.stack(list(traj_actions)).unsqueeze(0)
            seq = torch.cat((obs_seq_batch, act_seq_batch), dim=-1).flatten()
            bc_prob = discriminator(seq)
            if bc_prob.item()<0.3:
                prob[2] += 1
            if min_prob > bc_prob.item():  
                min_prob = bc_prob.item()
                prob[0] = min_prob
                prob[1] = info["step"]
            info["prob"] = bc_prob.item()
            info["min_prob"] = prob
            screens+= [plot_info_on_frame(Image.fromarray(np.uint8(sim.render())), info, font_size=10)]#info["frames"]
            if done:
                suc = info["is_success"]
                break
        
            mean_ep_rew += r
            ep_reward += r
            # res.append(r)
            # for k in range(3):
            #     grip_pos[k].append(obs["goal_pos"][k])
        # print(i)
        # with open("obs.pkl", "wb") as f:
        #     pickle.dump(save_dict, f)
            # print(j)
        if info["goal"] == 0:
            GOAL = "bottom"
        elif info["goal"] == 1:
            GOAL = "middle"
        elif info["goal"] == 2:
            GOAL = "top"
        else:
            GOAL = None
        if i % video_freq == 0 or not suc:
            file_name = str(suc) + "_" + GOAL + "_" + str(obs["handle_0_status"])+ str(obs["handle_1_status"])+ str(obs["handle_2_status"])+str(i)
            mp_config.change_settings({"IMAGEMAGICK_BINARY": "auto"})  # Optional config
            mp_config.default_logging = "error"  # Suppress logs
            moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
                        screens, fps=int(1/(0.005*20))
                    ).write_videofile(f"videos/{run_name}/{file_name}.mp4")
        num_suc += suc
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID' or 'MJPG'
        # out = cv2.VideoWriter('sim_depth_images.mp4', fourcc, 5.0, (depth_images[0].shape[1], depth_images[0].shape[0]), isColor=False)

        # # Save rotated grayscale images as frames
        # for img in depth_images:
        #     # img_copy = np.uint8(img)
        #     out.write(img)

        # Release the VideoWriter and close all windows
        # out.release()

        # with open(f"./sim_inp_{i}.pickle", 'wb') as file:
        #     pickle.dump(save_dict, file)

    # sys.stdout = orig_stdout
    # f.close()

    nf_name = f"{f_name}_{num_suc/num_ep}"
    print(f"mean_ep_reward: {mean_ep_rew/num_ep}")
    print(f"mean_suc: {num_suc/num_ep}")
    if os.path.exists(f_name):
        os.rename(f_name, nf_name) 
         # 폴더 이름 변경
    else:
        print(f"폴더 '{f_name}'이(가) 존재하지 않습니다.")

    # fig, ax = plt.subplots(2,2)
    # l=['x', 'y', 'z']
    # for i in  range(3):
    #     curr_ax = ax[i//2, i%2]
    #     ax[i//2, i%2].plot(range(len(grip_pos[i])), np.abs(grip_pos[i]))
    #     curr_ax.set_title(f"Delta to goal pos in {l[i]} dir")
    #     curr_ax.set(xlabel="Frames", ylabel="Delta")

    # ax[1,1].plot(range(len(res)), -np.array(res))
    # ax[1,1].set(xlabel="Frames", ylabel="l2 norm to goal")
    # ax[1,1].set_title("l2 norm to goal pos")
    # fig.suptitle("Simulation trajectory")
    # fig.tight_layout()
    # plt.savefig("traj.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--teacher_path", type=str, default="saved_models/teacher_drawer.zip",)
    parser.add_argument("--student_path", type=str,default="saved_models")
    parser.add_argument("--trained_id", type=int,default="-1")
    parser.add_argument("--save_path", type=str, default="saved_models/drawer")
    parser.add_argument("--identifier", type=str, default="train_01")
    parser.add_argument("--buffer_path", type=str,default="./buffer/drawer/buffer.pkl")
    parser.add_argument("--new_buffer_path", type=str, default=None)
    parser.add_argument("--num_traj", type=int, default=20000)
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--eval_its", type=int, default=20000)
    parser.add_argument("--num_cpu", type=int, default=16)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--collect_data", action="store_true")
    parser.add_argument("--collect_more_data", action="store_true")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--cnt_threshold", type=int, default=10)
    parser.add_argument("--aug_traj_mul", type=int, default=10)
    args = parser.parse_args()
    args_dict = vars(args)
    model_sv_path = os.path.join(args.save_path, args.identifier)    
    if not os.path.exists(model_sv_path):
                os.makedirs(model_sv_path)
    # Save to text file
    with open(os.path.join(model_sv_path, "arguments.txt"), "w") as f:
        for key, value in args_dict.items():
            f.write(f"{key}: {value}\n")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env_id = "StretchDrawer"
    sim = StretchDrawer(max_episode_length=args.max_episode_steps, student_obs=True)
    
    # env = make_env(env_id, depth_rendering=True)()
    # env.close()
    
    teacher = TQC.load(args.teacher_path)
    # print(teacher.policy)
    # student = StudentDrawerPolicy()
    student = BCNetwork(device=device)
    discriminator = Discriminator(device=device, seq_len=args.seq_len)
    if args.student_path == "n":
        pretrain = False
    else:
        pretrain = True
        if args.trained_id == -1:
            student_path = os.path.join(args.student_path, "student_drawer.pth")
            discriminator_path = os.path.join(args.student_path, "discriminator_drawer.pth")
        else:
            student_path = os.path.join(args.student_path, f"student{args.trained_id}.pth")
            discriminator_path = os.path.join(args.student_path, f"discriminator{args.trained_id}.pth")
    
        student.load_model(student_path)
        discriminator.load_model(discriminator_path)
    # trainer = BCD_trainer_light(
    #     student,
    #     teacher,
    #     discriminator,
    #     env_id,
    #     device,
    #     # args.save_path,
    #     model_sv_path,
    #     num_cpu=args.num_cpu,
    #     max_episode_steps=args.max_episode_steps,
    #     seq_len=args.seq_len
    # )
    trainer = BCD_trainer_drawer(
        student,
        teacher,
        discriminator,
        env_id,
        device,
        # args.save_path,
        model_sv_path,
        num_cpu=args.num_cpu,
        max_episode_steps=args.max_episode_steps,
        seq_len=args.seq_len
    )
    
    if args.collect_data:
        if not args.collect_more_data:
            trainer.collect_data(args.num_traj)
            trainer.buffer.save(args.buffer_path)
        else:
            with open(args.buffer_path, "rb") as file:
                loaded_buffer = pickle.load(file)
            
            trainer.buffer = loaded_buffer
            print(len(trainer.buffer))
            trainer.collect_data(args.num_traj)
            trainer.buffer.save(args.new_buffer_path)
            print(len(trainer.buffer))

    print("Training BC")
    with open(args.buffer_path, "rb") as f:
        trainer.buffer = pickle.load(f)

    ## wandb
    # wandb.init(
    #     project="dagger_training",
    #     config={
    #         "teacher_path": args.teacher_path,
    #         "save_path": args.save_path,
    #         "training_epochs": args.num_epoch,
    #         "max_episode_steps": args.max_episode_steps,
    #         "num_cpu": args.num_cpu,
    #         "difficulty": args.difficulty
    #     },
    #     name=args.save_path
    # )   
    # print("BUFFER LENGHT:", len(trainer.buffer))
    if not pretrain:
        # trainer.train(num_epoch=1000)# for buffer_light
        trainer.train(num_epoch=args.num_epoch, sv_interval=10, model_sv_path=model_sv_path) # for buffer 
        # trainer.train(num_epoch=1000, sv_interval=10, model_sv_path=model_sv_path) # for buffer 
        
        trainer.save_models(model_sv_path)
    data_aug_iter = 100    
    num_ep =20
    with tqdm(total=data_aug_iter) as pbar:
        for aug_i in range(data_aug_iter):
            # bank = ResetBank()
            bank = []
            bank = trainer.augment_data(
                args.max_episode_steps,
                record_video=True,
                bank=bank,
                model=student,
                discriminator=discriminator,
                run_name=f"{args.identifier}/round_{aug_i}",
                sim=sim,
                device=device,
                num_ep=num_ep,
                seq_len=args.seq_len,
                aug_iter = aug_i,
                cnt_threshold=args.cnt_threshold
            )
            print(f"len bank:{len(bank)}")
            # with open(f"./reset_states/student_states_start_{aug_i}.pickle", 'wb') as file:
            sv_path = os.path.join(model_sv_path, "reset_states")
            if not os.path.exists(sv_path):
                os.makedirs(sv_path)
            with open(f"{sv_path}/student_states_start_{aug_i}.pickle", 'wb') as file:
                pickle.dump(bank, file)
            if len(bank) < num_ep:
                trainer.collect_data(int(len(bank)*args.aug_traj_mul), bank) # 20개의 failed state를 모으고, 각 state 에서 1000번씩 traj 모은다. 
                trainer.buffer.save(args.buffer_path, args.identifier)
            else:
                print("Failed for all turns. skip data augmentation, train with existing dataset.")
            trainer.train(num_epoch=args.num_epoch)
            trainer.save_models(model_sv_path, aug_i)
            val_env(args.max_episode_steps,student, discriminator, f"{args.identifier}/test{aug_i}", sim, device, num_ep=20, seq_len=args.seq_len)    
            del bank

    # wandb.finish()