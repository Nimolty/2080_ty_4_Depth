# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 23:58:59 2023

@author: lenovo
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import glob
import os

def process_auc(auc_list):
    assert len(auc_list) == 5
    auc = float(auc_list[0].replace(" ADD AUC: ", ""))
    auc_mean = float(auc_list[2].replace(" ADD  Mean: ", ""))
    auc_median = float(auc_list[3].replace(" ADD  Median: ", ""))
    return [auc, auc_mean, auc_median]

def process_acc_meter(acc_list):
    assert len(acc_list) == 18
    all_acc_list = []
    for i in range(1, 18, 2):
        #print(i)
        this_acc = acc_list[i].replace(" acc: ", "")
        this_ccc = this_acc.replace(" ", "")
        this_acc = this_acc.replace("%", "")
        all_acc_list.append(float(this_acc))
    return all_acc_list

def process_acc_angle(acc_list):
    assert len(acc_list) == 22
    all_acc_list = []
    for i in range(1, 22, 2):
        #print(i)
        this_acc = acc_list[i].replace(" acc: ", "")
        this_ccc = this_acc.replace(" ", "")
        this_acc = this_acc.replace("%", "")
        all_acc_list.append(float(this_acc))
    return all_acc_list

def collect_txt_info(txt_path):
    info = {}
    info["ass_add"] = []
    info["ass_acc"] = []
    info["ori_add"] = []
    info["ori_acc"] = []
    info["angles_acc"] = []
    
    with open(txt_path,'r',encoding='utf8') as f:
        m = f.readlines()
        m = [j.replace('\n', '') for j in m if j != "\n"]
        
        
        #print(m[7:12])
        info["ass_add"] += process_auc(m[2:7])
        info["ori_add"] += process_auc(m[7:12])
        info["ass_acc"] += process_acc_meter(m[12:30])
        info["ori_acc"] += process_acc_meter(m[30:48])
        info["angles_acc"] += process_acc_angle(m[48:70])
        print("info", info)
    return info
    
    
if __name__ == "__main__":
    root_list = [f"./20_NUMERICAL_RESULTS",
                 f"./26_NUMERICAL_RESULTS",
                 f"./27_NUMERICAL_RESULTS"]
    fig = plt.figure(figsize=(12,20))
    for root_dir in root_list:
        info = {}
        info["ass_add"] = []
        info["ass_acc"] = []
        info["ori_add"] = []
        info["ori_acc"] = []
        info["angles_acc"] = []
        json_path_list = glob.glob(os.path.join(root_dir, '*_0.txt'))
        json_path_list.sort()
        for txt_path in json_path_list:
            print(txt_path)
            this_info = collect_txt_info(txt_path)
            for key in info.keys():
                info[key].append(this_info[key])
        
        
    #     ass_add_dict = {"0" : "ASS_AUC", "1" : "ASS_MEAN", "2" : "ASS_MEDIAN"}
    #     ass_add_info_np = np.array(info["ass_add"], dtype=np.float32)
    #     N, M = ass_add_info_np.shape
    #     legend_name = root_dir[2:4]
    #     for i in range(M):
    #         plt.subplot(M//2+1, 2, i+1)
    #         plt.plot(np.arange(5, 5 * N + 1, 5), ass_add_info_np[:, i],label=legend_name)
    #         plt.legend()
    #         plt.title(ass_add_dict[str(i)])
    # plt.savefig(f"./ass_auc.png")
        
        
    #     ass_add_dict = {"0" : "ORI_AUC", "1" : "ORI_MEAN", "2" : "ORI_MEDIAN"}
    #     ass_add_info_np = np.array(info["ori_add"], dtype=np.float32)
    #     N, M = ass_add_info_np.shape
    #     legend_name = root_dir[2:4]
    #     for i in range(M):
    #         plt.subplot(M//2+1, 2, i+1)
    #         plt.plot(np.arange(5, 5 * N + 1, 5), ass_add_info_np[:, i],label=legend_name)
    #         plt.legend()
    #         plt.title(ass_add_dict[str(i)])
    # plt.savefig(f"./ori_auc.png")
    
    #     ass_acc_dict = {"0" : "2cm", 
    #                     "1" : "3cm", 
    #                     "2" : "4cm",
    #                     "3" : "5cm",
    #                     "4" : "6cm",
    #                     "5" : "7cm",
    #                     "6" : "8cm",
    #                     "7" : "9cm",
    #                     "8" : "10cm",
    #                     }
    #     ass_add_info_np = np.array(info["ass_acc"], dtype=np.float32)
    #     print(ass_add_info_np.shape)
    #     N, M = ass_add_info_np.shape
    #     legend_name = root_dir[2:4]
    #     for i in range(M):
    #         plt.subplot(M//2+1, 2, i+1)
    #         plt.plot(np.arange(5, 5 * N + 1, 5), ass_add_info_np[:, i],label=legend_name)
    #         plt.legend()
    #         plt.title(ass_acc_dict[str(i)])
    # plt.savefig(f"./ass_acc.png")
    
    #     ori_acc_dict = {"0" : "2cm", 
    #                     "1" : "3cm", 
    #                     "2" : "4cm",
    #                     "3" : "5cm",
    #                     "4" : "6cm",
    #                     "5" : "7cm",
    #                     "6" : "8cm",
    #                     "7" : "9cm",
    #                     "8" : "10cm",
    #                     }
    #     ass_add_info_np = np.array(info["ori_acc"], dtype=np.float32)
    #     print(ass_add_info_np.shape)
    #     N, M = ass_add_info_np.shape
    #     legend_name = root_dir[2:4]
    #     for i in range(M):
    #         plt.subplot(M//2+1, 2, i+1)
    #         plt.plot(np.arange(5, 5 * N + 1, 5), ass_add_info_np[:, i],label=legend_name)
    #         plt.legend()
    #         plt.title(ori_acc_dict[str(i)])
    # plt.savefig(f"./ori_acc.png")
    
        angles_acc_dict = {"0" : "2.5 degree", 
                        "1" : "5.0 degree", 
                        "2" : "7.5 degree",
                        "3" : "10.0 degree",
                        "4" : "12.5 degree",
                        "5" : "15.0 degree",
                        "6" : "17.5 degree",
                        "7" : "20.0 degree",
                        "8" : "22.5 degree",
                        "9" : "25.0 degree",
                        "10" : "27.5 degree",
                        }
        angles_add_info_np = np.array(info["angles_acc"], dtype=np.float32)
        print(angles_add_info_np.shape)
        N, M = angles_add_info_np.shape
        legend_name = root_dir[2:4]
        for i in range(M):
            plt.subplot(M//2+1, 2, i+1)
            plt.plot(np.arange(5, 5 * N + 1, 5), angles_add_info_np[:, i],label=legend_name)
            plt.legend()
            plt.title(angles_acc_dict[str(i)])
    plt.savefig(f"./angles_acc.png")


