from typing import Mapping
import os
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from pathlib import Path
import save_svg
from losses import SDSLoss, ToneLoss, ConformalLoss, CLIPLoss, TrOCRLoss, FontClassLoss, IFLoss
from config import set_config
from utils import (
    check_and_create_dir,
    get_data_augs,
    save_image,
    preprocess,
    alignment,
    learning_rate_decay,
    combine_word,
    replicate_shapes,
    scale_into_wordsize,
    stack_all_svgs,
    rotate_svg,
    create_video)
import wandb
import warnings
import torchvision.transforms as tvt
import kornia.filters as KF
import shutil
import re
warnings.filterwarnings("ignore")
import glob
import pickle
render = pydiffvg.RenderFunction.apply
from PIL import Image

def generate_template_svg(template_word="OOO", font="IndieFlower-Regular", template_name="/scratch/gilbreth/zhao969/Word-As-Image/merge/template.svg"):
    preprocess(font, template_word, template_word, 1)
    file_path = f"/scratch/gilbreth/zhao969/Word-As-Image/code/data/init/{font}_{template_word}_scaled.svg"
    shutil.copy(file_path, template_name)
    return template_name

def generate_all_template_svg(font="IndieFlower-Regular", template_folder="/scratch/gilbreth/zhao969/Word-As-Image/merge/{temp_word}.svg"):
    for template_word in ["OOO", "OOOO", "OOOOO", "OOOOOO", "OOOOOOO", "OOOOOOOO", "OOOOOOOOO"]:
        preprocess(font, template_word, template_word, 1)
        file_path = f"/scratch/gilbreth/zhao969/Word-As-Image/code/data/init/{font}_{template_word}_scaled.svg"
        shutil.copy(file_path, template_folder.format(temp_word=template_word))

def read_choice(choice_file):
    files = []
    with open(choice_file, 'r') as f:
        next(f)
        for line in f:
            files.append(line.strip())
    return files

def copy_chosen_svg(files, experiment, word):
    result_svgs = []
    result_weight = []
    output_folder = f"/scratch/gilbreth/zhao969/Word-As-Image/output/{experiment}"
    for file in files:
        result = re.search("(.*)-(([a-zA-Z]_to_[a-zA-Z])_(.*)_(\d+\.\d+))-(.*[Ee]?[-+]?.*).png", file)
        font = result.group(1)
        folder = result.group(2)
        character_conversion = result.group(3)
        svg_result_path = f"{output_folder}/{font}/{folder}/output-svg/output.svg"
        shutil.copy(svg_result_path, f"/scratch/gilbreth/zhao969/Word-As-Image/merge/{word}/{character_conversion}_{result.group(5)}.svg")
        result_svgs.append(character_conversion)
        result_weight.append(f"{result.group(5)}")
    return result_svgs, result_weight

def copy_chosen_png(files, experiment, word):
    result_pngs = []
    output_folder = f"/scratch/gilbreth/zhao969/Word-As-Image/output/{experiment}"
    for file in files:
        result = re.search("(.*)-(([a-zA-Z]_to_[a-zA-Z])_(.*)_(\d+\.\d+))-(.*[Ee]?[-+]?.*).png", file)
        font = result.group(1)
        folder = result.group(2)
        character_conversion = result.group(3)
        png_result_path = f"{output_folder}/{font}/{folder}/output-png/"
        for images in sorted(os.listdir(png_result_path)):
            if (images.endswith(".png")):
                png_result_path = f"{png_result_path}/{images}"
                break
        shutil.copy(png_result_path, f"/scratch/gilbreth/zhao969/Word-As-Image/merge/{word}/png/{character_conversion}_{result.group(5)}.png")
        result_pngs.append(character_conversion)
    return result_pngs

def create_word(word, my_font="ambigram_font", weight=True):
    svgs_conversion = []
    conversion_weight = []
    for file in glob.glob(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/filter_svg/*.svg"):
        filename = os.path.splitext(os.path.basename(file).split('/')[-1])[0]
        conversion = filename[0:6]
        svgs_conversion.append(conversion)
        if weight:
            conversion_weight.append(float(filename[7:]))

    svgs_lower = [svg.lower() for svg in svgs_conversion]
    for i in range(len(word)):
        if (f"{word[i].lower()}_to_{word[len(word)-1-i].lower()}" not in svgs_lower):
            if (f"{word[len(word)-1-i].lower()}_to_{word[i].lower()}" in svgs_lower):
                index = svgs_lower.index(f"{word[len(word)-1-i].lower()}_to_{word[i].lower()}")
                if weight:
                    rotate_svg(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/filter_svg/{svgs_conversion[index]}_{conversion_weight[index]}.svg", 
                                f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/filter_svg/{svgs_conversion[index][-1]}_to_{svgs_conversion[index][0]}_{1.0-float(conversion_weight[index])}.svg")
                else:
                    rotate_svg(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/filter_svg/{svgs_conversion[index]}.svg", 
                                f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/filter_svg/{svgs_conversion[index][-1]}_to_{svgs_conversion[index][0]}.svg")
            else:
                raise Exception("Cannot find the svg for the character conversion.")
    
    folder_svg = []
    for f in os.listdir(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/filter_svg/"):
        if f.endswith('.svg'):
            folder_svg.append(f)
    if weight:
        folder_svg_lower = ["_".join(svg.lower().split("_")[0:-1]) for svg in folder_svg]
    else:
        folder_svg_lower = [svg.lower().split(".")[0] for svg in folder_svg]
    order_svg = []
    for i in range(len(word)):
        order_svg.append(folder_svg[folder_svg_lower.index(f"{word[i].lower()}_to_{word[len(word)-1-i].lower()}")])
    template_word = "O" * len(word)
    count = [1] * 26
    template_file = generate_template_svg(template_word=template_word, template_name=f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/template.svg")
    # replicate_shapes(template_file, template_file)
    for i in range(len(word)):
        shutil.copy(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/filter_svg/{order_svg[i]}", f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/temp/{i}_{order_svg[i]}")
    for i in range(len(word)):
        print(f"Processing {order_svg[i]} character")
        scale_into_wordsize(template_word, template_word[i], template_file, f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/temp/{i}_{order_svg[i]}", num_rec=count[ord(template_word[i])-ord('A')])
        count[ord(template_word[i])-ord('A')] = count[ord(template_word[i])-ord('A')] + 1
    order_svg_temp = []
    for idx, name in enumerate(order_svg):
        order_svg_temp.append(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/temp/{idx}_{name}")
    # order_svg = [f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/temp/{name}" for name in order_svg]
    stack_all_svgs(order_svg_temp, template_file)
    shutil.copy(template_file, f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/result/{word}.svg")

def svg_to_png(svg_file, result_folder):
    filename = os.path.splitext(os.path.basename(svg_file).split('/')[-1])[0]
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()
    w, h = 224, 224
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
        svg_file)

    # render init image
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        w, h, shapes_init, shape_groups_init)
    img_init = render(w, h, 2, 2, 0, None, *scene_args)
    img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + \
        torch.ones(img_init.shape[0], img_init.shape[1],
                3, device=device) * (1 - img_init[:, :, 3:4])
    img_init = img_init[:, :, :3]
    save_image(img_init, f"{result_folder}/{filename}.png", 1.0)

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def edit_distance_log(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    substitutions = []
    matches = []

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # Insert
                                   dp[i][j-1],      # Remove
                                   dp[i-1][j-1])    # Replace

    # Find substitutions and matches
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            matches.append((i-1, s1[i-1]))
            i -= 1
            j -= 1
        elif s1[i-1] != s2[j-1] and dp[i][j] == 1 + dp[i-1][j-1]:
            substitutions.append((i-1, s1[i-1], s2[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == 1 + dp[i-1][j]:
            i -= 1
        elif j > 0 and dp[i][j] == 1 + dp[i][j-1]:
            j -= 1
        else:
            i -= 1
            j -= 1

    return dp[m][n], substitutions[::-1], matches[::-1]

def run_OCR():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed').to(device)
    for custom_font in ["ambigram_font", "ambigram_font_2", "ambifusion_font"]:
        result = []
        total_dist = 0
        confusion_matrix = [[0 for i in range(27)] for j in range(26)]
        for file in glob.glob(f"/scratch/gilbreth/zhao969/Word-As-Image/{custom_font}/high_res_word/*.png"):
            image = Image.open(file).convert("RGB")
            word = os.path.splitext(os.path.basename(file).split('/')[-1])[0]
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text_norm = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # edit_dist_norm = levenshteinDistance(word.lower(), generated_text_norm.lower())
            edit_dist_norm, substitutions_norm, match_norm = edit_distance_log(word.lower(), generated_text_norm.lower())
            for _, orig, new in substitutions_norm:
                if (ord(new)-ord('a')) < 0 or (ord(new)-ord('a')) > 25:
                    confusion_matrix[ord(orig)-ord('a')][26] = confusion_matrix[ord(orig)-ord('a')][26] + 1
                else:
                    confusion_matrix[ord(orig)-ord('a')][ord(new)-ord('a')] = confusion_matrix[ord(orig)-ord('a')][ord(new)-ord('a')] + 1
            for _, char in match_norm:
                confusion_matrix[ord(char)-ord('a')][ord(char)-ord('a')] = confusion_matrix[ord(char)-ord('a')][ord(char)-ord('a')] + 1
            rotated_image = image.rotate(180)
            pixel_values = processor(images=rotated_image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text_rotated = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # edit_dist_rotated = levenshteinDistance(word.lower(), generated_text_rotated.lower())
            edit_dist_rotated, substitutions_rotated, match_rotated = edit_distance_log(word.lower(), generated_text_rotated.lower())
            for _, orig, new in substitutions_rotated:
                if (ord(new)-ord('a')) < 0 or (ord(new)-ord('a')) > 25:
                    confusion_matrix[ord(orig)-ord('a')][26] = confusion_matrix[ord(orig)-ord('a')][26] + 1
                else:
                    confusion_matrix[ord(orig)-ord('a')][ord(new)-ord('a')] = confusion_matrix[ord(orig)-ord('a')][ord(new)-ord('a')] + 1
            for _, char in match_rotated:
                confusion_matrix[ord(char)-ord('a')][ord(char)-ord('a')] = confusion_matrix[ord(char)-ord('a')][ord(char)-ord('a')] + 1
            total_dist = total_dist + edit_dist_norm + edit_dist_rotated

            result.append(f"{word} ----> norm: {generated_text_norm} rotate: {generated_text_rotated} with edit distance: {edit_dist_norm} and {edit_dist_rotated}")
        average_dist = total_dist / len(result)
        result.insert(0, f"Average edit distance: {average_dist}")
        confusion_matrix = [[float(confusion_matrix[i][j]) / sum(confusion_matrix[i]) for j in range(27)] for i in range(26)]
        with open(f'./OCR_result/{custom_font}_result.txt', 'w') as f:
            for line in result:
                f.write(f"{line}\n")
        # with open(f'./OCR_result/{custom_font}_confusion_matrix.pkl', 'wb') as f:
        #     pickle.dump(confusion_matrix, f)
        heat_map_gen(confusion_matrix, f"./OCR_result/{custom_font}_confusion_matrix")
    
    for baseline_font in ["ambimaticv2", "ambidream", "ambigramania", "dsmonoHD"]:
        result = []
        total_dist = 0
        confusion_matrix = [[0 for i in range(27)] for j in range(26)]
        for file in glob.glob(f"/scratch/gilbreth/zhao969/BenchMark/{baseline_font}/resized/*.png"):
            image = Image.open(file).convert("RGB")
            word = os.path.splitext(os.path.basename(file).split('/')[-1])[0]
            word = word.split('_')[0]
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text_norm = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # edit_dist_norm = levenshteinDistance(word.lower(), generated_text_norm.lower())
            edit_dist_norm, substitutions_norm, match_norm = edit_distance_log(word.lower(), generated_text_norm.lower())
            for _, orig, new in substitutions_norm:
                if (ord(new)-ord('a')) < 0 or (ord(new)-ord('a')) > 25:
                    confusion_matrix[ord(orig)-ord('a')][26] = confusion_matrix[ord(orig)-ord('a')][26] + 1
                else:
                    confusion_matrix[ord(orig)-ord('a')][ord(new)-ord('a')] = confusion_matrix[ord(orig)-ord('a')][ord(new)-ord('a')] + 1
            for _, char in match_norm:
                confusion_matrix[ord(char)-ord('a')][ord(char)-ord('a')] = confusion_matrix[ord(char)-ord('a')][ord(char)-ord('a')] + 1
            rotated_image = image.rotate(180)
            pixel_values = processor(images=rotated_image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text_rotated = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # edit_dist_rotated = levenshteinDistance(word.lower(), generated_text_rotated.lower())
            edit_dist_rotated, substitutions_rotated, match_rotated = edit_distance_log(word.lower(), generated_text_rotated.lower())
            for _, orig, new in substitutions_rotated:
                if (ord(new)-ord('a')) < 0 or (ord(new)-ord('a')) > 25:
                    confusion_matrix[ord(orig)-ord('a')][26] = confusion_matrix[ord(orig)-ord('a')][26] + 1
                else:
                    confusion_matrix[ord(orig)-ord('a')][ord(new)-ord('a')] = confusion_matrix[ord(orig)-ord('a')][ord(new)-ord('a')] + 1
            for _, char in match_rotated:
                confusion_matrix[ord(char)-ord('a')][ord(char)-ord('a')] = confusion_matrix[ord(char)-ord('a')][ord(char)-ord('a')] + 1
            total_dist = total_dist + edit_dist_norm + edit_dist_rotated
            
            result.append(f"{word} ----> norm: {generated_text_norm} rotate: {generated_text_rotated} with edit distance: {edit_dist_norm} and {edit_dist_rotated}")
        average_dist = total_dist / len(result)
        result.insert(0, f"Average edit distance: {average_dist}")
        confusion_matrix = [[float(confusion_matrix[i][j]) / sum(confusion_matrix[i]) for j in range(27)] for i in range(26)]
        with open(f'./OCR_result/{baseline_font}_result.txt', 'w') as f:
            for line in result:
                f.write(f"{line}\n")
        # with open(f'./OCR_result/{baseline_font}_confusion_matrix.pkl', 'wb') as f:
        #     pickle.dump(confusion_matrix, f)   
        heat_map_gen(confusion_matrix, f"./OCR_result/{baseline_font}_confusion_matrix")

def run_OCR_word_optimized():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed').to(device)
    # for custom_font in ["ambigram_font", "ambigram_font_2"]:
    for custom_font in ["ambigram_font_2"]:
        result = []
        total_dist = 0
        confusion_matrix = [[0 for i in range(27)] for j in range(26)]
        for file in glob.glob(f"/scratch/gilbreth/zhao969/Word-As-Image/{custom_font}/after_whole_word_opt_sym/png/*.png"):
            image = Image.open(file).convert("RGB")
            word = os.path.splitext(os.path.basename(file).split('/')[-1])[0]
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text_norm = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # edit_dist_norm = levenshteinDistance(word.lower(), generated_text_norm.lower())
            edit_dist_norm, substitutions_norm, match_norm = edit_distance_log(word.lower(), generated_text_norm.lower())
            for _, orig, new in substitutions_norm:
                if (ord(new)-ord('a')) < 0 or (ord(new)-ord('a')) > 25:
                    confusion_matrix[ord(orig)-ord('a')][26] = confusion_matrix[ord(orig)-ord('a')][26] + 1
                else:
                    confusion_matrix[ord(orig)-ord('a')][ord(new)-ord('a')] = confusion_matrix[ord(orig)-ord('a')][ord(new)-ord('a')] + 1
            for _, char in match_norm:
                confusion_matrix[ord(char)-ord('a')][ord(char)-ord('a')] = confusion_matrix[ord(char)-ord('a')][ord(char)-ord('a')] + 1
            rotated_image = image.rotate(180)
            pixel_values = processor(images=rotated_image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text_rotated = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # edit_dist_rotated = levenshteinDistance(word.lower(), generated_text_rotated.lower())
            edit_dist_rotated, substitutions_rotated, match_rotated = edit_distance_log(word.lower(), generated_text_rotated.lower())
            for _, orig, new in substitutions_rotated:
                if (ord(new)-ord('a')) < 0 or (ord(new)-ord('a')) > 25:
                    confusion_matrix[ord(orig)-ord('a')][26] = confusion_matrix[ord(orig)-ord('a')][26] + 1
                else:
                    confusion_matrix[ord(orig)-ord('a')][ord(new)-ord('a')] = confusion_matrix[ord(orig)-ord('a')][ord(new)-ord('a')] + 1
            for _, char in match_rotated:
                confusion_matrix[ord(char)-ord('a')][ord(char)-ord('a')] = confusion_matrix[ord(char)-ord('a')][ord(char)-ord('a')] + 1
            total_dist = total_dist + edit_dist_norm + edit_dist_rotated

            result.append(f"{word} ----> norm: {generated_text_norm} rotate: {generated_text_rotated} with edit distance: {edit_dist_norm} and {edit_dist_rotated}")
        average_dist = total_dist / len(result)
        result.insert(0, f"Average edit distance: {average_dist}")
        confusion_matrix = [[float(confusion_matrix[i][j]) / sum(confusion_matrix[i]) for j in range(27)] for i in range(26)]
        with open(f'./OCR_result/{custom_font}_whole_word_opt_sym_result.txt', 'w') as f:
            for line in result:
                f.write(f"{line}\n")
        # with open(f'./OCR_result/{custom_font}_confusion_matrix.pkl', 'wb') as f:
        #     pickle.dump(confusion_matrix, f)
        heat_map_gen(confusion_matrix, f"./OCR_result/{custom_font}_whole_word_opt_sym_confusion_matrix")
    

def heat_map_gen(confusion_matrix, file):
    data = np.array(confusion_matrix)
    fig, ax = plt.subplots(figsize=(20,15))
    sns.heatmap(data, linewidth=0.5, xticklabels="abcdefghijklmnopqrstuvwxyz", yticklabels="abcdefghijklmnopqrstuvwxyz", annot=True, fmt=".2f", ax=ax)
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    # plt.savefig(file)
    plt.savefig(f"{file}.pdf", format="pdf")
    plt.clf()
    plt.close()

def svg_scaling(font_name="ambigram_font", scale=4):
    import svgutils
    import math
    int_size = math.ceil(float(224) * scale)
    for file in glob.glob(f"/scratch/gilbreth/zhao969/Word-As-Image/{font_name}/result/*.svg"):
        filename = os.path.splitext(os.path.basename(file).split('/')[-1])[0]
        originalSVG = svgutils.compose.SVG(file)
        originalSVG.scale(scale)
        w, h = int_size, int_size
        device = pydiffvg.get_device()
        figure = svgutils.compose.Figure(float(224) * scale, float(224) * scale, originalSVG)
        figure.save(f"/scratch/gilbreth/zhao969/Word-As-Image/{font_name}/high_res_word/{filename}.svg")
        render = pydiffvg.RenderFunction.apply
        canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
                    f"/scratch/gilbreth/zhao969/Word-As-Image/{font_name}/high_res_word/{filename}.svg")
        scene_args = pydiffvg.RenderFunction.serialize_scene(
                    int_size, int_size, shapes_init, shape_groups_init)
                
        word_img = render(w, h, 2, 2, 0, None, *scene_args)
        word_img = word_img[:, :, 3:4] * word_img[:, :, :3] + \
            torch.ones(word_img.shape[0], word_img.shape[1],
                    3, device=device) * (1 - word_img[:, :, 3:4])
        word_img = word_img[:, :, :3]
        x = word_img.unsqueeze(0).permute(0, 3, 1, 2)
        x_blured = KF.median_blur(x.detach(), (5, 5))
        plt.imsave(f"/scratch/gilbreth/zhao969/Word-As-Image/{font_name}/high_res_word/{filename}.png", x_blured.squeeze().permute(1, 2, 0).detach().cpu().numpy())

def glue_png_word(word):
    png_path = "/scratch/gilbreth/zhao969/Word-As-Image/ambifusion_font/all_pairs/{filename}.png"
    input_files = []
    for i in range(len(word)):
        input_files.append(png_path.format(filename=f"{word[i].lower()}_to_{word[len(word)-1-i].lower()}"))
    images = [Image.open(file) for file in input_files]
    total_width = sum(image.width for image in images)
    height = 64
    output_height = height + 20  # Add 10 pixels of white space above and below
    output_image = Image.new('RGBA', (total_width, output_height), (255, 255, 255))  # White background
    y_offset = (output_height - height) // 2
    x_offset = 0
    for image in images:
        output_image.paste(image, (x_offset, y_offset))
        x_offset += image.width
    output_image.save(f'/scratch/gilbreth/zhao969/Word-As-Image/ambifusion_font/result_png/{word}.png')
    for image in images:
        image.close()

def resize(image_pil, width, height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    ratio_w = width / image_pil.width
    ratio_h = height / image_pil.height
    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)
    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)
        resize_height = height
    image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert('RGB')

def resize_output_png(size=384):
    for custom_font in ["ambifusion_font"]:
        for file in glob.glob(f"/scratch/gilbreth/zhao969/Word-As-Image/{custom_font}/result_png/*.png"):
            image = Image.open(file).convert("RGB")
            resize(image, size, size).save(file)

    for baseline_font in ["ambimaticv2", "ambidream", "ambigramania", "dsmonoHD"]:
        for file in glob.glob(f"/scratch/gilbreth/zhao969/BenchMark/{baseline_font}/resized/*.png"):
            image = Image.open(file).convert("RGB")
            resize(image, size, size).save(file)

if __name__ == "__main__":


    # run_OCR()
    run_OCR_word_optimized()

    # resize_output_png()
    # scale = 384.0 / 224.0
    # svg_scaling(font_name="ambigram_font", scale=scale)
    # svg_scaling(font_name="ambigram_font_2", scale=scale)

    word = "abcdefghijklmnopqrstuvwxyz"
    # First Step. Copy All the output png to the merge folder so that we can convert the png to svg
    
    # files = read_choice(f"/scratch/gilbreth/zhao969/Word-As-Image/merge/{word}/selected_images.txt")
    # Path(f"/scratch/gilbreth/zhao969/Word-As-Image/merge/{word}/png/").mkdir(parents=True, exist_ok=True)
    # copy_chosen_png(files, f"dual_if_{word}", word)

    # Step 2 Median filter the png
    # import kornia.filters as KF
    # import torchvision
    # import glob
    # import torchvision.transforms as tvt
    # Path(f"/scratch/gilbreth/zhao969/Word-As-Image/merge/{word}/median_filter_png/").mkdir(parents=True, exist_ok=True)
    # for file in glob.glob(f"/scratch/gilbreth/zhao969/Word-As-Image/merge/{word}/png/*.png"):
    #     img = torchvision.io.read_image(file)
    #     word_img = img.float() / 255.0
    #     x = tvt.functional.invert(word_img).unsqueeze(0)
    #     x_blured = KF.median_blur(x, (5, 5))
    #     filename = os.path.splitext(os.path.basename(file).split('/')[-1])[0]
    #     plt.imsave(f"/scratch/gilbreth/zhao969/Word-As-Image/merge/{word}/median_filter_png/{filename}.png", tvt.functional.invert(x_blured.squeeze()).permute(1, 2, 0).detach().cpu().numpy())

    # Step 3 Copy the folder to the other server for png->svg conversion

    # Step 4 Copy the result back (should be in a folder called filter_svg) and generate all words result
    # my_font = "ambigram_font_2"
    # Path(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/temp/").mkdir(parents=True, exist_ok=True)
    # Path(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/result/").mkdir(parents=True, exist_ok=True)
    # Path(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/high_res_word/").mkdir(parents=True, exist_ok=True)
    # Path(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/result_png/").mkdir(parents=True, exist_ok=True)
    
    # words = []
    # with open("/scratch/gilbreth/zhao969/BenchMark/words.txt", "r") as input_file:
    #     for line in input_file:
    #         word = line.strip()
    #         words.append(word)
    
    # for word in words:
    #     create_word(word, my_font=my_font)

    # svg_scaling(font_name=my_font)

    # for file in glob.glob(f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/result/*.svg"):
    #     svg_to_png(file, f"/scratch/gilbreth/zhao969/Word-As-Image/{my_font}/result_png")