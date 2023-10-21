import os
from glob import glob
import shutil
import jinja2
import re

def copy_and_rename(reg='/scratch/gilbreth/zhao969/Word-As-Image/output/*/*/*/output-png/*.png'):
    for filename in glob(reg, recursive=True):
        shutil.copy(filename, "/scratch/gilbreth/zhao969/Word-As-Image/code/selection_img/{}-{}-{}.png".format(filename.split('/')[-4], filename.split('/')[-3], filename.split('/')[-1].split('_')[0]))

def copy_and_rename_whole_word(reg='/scratch/gilbreth/zhao969/Word-As-Image/output/*/*/*/output-png/*.png'):
    for filename in glob(reg, recursive=True):
        shutil.copy(filename, "/scratch/gilbreth/zhao969/Word-As-Image/ambigram_font_2/after_whole_word_opt_sym/png/{}".format(filename.split('/')[-1].split('_')[-1]))

def copy_and_rename_whole_word_svg(reg='/scratch/gilbreth/zhao969/Word-As-Image/output/*/*/*/output-svg/*.svg'):
    for filename in glob(reg, recursive=True):
        name = os.path.basename(filename)
        if name.startswith('scaled'):
            shutil.copy(filename, "/scratch/gilbreth/zhao969/Word-As-Image/ambigram_font_2/after_whole_word_opt_sym/svg/{}".format(filename.split('/')[-1].split('_')[-1]))

if __name__ == '__main__':
    copy_and_rename_whole_word_svg()
    copy_and_rename_whole_word()
    # copy_and_rename()

    # print("Finish copying and renaming.")

    # env = jinja2.Environment(
    #         loader=jinja2.FileSystemLoader('.'),
    #         trim_blocks=True,
    #         lstrip_blocks=True,
    #     )

    # template = env.get_template("template_choice.html")
    # template_vars = [[] for i in range(26 * 26)]
    # for filename in glob("/scratch/gilbreth/zhao969/Word-As-Image/code/selection_img/*.png"):
    #     # print(filename.split('/')[-1])
    #     init_char = re.search(".*-(.)_to_(.).*_(\d+\.\d+)-(.*[Ee]?[-+]?.*).png", filename.split('/')[-1]).group(1)
    #     target_char = re.search(".*-(.)_to_(.).*_(\d+\.\d+)-(.*[Ee]?[-+]?.*).png", filename.split('/')[-1]).group(2)
    #     weight = float(re.search(".*-(.)_to_(.).*_(\d+\.\d+)-(.*[Ee]?[-+]?.*).png", filename.split('/')[-1]).group(3))
    #     loss = float(re.search(".*-(.)_to_(.).*_(\d+\.\d+)-(.*[Ee]?[-+]?.*).png", filename.split('/')[-1]).group(4))
    #     template_vars[(ord(init_char.lower()) - ord('a')) * 26 + (ord(target_char.lower()) - ord('a'))].append({"title": filename.split('/')[-1], "graph": filename, "weight": weight, "name": init_char + " to " + target_char + " with weight " + str(weight), "init_char": init_char, "target_char": target_char, "loss": loss})

    # #print(template_vars)
    # #template_vars = [sorted(sorted(images, key=lambda d: d['loss']), key=lambda d:ord(d["target_char"])) for images in template_vars]
    # template_vars = [sorted(images, key=lambda d: d['loss'])[0:6] for images in template_vars]
    # template_vars = [images for images in template_vars if images != []]
    # text = template.render(template_vars=template_vars)

    # with open("make_selection.html", "w") as f_out:
    #     f_out.write(text)