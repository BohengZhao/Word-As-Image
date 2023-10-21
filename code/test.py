import os
weight_sweep=[0.4, 0.45, 0.5, 0.55, 0.6]
# alignment=["none", "overlap", "seperated_2", "seperated"]
alignment=["none", "overlap"]
alphabet = "abcdefghijklmnopqrstuvwxyz"
folders = [name for name in os.listdir("/scratch/gilbreth/zhao969/Word-As-Image/output/dual_if_abcdefghijklmnopqrstuvwxyz/IndieFlower-Regular/")]
folders_lowercase = [name.lower() for name in folders]
set_of_conversion = []
for foldername in folders_lowercase:
     convert = f"{foldername[0:6]}"
     set_of_conversion.append(convert)
set_of_conversion = set(set_of_conversion)
print(len(set_of_conversion))
ideal_conversion = []
for idx in range(len(alphabet)):
    for idx_2 in range(idx, len(alphabet)):
        ideal_conversion.append(f"{alphabet[idx]}_to_{alphabet[idx_2]}")
print(len(ideal_conversion))
ideal_conversion = set(ideal_conversion)
print(ideal_conversion.difference(set_of_conversion))
for diff in sorted(ideal_conversion.difference(set_of_conversion)):
    init_char = ord(diff[0])-ord('a')
    target_char = ord(diff[-1])-ord('a')
    print(f"run_{init_char}_to_{target_char}_0.sh")

missing = []
missing_bash = []
for idx in range(len(alphabet)):
    for idx_2 in range(idx, len(alphabet)):
        for align_idx in range(len(alignment)):
            init_char = [alphabet[idx].upper(), alphabet[idx].lower()]
            target_char = [alphabet[idx_2].upper(), alphabet[idx_2].lower()]
            alignments = alignment[align_idx]
            for weight in weight_sweep:
                if f"{init_char[1]}_to_{target_char[0]}_dual_if_{alignments}_{weight}" not in folders:
                    missing_bash.append(f"run_{idx}_to_{idx_2}_{align_idx}.sh")
                    missing.append(f"{init_char[1]}_to_{target_char[0]}_dual_if_{alignments}_{weight}")
                if f"{init_char[1]}_to_{target_char[1]}_dual_if_{alignments}_{weight}" not in folders:
                    missing_bash.append(f"run_{idx}_to_{idx_2}_{align_idx}.sh")
                    missing.append(f"{init_char[1]}_to_{target_char[1]}_dual_if_{alignments}_{weight}")
                if f"{init_char[0]}_to_{target_char[0]}_dual_if_{alignments}_{weight}" not in folders:
                    missing_bash.append(f"run_{idx}_to_{idx_2}_{align_idx}.sh")
                    missing.append(f"{init_char[0]}_to_{target_char[0]}_dual_if_{alignments}_{weight}")
                if f"{init_char[0]}_to_{target_char[1]}_dual_if_{alignments}_{weight}" not in folders:
                    missing_bash.append(f"run_{idx}_to_{idx_2}_{align_idx}.sh")
                    missing.append(f"{init_char[0]}_to_{target_char[1]}_dual_if_{alignments}_{weight}")
print(sorted(set(missing_bash)))