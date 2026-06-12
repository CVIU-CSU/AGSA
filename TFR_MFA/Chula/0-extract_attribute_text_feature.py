# env : cell2


"shape"
shape = \
[['The attribute description is unclear'],["overall irregular circle, lumpy or tumor-like protrusions along the edges"], ["the cell is circular in shape"], ["irregular or fragmented circle"],
 ["circular or oval in shape"],["small and circular"],["teardrop-shaped, with one end pointed and the other end rounded."],
 ["elongated oval-shaped"],["small in size, with irregular shapes, resembling spikes, helmets, triangles, or other forms"],["irregular circular"],
 ["Small circular or irregular shapes, possibly with small protrusions at the edges."],["Relatively small circular or irregular shape."],["Irregular or round or rectangular in shape"],
 ["highly irregular, often with small protrusions"],["oval-shaped, with many large protrusions."],["The cell body is extremely large, oval or irregular"],
 ["Irregular shape, with the cell edges being rugged and uneven, resembling a coastline."],["The shape is irregular, bluish-purple, with many small bluish-purple dots attached around the edges."],["An irregular purple clump, often containing many white gaps, resembling something that has been shattered."],
 ["Very small, tiny blue-purple dots."],["A small number of blue-purple dots gathered together."],["A large number of platelets gathered in a clump."], ["The cell is round and noticeably larger than a normal red blood cell, consistent with a macrocytic appearance."],
 ["The cell shows an irregular and asymmetric outline rather than a smooth round contour."], ["The cell is generally round with a relatively thin rim of cytoplasm."], ["The cell is small and round, with an overall diameter clearly reduced compared with normal erythrocytes."],
 ["The cell is round and spherical, lacking the normal biconcave contour."], ["The cell is elongated and oval in shape rather than round."], ["The cell is round with multiple short, evenly spaced projections along the cell margin."]]
print(len(shape))

"nucleus" # nucleus or central pallor
nucleus = \
[['The attribute description is unclear'],["blue-purple circular or oval nucleus"], ["have a deep blue or purple circular nucleus"],["dark purple nucleus of round or oval shape"],
 ["Small dark blue-purple nucleus, round in shape, with a clear boundary"],["Two nuclei, positioned closely together."],["The nucleus is dispersed, or split into two parts."],
 ["Purple, round cell nucleus with an unclear boundary."],["oval-shaped nucleus in blue or purple color"],["The nucleus is indented, resembling the shape of a broad bean, and is positioned toward one side of the cell."],
 ["The nucleus is distinctly indented and elongated, deep purple in color, resembling the letter 'U' or a long band"],["The nucleus is divided into several parts, resembling separated leaves connected by filaments."],["Purple oval-shaped nucleus, occupying a very large proportion."],
 ["A circular dark blue nucleus, with almost no cytoplasm."],["Irregularly shaped light purple nucleus with a large nuclear-cytoplasmic ratio."],["The nucleus is oval or irregular in shape, purple, often with indentations or notches"],
 ["The nucleus is round or oval, bluish-purpler, and clearly positioned towards one side of the cell."],["Two or more nuclei."],["The nucleus is large, purple, and shaped like a crescent or fetus."],
 ["The nucleus has an irregular shape, dark blue, often containing multiple nuclei that overlap or are dispersed."],["The nucleus is deep blue and often occupies the entire cell."],["There is only a giant bluish-purple nucleus, irregular in shape, resembling a tumor."],
 ["The center appears as a lightly stained circular area, occupying about one-third of the cell diameter."], ["The central light staining area appears as a narrow longitudinal crack or elliptical strip, resembling a 'mouth' shape, with significantly weakened central staining."],
 ["The central lightly stained area is usually not obvious or irregularly lightly stained, with uneven staining and accompanied by fragmented or serrated edges."],
 ["The central light staining area is usually located at the wider blunt end, presenting as a localized or irregular light staining area, gradually weakening or disappearing towards the tip."],
 ["The central lightly stained area is usually elongated and aligned with the long axis direction of the cell."], ["There will be an isolated deep dye area within the central light dye zone, resembling a shooting bullseye."],
 ["The central lightly stained area is reduced or less distinct compared with normal erythrocytes."], ["A central lightly stained area is present but appears uneven and not well centered."],
 ["The central lightly stained area is markedly enlarged, occupying a large proportion of the cell."], ["The central lightly stained area is relatively large in proportion to the cell size."], ["The central lightly stained area is absent or nearly absent."],
 ["The central lightly stained area is present and often appears elongated following the cell’s long axis."], ["The central lightly stained area is present but may appear reduced or irregular."]]
print(len(nucleus))

"chromatin"
chromatin = \
[['The attribute description is unclear'],["the chromatin appears uniformly coarse-grained"], ["the chromatin is uniform and loosely distributed"], ["the chromatin is overall uniform and partially aggregated into clumps"],
 ["The chromatin is concentrated into distinct dark purple clumps, with clear gaps between them, resembling fragmented pieces."], ["The chromatin is condensed into a uniform dark purple-black"],["The chromatin is like dark purple block or stripe."],
 ["Chromatin is unevenly distributed, forming purple clumps."],["The chromatin exhibits a fine, loose, mesh-like structure."],["The chromatin is condensed into distinct blue clumps."]
 ]
print(len(chromatin))

"cytoplasm"
cytoplasm = \
[['The attribute description is unclear'],["cytoplasm is dark blue, resembling a blue ring around the nucleus"], ["blue cytoplasm"], ["light blue or light purple cytoplasm"],
 ["The cytoplasm is very pale and light, causing a sharp contrast with the nucleus"],["the center of the cytoplasm is lighter than the edge of the cell"],["The cytoplasm has a central dark-stained area, surrounded by a lighter zone, and at the outer edge is a dark-stained ring, resembling a shooting target."],
 ["There is a narrow, flat, light-colored fissure in the center of the cytoplasm, resembling an open mouth overall."],["The cytoplasm is uniformly stained."],["The cytoplasm is light purple and clear."],
 ["The cytoplasm is filled with blue granules of varying sizes."],["The cytoplasm is light purple, containing a few small purple granules."],["The cytoplasm is a very light purple color, containing tiny, scattered purple granules."],
 ["The cytoplasm is rich in numerous pink or red particles."],["The entire cell contains abundant large deep blue particles."],["The cytoplasm is light blue or light purple, very little, and adheres to one side of the nucleus."],
 ["The cytoplasm is a very light, transparent pale blue."],["The cytoplasm is a murky purple, containing evenly distributed purple granules, occasionally with vacuoles."],["The cytoplasm is opaque deep blue."],
 ["The cytoplasm is blue or light purple, cloudy, abundant, with a foamy appearance, sometimes containing bubbles."],["The cytoplasm is light purple, very cloudy, sparse, and contains purple clumps."],["The cytoplasm is very abundant, light purple in color, and filled with densely packed small tiny dots, giving an overall foamy appearance."],
 ["The cytoplasm is light purple, very cloudy with a strong foamy appearance, containing a few blue round dots (platelets)."], ["The cytoplasm appears abundant and uniformly stained, reflecting increased cell volume."], ["The cytoplasm shows heterogeneous staining intensity with localized variations across the cell."],
 ["The cytoplasm is pale and lightly stained, consistent with reduced hemoglobin content."], ["The cytoplasm is scant and often appears more densely packed due to the reduced cell volume."], ["The cytoplasm appears dense and uniformly stained, giving a hyperchromic appearance."],
 ["The cytoplasm is evenly distributed with relatively uniform staining."], ["The cytoplasm is relatively uniform in staining, with the membrane projections clearly outlined."]]
print(len(cytoplasm))

"ratio" 
ratio = \
[['The attribute description is unclear'],["very large nucleus/cytoplasm ratio"], ["relatively big nucleus/cytoplasm ratio"], ["middle nucleus/cytoplasm ratio, about 1/2"],
 ["small nucleus/cytoplasm ratio"],["its nucleus/cytoplasm ratio is about 2/3"]]
print(len(ratio))


correspondence_matrix = [
    [5, 22, 0, 5, 0], # 0
    [22, 28, 0, 23, 0], # 1
    [23, 29, 0, 24, 0], # 10
    [24, 30, 0, 25, 0], # 11
    [7, 26, 0, 0, 0], # 12
    [25, 31, 0, 26, 0], # 2
    [26, 32, 0, 27, 0], # 3
    [5, 27, 0, 6, 0], # 4
    [5, 23, 0, 7, 0], # 5
    [27, 33, 0, 28, 0], # 6
    [6, 25, 0, 5, 0], # 7
    [28, 34, 0, 29, 0], # 8
    [8, 24, 0, 8, 0]  # 9
]


all_texts = [shape, nucleus, chromatin, cytoplasm, ratio]
all_cols = [0, 1, 2, 3, 4] 

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import open_clip
import os
import torch
import json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

print('over')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
model.to(device)
model.eval()

context_length = 256
import torch.nn.functional as F
save_data = dict()
with torch.no_grad():
    for i, cls in enumerate(all_texts):

        if i == 0: att_name = 'shape_des_pool'
        elif i == 1: att_name = 'nucleus_des_pool'
        elif i == 2: att_name = 'chromatin_des_pool'
        elif i == 3: att_name = 'cytoplasm_des_pool'
        elif i == 4: att_name = 'ratio_des_pool'

        cls_embeddings = None
        for i, att_text in enumerate(cls):
            texts = tokenizer(att_text, context_length=context_length).to(device)
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings = F.normalize(class_embeddings, dim=-1, p=2)
            if cls_embeddings is None: cls_embeddings = class_embeddings
            else: cls_embeddings = torch.cat((cls_embeddings, class_embeddings), dim=0)

        save_data[att_name] = cls_embeddings.cpu().detach()

    for col in all_cols:

        if col == 0: att_name = 'shape_cls_label'
        elif col == 1: att_name = 'nucleus_cls_label'
        elif col == 2: att_name = 'chromatin_cls_label'
        elif col == 3: att_name = 'cytoplasm_cls_label'
        elif col == 4: att_name = 'ratio_cls_label'

        tmp_list = []
        for item in correspondence_matrix:
            tmp_list.append(item[col])

        save_data[att_name] = torch.tensor(tmp_list)

# save
torch.save(save_data, 'xxx/att_text_raw_feat.pth')