# Retouche du visage, de la peau et restauration par IA : rapport

Branche `skin-wrinkle-model`. Pull request en brouillon : [tybiboune/RapidRAW#12](https://github.com/tybiboune/RapidRAW/pull/12).
Date du rapport : 22 novembre 2026.

Ce document décrit tout ce qui a été ajouté ou modifié dans RapidRAW sur cette branche :

- ce que fait chaque fonction et comment l'utiliser ;
- comment elle est construite (fichiers, algorithmes, modèles) ;
- ce qu'il faut tester, avec la marche à suivre.

**Important — ce qui n'a pas été fait.** Aucune de ces fonctions n'a encore été lancée dans l'application. Ce PC n'a pas de chaîne de compilation Rust ; la compilation a été vérifiée par la CI GitHub. Les algorithmes ont été validés à part, en Python, avec les mêmes modèles ONNX et des copies fidèles des calculs. Le comportement réel dans RapidRAW reste à vérifier en suivant la [liste de tests](#7-liste-de-tests-dans-lordre).

---

## Sommaire

1. [Vue d'ensemble](#1-vue-densemble)
2. [Les modèles et leur installation](#2-les-modèles-et-leur-installation)
3. [Outil Peau : nouvelles corrections](#3-outil-peau--nouvelles-corrections)
4. [Segmentation fine du visage](#4-segmentation-fine-du-visage)
5. [Outils de retouche au pinceau](#5-outils-de-retouche-au-pinceau)
6. [Agrandissement ×4](#6-agrandissement-4)
7. [Liste de tests, dans l'ordre](#7-liste-de-tests-dans-lordre)
8. [Ce qui a été vérifié, et comment](#8-ce-qui-a-été-vérifié-et-comment)
9. [Limites connues et points d'attention](#9-limites-connues-et-points-dattention)
10. [Fichiers modifiés](#10-fichiers-modifiés)
11. [Licences](#11-licences)

---

## 1. Vue d'ensemble

| Fonction | Où dans l'interface | Modèle / méthode |
|---|---|---|
| Rides par IA | Effets → Peau → curseur **Lines** → interrupteur **Find lines with AI** | FFHQ-Wrinkle U-Net |
| Boutons, grains de beauté, milium par IA | Effets → Peau → curseur **Spots** → interrupteur **Find spots with AI** | Détecteur de défauts ABPN (Alibaba) |
| Taches pigmentaires | Effets → Peau → curseur **Pigment spots** (+ curseur de détection) | Calcul dans le shader |
| Uniformisation du teint | Effets → Peau → curseur **Even tone** | Calcul dans le shader |
| Peau trouvée par IA | Effets → Peau → **Skin found by** : Landmarks / AI parsing / Both | SegFormer face parsing |
| Masques de parties du visage plus fins | Masques → IA → **Face** → **Found by** + 3 nouvelles parties (oreilles, cou, lunettes) | SegFormer face parsing |
| Restauration de visage | Panneau IA → outils de retouche → **Restore Face** (pinceau) | CodeFormer |
| Défloutage | Panneau IA → outils de retouche → **Deblur** (pinceau) | NAFNet (bougé) / Restormer (mise au point) |
| Taches de capteur | Panneau IA → outils de retouche → **Dust Spots** (pinceau) | Détection et correction par calcul |
| Fils électriques | Panneau IA → outils de retouche → **Power Lines** (pinceau) | Détection et remplissage par calcul |
| Agrandissement ×4 | Détails → Super résolution → **×2 / ×4** | RealPLKSR de darktable |

L'interface est en anglais dans toutes les langues pour ces nouveaux libellés : c'est la règle du dépôt, les traductions viennent après.

Les modèles extraits de l'archive `onnxmodels.7z` (qui proviennent de Luminar Neo) n'ont **pas** été utilisés. Chaque fonction s'appuie sur un modèle publié ouvertement par ses auteurs, ou sur un calcul.

---

## 2. Les modèles et leur installation

Les modèles se trouvent dans le dossier des modèles de RapidRAW :
`%APPDATA%\io.github.CyberTimon.RapidRAW\models`

Il y a deux modes d'installation :

- **Automatique** : RapidRAW télécharge le fichier à la première utilisation et vérifie son empreinte SHA-256. L'interface affiche alors « téléchargement du modèle ».
- **Manuel** : aucun site n'héberge ce modèle au format ONNX. Le fichier doit être produit par le script indiqué, puis copié dans le dossier. S'il manque, RapidRAW affiche un message qui donne le nom du fichier et le dossier où le mettre.

| Modèle | Fichier | Taille | Installation | Source | Licence |
|---|---|---|---|---|---|
| FFHQ-Wrinkle U-Net | `ffhq_wrinkle_unet.onnx` | 289 Mo | Manuelle, [`scripts/export_wrinkle_onnx.py`](../scripts/export_wrinkle_onnx.py) | [rmsandu/ffhq-wrinkle-unet](https://huggingface.co/rmsandu/ffhq-wrinkle-unet) | CC BY-NC-SA 4.0 |
| Détecteur de défauts ABPN | `abpn_blemish_detection.onnx` | 72 Mo | Manuelle, [`scripts/export_blemish_onnx.py`](../scripts/export_blemish_onnx.py) | [ModelScope damo/cv_unet_skin_retouching_torch](https://www.modelscope.cn/models/damo/cv_unet_skin_retouching_torch) | Apache-2.0 |
| SegFormer-B5 face parsing | `face_parsing_segformer_b5.onnx` | 340 Mo | Automatique | [jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing) | Recherche / non commercial (CelebAMask-HQ) |
| CodeFormer | `codeformer.onnx` | 377 Mo | Automatique | [facefusion-assets](https://github.com/facefusion/facefusion-assets/releases/tag/models-3.0.0) | S-Lab 1.0 (non commercial) |
| RealPLKSR ×4 | `realplksr_x4_darktable.onnx` | 30 Mo | Automatique | [darktable-org/upscale-realplksr-onnx](https://huggingface.co/darktable-org/upscale-realplksr-onnx) | MIT |
| NAFNet GoPro (bougé) | `deblur_motion_nafnet.onnx` | 69 Mo | Manuelle, [`scripts/export_deblur_onnx.py`](../scripts/export_deblur_onnx.py) | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | MIT |
| Restormer (mise au point) | `deblur_defocus_restormer.onnx` | 107 Mo | Manuelle, même script | [swz30/Restormer](https://github.com/swz30/Restormer) | MIT |

**Sur ce PC, les quatre modèles à installation manuelle sont déjà en place** : rides, défauts, et les deux de défloutage. Les trois automatiques se téléchargeront à la première utilisation.

Chaque export a été comparé au modèle PyTorch d'origine sur la même entrée :

| Modèle | Écart maximal ONNX / PyTorch |
|---|---|
| Rides | 1,5 × 10⁻⁶ |
| Défauts | 1,4 × 10⁻⁷ |
| NAFNet | 1,1 × 10⁻⁴ |
| Restormer | 1,2 × 10⁻⁶ |

Les scripts d'export lisent les poids PyTorch en mode `weights_only=True` : seuls les tenseurs sont lus, aucun code contenu dans le fichier n'est exécuté.

Pour refaire un modèle manuel sur une autre machine, il faut un Python avec `torch`, `onnx`, `onnxruntime` et `numpy`, plus `spandrel` et `spandrel_extra_arches` pour le défloutage. Les commandes sont en tête de chaque script.

---

## 3. Outil Peau : nouvelles corrections

L'outil Peau (Effets → **Skin**) corrigeait déjà quatre types de défauts : boutons (Spots), rougeurs, brillance et rides, plus les cernes. Chaque correction s'appuie sur des cartes calculées par le GPU dans [`skin.wgsl`](../src-tauri/src/shaders/skin.wgsl). Le grain de la peau est remis par-dessus chaque correction.

### 3.1 Rides trouvées par IA

**Ce que ça fait.** Le modèle FFHQ-Wrinkle repère les rides sur chaque visage. Sa carte remplace la détection par courbure de la luminosité pour décider *où* se trouvent les rides. Le calcul existant garde deux rôles :

- trouver la peau de part et d'autre de chaque ride, pour la combler à ce niveau ;
- écarter les bords d'ombre et de narine, qui ne sont pas des rides.

**Utilisation.**

1. Effets → Skin.
2. Monter le curseur **Lines** au-dessus de 0.
3. Activer **Find lines with AI**.

La carte est calculée au premier clic (« Finding the lines »), puis enregistrée avec la retouche. Le curseur de détection **Lines** (section *Detection*) déplace le seuil de la carte.

**Implémentation.**

- Cadrage de chaque visage exactement comme les visages d'entraînement (alignement FFHQ à partir des 478 points MediaPipe : centres des yeux, coins de la bouche) : `ffhq_frame` dans [`face.rs`](../src-tauri/src/masks/face.rs).
- Analyse à 512 px, normalisation ImageNet. Un visage beaucoup plus grand est d'abord réduit avec un vrai filtre, pour éviter le crénelage : `face_tensor`.
- Recomposition en une carte de 2048 px de côté au maximum dans l'espace de l'image source : `compose_face_map` / `compose_wrinkle_map`. Commande Tauri : `generate_full_image_wrinkle_map`.
- Côté GPU, la carte occupe le 3ᵉ canal de la texture des régions du visage, car les six emplacements de texture étaient déjà pris. Le shader l'applique dans `sk_lines`, quand `lines_ai = 1`.

**À tester.**

- Visage de face avec rides du front et pattes d'oie, curseur Lines à 60 : comparer l'interrupteur coupé puis activé. Avec l'IA, les rides fines doivent être prises et les bords du nez et des lèvres épargnés.
- Le curseur de détection Lines doit faire varier la quantité de rides traitées.
- Visage incliné ou de trois quarts : la carte doit rester calée sur les rides.
- Supprimer temporairement `ffhq_wrinkle_unet.onnx` : l'activation doit afficher un message clair et l'interrupteur doit se recouper.
- Rouvrir la photo : la carte doit être relue sans nouveau calcul.

### 3.2 Boutons, grains de beauté, milium par IA

**Ce que ça fait.** Le détecteur de défauts du pipeline de retouche ABPN d'Alibaba, entraîné sur des portraits ordinaires et non sur des photos de dermatologie, marque les boutons, grains de beauté, milium et marques d'acné. Sa carte remplace la forme de la luminosité pour décider *où* sont les défauts. Le soin lui-même reste celui de l'outil : remplacement par l'anneau de peau propre autour, puis remise du grain.

**Utilisation.**

1. Effets → Skin.
2. Monter le curseur **Spots** au-dessus de 0.
3. Activer **Find spots with AI** (« Finding the spots » au premier clic).

Le curseur de détection **Spots** déplace le seuil. Le réglage par défaut reprend les seuils de la chaîne d'origine : 0,5 = défaut certain, 0,33 = rien.

**Implémentation.**

- Cadre carré de 1,5 × la taille du visage, droit, à 768 px, valeurs RVB entre −1 et 1 : `blemish_frame`, `run_blemishes` dans [`face.rs`](../src-tauri/src/masks/face.rs).
- Commande `generate_full_image_blemish_map`. Le chargement des modèles « manuels » (rides, défauts, défloutage) passe par une seule fonction, `get_or_init_manual_model` dans [`models.rs`](../src-tauri/src/ai/models.rs).
- Côté GPU, la carte occupe le 4ᵉ canal de la texture des régions. Le passage `sk_blob` la lit quand `spots_ai = 1` et la mélange au résultat de la détection par forme.

**À tester.**

- Portrait avec quelques boutons et grains de beauté, curseur Spots à 70 : comparer l'interrupteur coupé puis activé. L'IA doit prendre des défauts que la forme seule manquait, et ne pas toucher les pores.
- Faire varier le curseur de détection Spots.
- Vérifier que les sourcils, les cils et le bord des lèvres ne sont jamais « soignés ».

### 3.3 Taches pigmentaires

**Ce que ça fait.** Le curseur éclaircit les taches de soleil, taches de vieillesse, taches de rousseur et le masque de grossesse. Il ramène vers la peau voisine leur luminosité et leur teinte brune. Le grain qui traverse la tache est conservé : rien n'est recouvert, la tache cesse seulement d'être plus sombre et plus brune.

**Utilisation.** Effets → Skin → **Pigment spots**. Le curseur de détection **Pigment spots** règle à partir de quand une zone compte comme tache.

**Implémentation.** Fonction `sk_apply` dans [`skin.wgsl`](../src-tauri/src/shaders/skin.wgsl).

- La couleur est mesurée **rapportée à la luminosité** (chrominance divisée par la luminance). Une ombre est plus sombre mais garde sa teinte ; une tache est plus brune *pour sa luminosité*. Sans ce rapport, toutes les ombres du visage (orbites, dessous du nez) étaient prises pour des taches : le premier essai l'a montré, et cette version le corrige.
- Une zone est une tache si elle est à la fois plus brune (moins de bleu, plus de rouge pour sa luminosité) et plus sombre que la peau voisine, aux échelles « bouton » et « joue ».
- Nouveaux paramètres : `pigment`, `tone` et `thr_pigment`. Ils occupent les trois emplacements de remplissage de la structure `Params`, dont la taille ne change pas.

**À tester.**

- Visage avec taches de rousseur ou de soleil, curseur à 100 : les taches s'estompent, la texture reste.
- Visage très éclairé d'un côté : les ombres (orbites, nez, cou) ne doivent **pas** s'éclaircir.
- Faire varier le curseur de détection.

### 3.4 Uniformisation du teint

**Ce que ça fait.** Le curseur atténue les plaques de couleur : joue plus rouge que le front, zone jaunâtre, marbrures. Chaque plaque prend la couleur de la peau qui l'entoure, à luminosité égale. Le grain n'est pas touché : la peau est uniformisée, pas lissée. La luminosité reçoit une petite part de la correction, car à cette échelle elle dessine surtout le relief du visage.

**Utilisation.** Effets → Skin → **Even tone**.

**Implémentation.** Dans `sk_apply`, avec la même mesure de couleur rapportée à la luminosité.

**À tester.**

- Peau marbrée ou rougeurs diffuses, curseur à 100 : le teint s'uniformise sans effet « plastique ».
- Pas de bascule de teinte globale : le visage ne doit pas virer à l'orange ou au gris.
- Combiner avec **Redness** : les deux doivent se compléter.

Un test GPU automatique (`pigment_lifts_a_brown_patch_and_leaves_a_shadow`, dans `skin.rs`) vérifie sur une image de synthèse qu'une tache brune est éclaircie et qu'une ombre de même profondeur ne l'est pas. Il n'est pas exécuté par la CI, qui compile les tests sans les lancer. Voir la [section 8](#8-ce-qui-a-été-vérifié-et-comment).

---

## 4. Segmentation fine du visage

**Ce que ça fait.** SegFormer-B5, entraîné sur CelebAMask-HQ, attribue à chaque pixel du visage une classe parmi 19 : peau, nez, yeux, sourcils, oreilles, bouche, lèvres, cheveux, cou, lunettes, chapeau, boucles d'oreilles, vêtements, etc. Les contours sont donc ceux de l'image elle-même, et non des formes tracées entre les points du visage. Le gain est net sur le bord des cheveux, la barbe, les lunettes, les oreilles et le cou.

Comme demandé, les deux méthodes existent **côte à côte** :

- **Landmarks** : la méthode existante, inchangée, par défaut.
- **AI parsing** : la segmentation IA seule.
- **Both** : la moyenne des deux. Les zones où elles s'accordent sont sûres ; celles qu'une seule voit sont à moitié prises.

**Utilisation.**

- **Masques** : Masques → IA → **Face**. Sous l'illustration, le nouveau choix **Found by** : Landmarks / AI parsing / Both. Trois nouvelles parties, que seule la segmentation IA sait trouver : **Ears**, **Neck**, **Glasses**. Le masque se recalcule à chaque changement.
- **Outil Peau** : Effets → Skin → **Skin found by** : Landmarks / AI parsing / Both. La carte de peau sur laquelle travaille l'outil est recalculée.

**Implémentation.**

- Dans [`face.rs`](../src-tauri/src/masks/face.rs) :
  - `FaceMethod`, `FacePart::parsed_classes` (correspondance partie → classes) ;
  - `run_parser` : softmax sur les 19 classes, réponse à 128 px pour une entrée de 512 ;
  - `parse_faces`, `parsed_share` ;
  - `face_parts_mask` réécrite avec deux sources, `from_landmarks` et `from_parser`.
- Chaque partie issue de la segmentation est affinée par le filtre guidé déjà utilisé pour les masques. L'iris n'existe pas dans les classes de SegFormer : c'est le disque des points MediaPipe, limité aux yeux vus par la segmentation. Les dents restent détectées par couleur, dans la bouche vue par la segmentation.
- Dans [`commands.rs`](../src-tauri/src/ai/commands.rs) : la segmentation est calculée une fois par analyse de visage et gardée en cache (`cached_face_parsing`). Le modèle n'est chargé que si la méthode ou les parties choisies l'exigent (`face_mask_needs_parser`).
- [`jobs.rs`](../src-tauri/src/ai/jobs.rs) : les tâches en arrière-plan (masques recalculés sur d'autres images, copier-coller, export par lot) transmettent la méthode et chargent le modèle si besoin.

**À tester.**

1. Masque Face, parties *Face Skin* : comparer Landmarks, AI parsing et Both sur un visage avec mèches sur le front, barbe ou lunettes. AI parsing doit suivre les mèches et exclure les lunettes.
2. Parties *Ears*, *Neck* et *Glasses* : avec la méthode Landmarks aussi, elles doivent fonctionner (le modèle est alors chargé automatiquement).
3. *Lips*, *Eyebrows*, *Eye Whites*, *Iris*, *Teeth*, *Nose* et *Hair* avec AI parsing.
4. Premier usage : le modèle de 340 Mo se télécharge (vérifier l'affichage), puis les usages suivants sont immédiats.
5. Outil Peau → Skin found by → AI parsing : le front sous une frange, le bord de la barbe ou la monture de lunettes ne doivent plus être traités comme de la peau.
6. Photo avec deux ou trois visages, dont un petit.
7. Copier les réglages d'une photo à une autre (masque Face en AI parsing) : le masque doit être recalculé sur la nouvelle image avec la même méthode.

---

## 5. Outils de retouche au pinceau

Quatre nouveaux outils rejoignent **Liquify** et **Retouch** dans les outils de retouche du panneau IA. Ils fonctionnent tous de la même manière :

1. Choisir l'outil.
2. Peindre la zone à traiter.
3. Au relâchement du pinceau, le calcul part et produit un *patch* : une couche de pixels, limitée à la zone peinte, posée sur l'image avant le développement.

Changer un réglage de l'outil relance le calcul. Le patch se masque, se supprime ou s'efface à la gomme comme les autres.

Côté code :

- Interface : un même assistant, `isRepairTool()` dans [`types/masks.ts`](../src/types/masks.ts), couvre les quatre outils dans le canevas, l'éditeur et le panneau IA. Les réglages sont dans `SUB_MASK_CONFIG` de [`AIPanel.tsx`](../src/components/panel/right/AIPanel.tsx).
- Backend : chaque outil a sa commande dans [`inpainting.rs`](../src-tauri/src/retouch/inpainting.rs). Elles partagent une structure `Brushed` qui regroupe l'image source, le masque peint, son cadre et l'encodage du patch.

### 5.1 Restore Face (CodeFormer)

**Ce que ça fait.** CodeFormer reconstruit tous les visages que le coup de pinceau touche. Il convient aux visages flous, bruités, très compressés, ou aux vieilles photos.

**Utilisation.** Outil **Restore Face**, puis peindre sur le visage. Deux réglages :

- **Fidelity** (50 par défaut) : proche de 0, CodeFormer reconstruit davantage ce qu'il sait d'un visage (plus net, mais moins fidèle). Proche de 100, il reste proche de ce visage.
- **Strength** (80 par défaut) : la part du visage restauré posée sur l'original.

**Implémentation.** Fichier [`face_restore.rs`](../src-tauri/src/retouch/face_restore.rs) :

1. Cinq points pris sur les 478 points MediaPipe (centres des yeux, bout du nez, coins de la bouche) : `alignment_points`.
2. Similitude (rotation, échelle, translation) ajustée par moindres carrés sur le gabarit FFHQ 512 que CodeFormer attend.
3. Extraction du visage avec sur-échantillonnage s'il est grand, puis passage dans CodeFormer.
4. Remise en place par la transformation inverse, avec un fondu sur les bords du carré, puis limitation à la zone peinte.

CodeFormer tourne sur le processeur, pour la robustesse (environ 2 s par visage).

**À tester.**

- Visage flou ou petit dans une photo de groupe : peindre dessus avec Fidelity 50 et Strength 80.
- Faire varier Fidelity entre 0 et 100 : à 0, plus net mais la ressemblance peut changer ; à 100, plus fidèle.
- Visage tourné ou incliné : pas de décalage ni de « double visage » à la jonction.
- Peindre sur deux visages en un seul trait : les deux doivent être restaurés.
- Peindre une zone sans visage : un message « No faces found » doit s'afficher.
- Bords du patch invisibles à 100 % de zoom.

### 5.2 Deblur (NAFNet / Restormer)

**Ce que ça fait.** Le flou est réduit dans la zone peinte. Deux types de flou sont proposés :

- **Camera shake** (bougé) : NAFNet, entraîné sur le jeu GoPro.
- **Out of focus** (mise au point manquée) : Restormer, entraîné sur DPDD.

**Utilisation.** Outil **Deblur**, choisir *Blur* : Camera shake ou Out of focus, puis peindre. Deux réglages :

- **Blur Size** (0 par défaut) : la largeur du flou. 0 correspond à un flou d'un ou deux pixels, traité en pleine taille. Plus la valeur monte, plus la zone est traitée sur une copie réduite (jusqu'au quart à 100), où le flou retrouve la taille vue à l'entraînement. Seule la correction est remise à l'échelle et ajoutée, ce qui préserve le grain fin de l'image.
- **Strength** (80 par défaut) : la part de la correction conservée.

**Implémentation.** `deblur` dans [`repair.rs`](../src-tauri/src/retouch/repair.rs). Le traitement se fait par tuiles de 384 px avec recouvrement (`run_ai_deblur` dans [`tiling.rs`](../src-tauri/src/ai/tiling.rs)) : NAFNet regarde toute sa tuile, et cette taille correspond à celle de ses tests. Les modèles sont du type « manuel ».

**À tester.**

- Photo légèrement bougée : Camera shake, Blur Size 0, puis 30 et 60. Choisir le réglage le plus net sans artefacts.
- Photo à la mise au point ratée : Out of focus, même démarche. Restormer est lent sur processeur (environ 10 s pour 512 × 512) : tester d'abord sur une petite zone.
- Grande zone, par exemple toute l'image : mesurer le temps de calcul et vérifier qu'aucune jointure de tuile n'est visible.
- Strength à 100 puis à 40.

### 5.3 Dust Spots (taches de capteur)

**Ce que ça fait.** Les taches qu'une poussière sur le capteur laisse sur chaque image sont retirées. Ce sont des disques gris, doux, surtout visibles à petite ouverture dans les zones unies (ciel, mur). Une tache est **divisée** par l'assombrissement qu'elle cause, ce qui fait réapparaître ce qu'il y avait dessous : la texture, les dégradés et les nuages fins sont conservés.

**Utilisation.** Outil **Dust Spots**, puis peindre largement sur le ciel ou la zone touchée (on peut tout couvrir d'un geste). Deux réglages :

- **Spot Size** (30 px par défaut) : le diamètre maximal d'une tache.
- **Sensitivity** (50 par défaut) : à partir de quelle discrétion une tache compte.

**Implémentation.** `remove_dust` dans [`repair.rs`](../src-tauri/src/retouch/repair.rs), sans réseau : aucun modèle ouvert n'est assez bon, le meilleur trouvé a un rappel d'environ 30 %.

1. Travail sur une copie réduite où une tache fait environ une douzaine de pixels.
2. Fond estimé par fermeture morphologique puis flou, pour la luminance et pour chaque couleur.
3. Assombrissement relatif = 1 − valeur / fond.
4. Seules les zones calmes comptent : la variation locale doit rester faible.
5. Chaque composante gardée doit être ronde (allongement ≤ 2), de taille compatible, et **neutre** (aussi sombre en rouge, vert et bleu), ce qui écarte un oiseau ou une feuille.
6. Correction par division de la lumière par la transmission estimée, avec bords adoucis.

Le prototype Python a été corrigé et validé : trois taches de synthèse ramenées à 1 à 3 niveaux de la valeur du ciel.

**À tester.**

- Photo de ciel à f/11–f/22 avec poussières visibles : peindre tout le ciel. Les taches doivent disparaître et le ciel rester intact.
- Monter Sensitivity pour les taches très pâles, et Spot Size pour les grosses taches floues (ouverture moyenne).
- **Faux positifs** : oiseaux, avions, étoiles, petits nuages sombres ne doivent pas être effacés. Un oiseau coloré est écarté par le test de neutralité, mais un petit oiseau gris peut passer : c'est à vérifier.
- Zone texturée (feuillage) peinte par erreur : rien ne doit se produire.

### 5.4 Power Lines (fils électriques)

**Ce que ça fait.** Les fils et câbles qui traversent un ciel sont détectés dans la zone peinte et remplis à partir du ciel de part et d'autre.

**Utilisation.** Outil **Power Lines**, puis peindre grossièrement sur la zone des fils : pas besoin de suivre chaque fil, la détection les trouve. Deux réglages :

- **Wire Width** (6 px par défaut) : la largeur maximale d'un fil.
- **Sensitivity** (50 par défaut) : à partir de quelle discrétion une ligne compte.

**Implémentation.** `remove_wires` dans [`repair.rs`](../src-tauri/src/retouch/repair.rs), sans réseau : le seul modèle ouvert spécialisé est jeune et peu entraîné, et celui d'Adobe n'est pas publié.

1. Mesure de Frangi (crête sombre : forte courbure en travers, presque nulle le long) sur le logarithme de la luminosité, à trois échelles adaptées à la largeur choisie.
2. Seuil à hystérésis, puis on ne garde que les composantes **longues** (au moins 4 % de la zone, et au moins six fois la largeur) et **minces** (allongement ≥ 6). Les branches et les brindilles sont ainsi écartées.
3. Le trou est élargi à la largeur du fil plus une marge, puis rempli par *push-pull* (pyramide de moyennes des pixels connus), qui donne un remplissage lisse et sans couture, adapté à un ciel.

**À tester.**

- Ciel avec lignes à haute tension, puis câbles fins : peindre la zone et vérifier la disparition sans trace.
- Fil devant un nuage net : le remplissage lisse peut adoucir légèrement la texture du nuage sur la largeur du fil. C'est attendu ; à évaluer.
- Fil épais ou proche : monter Wire Width.
- Arbres ou bord de toit dans la zone peinte : ils ne doivent pas être effacés. Une antenne droite et fine **le sera** : c'est un cas limite connu.

---

## 6. Agrandissement ×4

**Ce que ça fait.** La super résolution propose maintenant **×2** (modèle SPAN existant, inchangé) ou **×4**. Le ×4 utilise le modèle RealPLKSR de darktable, conçu pour la photographie. Sur une image test, il rend nettement mieux les cheveux et la texture de peau qu'un bicubique.

**Utilisation.** Détails → Super résolution → activer → **×2 / ×4**. Le curseur Detail s'applique aux deux.

**Limite.** Le rendu se fait dans des textures GPU, limitées à 16 384 px sur la plupart des cartes. Une image dont le résultat ×4 dépasserait cette taille (côté long au-delà de 4 096 px) est agrandie ×2 à la place, et une note dans l'interface l'indique. Un RAW de 24 Mpx (6 000 px) sera donc agrandi ×2 : pour du ×4, recadrer ou réduire d'abord.

**Implémentation.** Dans [`upscale.rs`](../src-tauri/src/ai/upscale.rs) :

- Le facteur est maintenant un paramètre : `requested_factor`, et `factor` qui tient compte de la taille.
- L'interpolation de la lumière hors gamme (`upsample_add`) et le passage de détail (`DetailField`), auparavant écrits pour ×2 seulement, acceptent n'importe quel facteur. Pour ×2, les poids d'interpolation restent identiques aux anciens.
- Les résultats ×2 et ×4 sont mis en cache séparément.
- Le modèle ×4 a une entrée fixe de 256 px : il est traité par tuiles de 256 avec marge et recouvrement de 16 px (`UPSCALE4_TILE` dans `tiling.rs`). Son chargement est assuré par `get_or_init_upscale4_model`.

**À tester.**

- Image d'environ 3 000 px de côté long, ×2 puis ×4 : comparer à 100 %. Vérifier les dimensions exportées (×4 = quatre fois la largeur).
- Image de plus de 4 096 px avec ×4 : le résultat doit être en ×2, sans erreur.
- **Non-régression du ×2** : le résultat doit être le même qu'avant cette branche, puisque les calculs ×2 sont restés identiques.
- Combiné avec le débruitage IA.
- Temps de calcul et mémoire : à 4 096 px en ×4, la sortie fait environ 180 Mpx, ce qui demande plusieurs Go de mémoire.

---

## 7. Liste de tests, dans l'ordre

1. **Compiler et lancer** la branche (`npm run tauri dev`). Vérifier qu'il n'y a pas d'erreur au démarrage.
2. **Non-régression** :
   - ouvrir une ancienne retouche qui utilise l'outil Peau : le rendu doit être identique ;
   - ouvrir un ancien masque Face : méthode Landmarks par défaut ;
   - agrandissement ×2 : identique à avant.
3. **Outil Peau** :
   - Pigment spots, puis Even tone ([3.3](#33-taches-pigmentaires), [3.4](#34-uniformisation-du-teint)) ;
   - Find lines with AI ([3.1](#31-rides-trouvées-par-ia)) ;
   - Find spots with AI ([3.2](#32-boutons-grains-de-beauté-milium-par-ia)) ;
   - Skin found by → AI parsing ([section 4](#4-segmentation-fine-du-visage)).
4. **Masques Face** : les trois méthodes et les trois nouvelles parties ([section 4](#4-segmentation-fine-du-visage)).
5. **Restore Face** ([5.1](#51-restore-face-codeformer)) : le premier usage télécharge 377 Mo.
6. **Deblur** : bougé, puis mise au point ([5.2](#52-deblur-nafnet--restormer)).
7. **Dust Spots** sur un ciel ([5.3](#53-dust-spots-taches-de-capteur)).
8. **Power Lines** sur un ciel ([5.4](#54-power-lines-fils-électriques)).
9. **×4** ([section 6](#6-agrandissement-4)).
10. **Export** d'une image utilisant toutes ces fonctions, au format JPEG puis TIFF. Vérifier que le fichier exporté correspond à l'aperçu.
11. **GPU / processeur** : refaire quelques tests avec l'accélération GPU coupée dans les réglages. Les modèles doivent aussi fonctionner sur processeur, plus lentement.
12. Facultatif, sur une machine avec Rust :
    ```
    cargo test --manifest-path src-tauri/Cargo.toml -- skin
    cargo test --manifest-path src-tauri/Cargo.toml -- face_restore repair
    ```
    Le premier lance les tests GPU de l'outil Peau, le second les tests de la restauration et des réparations.

Pour chaque problème, noter : la photo (taille, boîtier), l'outil, les réglages, ce qui était attendu et ce qui s'est passé, et le message d'erreur éventuel (journal de RapidRAW).

---

## 8. Ce qui a été vérifié, et comment

| Vérification | Résultat |
|---|---|
| `cargo clippy --all-targets -- -D warnings` (CI GitHub) | Passe sur chaque étape, jusqu'aux réparations au pinceau comprises : tout le code Rust compile, tests compris, sans aucun avertissement. |
| `cargo fmt --check` (CI) | Les écarts relevés ont été appliqués à chaque étape. |
| Compilation complète de l'application (`tauri build`, CI) | Lancée par la PR à chaque poussée ; les exécutions intermédiaires ont été annulées pour libérer la file d'attente. Aucun exécutable n'est publié. |
| Shader `skin.wgsl` | Validé par naga (le compilateur de wgpu) sur la RTX A2000 : les 8 points d'entrée compilent. |
| TypeScript (`tsc --noEmit`) | Passe. |
| ESLint | Aucune nouvelle erreur sur les lignes modifiées. Les erreurs `no-explicit-any` qui restent existaient avant. |
| Prettier | Passe. |
| Traductions (`npm run i18n:check`) | Passe : 17 langues, formes plurielles comprises. |
| Exports ONNX | Comparés à PyTorch (écarts en [section 2](#2-les-modèles-et-leur-installation)). |
| Modèles sur photos réelles (Python, onnxruntime) | Segmentation, CodeFormer, défauts, rides, ×4, NAFNet, Restormer : sorties correctes sur des portraits d'exemple. |
| Taches pigmentaires et teint | Copie numpy du calcul du shader, essayée sur trois portraits ; la première version (ombres prises pour des taches) a été corrigée. |
| Taches de capteur et fils | Copie numpy des algorithmes sur des ciels de synthèse : taches corrigées à 1–3 niveaux du ciel, fil entièrement rempli. |
| **Fonctionnement dans RapidRAW** | **Non testé.** |
| Tests GPU Rust (`skin.rs`) | Compilés par clippy, **non exécutés**. |
| DirectML (GPU Windows) avec les nouveaux modèles | **Non testé.** CodeFormer est forcé sur le processeur. |

---

## 9. Limites connues et points d'attention

- **Licences non commerciales** : FFHQ-Wrinkle, SegFormer face parsing (CelebAMask-HQ) et CodeFormer. Cela convient à un usage personnel, mais il ne faut ni redistribuer ces fichiers ni fusionner cette branche dans le dépôt public d'origine sans retirer ces modèles.
- **Taille des téléchargements** : SegFormer (340 Mo) et CodeFormer (377 Mo) sont lourds. La version quantifiée de SegFormer (89 Mo) existe, mais n'a pas été testée avec DirectML.
- **Restormer est lent** sur processeur : environ 10 s pour 512 × 512. Sur GPU, ce sera plus rapide, si DirectML accepte le modèle (non vérifié).
- **×4** : limité aux images de 4 096 px de côté long au plus. La mémoire nécessaire est importante.
- **Dust Spots / Power Lines** sont des méthodes par calcul :
  - une petite forme grise et ronde peut être prise pour une tache ;
  - une antenne droite et fine peut être prise pour un fil ;
  - le remplissage d'un fil est lisse et peut adoucir un nuage très texturé sur la largeur du fil.
- **Restore Face** : CodeFormer peut modifier légèrement les traits à faible Fidelity. Utiliser Strength pour doser.
- **Segmentation fine** : SegFormer a appris sur des visages de célébrités cadrés de face. Ses résultats sur des visages très de profil ou très petits restent à vérifier.
- La **carte de rides ou de défauts** est calculée une fois et enregistrée avec la retouche. Après une modification de la géométrie de l'image (recadrage, rotation), elle est reprojetée comme les autres cartes de l'outil Peau. Si l'image source change, il faut couper puis réactiver l'interrupteur.

---

## 10. Fichiers modifiés

**Backend (Rust).**

| Fichier | Contenu |
|---|---|
| `src-tauri/src/masks/face.rs` | Cadrage FFHQ, cartes de rides et de défauts, segmentation et masques à trois méthodes, points d'alignement |
| `src-tauri/src/effects/skin.rs`, `src-tauri/src/shaders/skin.wgsl` | Rides IA, défauts IA, taches pigmentaires, teint |
| `src-tauri/src/ai/models.rs` | Nouveaux modèles (téléchargés ou manuels), chargeur des modèles manuels, modèle ×4 |
| `src-tauri/src/ai/commands.rs` | Commandes de cartes (rides, défauts), méthode des masques de visage, cache de segmentation |
| `src-tauri/src/ai/jobs.rs` | Méthode des masques de visage dans les tâches en arrière-plan |
| `src-tauri/src/ai/processing.rs` | Champ `method` des paramètres de masque de visage |
| `src-tauri/src/ai/upscale.rs`, `src-tauri/src/ai/tiling.rs` | Facteur ×2 / ×4, tuiles du ×4 et du défloutage |
| `src-tauri/src/retouch/face_restore.rs` (nouveau) | CodeFormer |
| `src-tauri/src/retouch/repair.rs` (nouveau) | Défloutage, taches de capteur, fils |
| `src-tauri/src/retouch/inpainting.rs` | Commandes des quatre outils au pinceau |
| `src-tauri/src/masks/generation.rs` | Les nouveaux outils sont des masques au pinceau |
| `src-tauri/src/render/cache_utils.rs`, `src-tauri/src/app/settings.rs`, `src-tauri/src/export/geometry.rs`, `src-tauri/src/lib.rs` | Clés de cache, réglages, facteur d'export, enregistrement des commandes |

**Interface (TypeScript).**

| Fichier | Contenu |
|---|---|
| `src/components/adjustments/Effects.tsx` | Outil Peau : nouveaux curseurs et interrupteurs, choix de la méthode |
| `src/components/adjustments/Details.tsx` | Choix ×2 / ×4 |
| `src/components/panel/right/MasksPanel.tsx`, `Masks.tsx`, `FacePartsIllustration.tsx` | Méthode et nouvelles parties du masque Face |
| `src/components/panel/right/AIPanel.tsx`, `src/components/panel/editor/ImageCanvas.tsx`, `src/components/panel/Editor.tsx`, `src/hooks/useAiMasking.ts`, `src/types/masks.ts`, `src/utils/maskNames.ts` | Les quatre outils au pinceau |
| `src/utils/adjustments.ts`, `src/utils/defaultAdjustments.json` | Nouveaux réglages et leurs valeurs par défaut |
| `src/i18n/locales/*.json` | Libellés (17 langues) |

**Scripts.** `scripts/export_wrinkle_onnx.py`, `scripts/export_blemish_onnx.py`, `scripts/export_deblur_onnx.py`.

**Nouveaux réglages enregistrés dans les retouches** (tous avec une valeur par défaut neutre, donc les anciennes retouches ne changent pas) :

- Peau : `skinLinesAi`, `skinLinesAiMap`, `skinSpotsAi`, `skinSpotsAiMap`, `skinPigment`, `skinPigmentSensitivity`, `skinTone`, `skinFaceMethod`.
- Agrandissement : `aiUpscaleFactor`.
- Masques Face : paramètre `method`.

---

## 11. Licences

| Composant | Licence | Conséquence |
|---|---|---|
| FFHQ-Wrinkle U-Net | CC BY-NC-SA 4.0 | Usage non commercial, ne pas redistribuer avec l'application |
| SegFormer face parsing (CelebAMask-HQ) | Recherche / non commercial | Idem |
| CodeFormer | S-Lab License 1.0 | Non commercial |
| Détecteur ABPN (Alibaba) | Apache-2.0 | Libre |
| RealPLKSR ×4 (darktable) | MIT | Libre |
| NAFNet, Restormer | MIT | Libre |
| Code ajouté à RapidRAW | Licence du dépôt (AGPL-3.0) | — |

Les poids des modèles ne sont pas dans le dépôt Git : ils sont téléchargés ou installés à part.
