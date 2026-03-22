# 🌀 Rosetta BGE: The Monster Brain 🧠

Rosetta est un dénoiseur sémantique basé sur la **Diffusion Transformer**, conçu pour transformer les vecteurs latents de **BGE-small-en-v1.5** (384d) en texte cohérent de manière non-autorégressive.

## 🏗️ Architecture: Rosetta-Monster (v5.2)
Le modèle actuel, surnommé le **"Monster Brain"**, est une version survitaminée de l'architecture originale :
- **Paramètres :** ~70M
- **Dimension Latente (d_model) :** 1024
- **Profondeur :** Dual-Recursive Transformer (12 layers effectives via 6 cycles de récursion).
- **Conditionnement :** Projection BGE vers 4 "Guide Tokens" dynamiques.
- **Vitesse de pointe :** Entraîné sur **RTX 5080** avec `torch.compile` (standard mode).

## 🚀 Percée : Training via Sliding Window (Fragments)
La grande innovation de la v5.2 est l'intégration de la **Sliding Window Augmentation** pendant l'entraînement :
- **10% des batchs** sont transformés en fragments de texte aléatoires (de 1 à 16 tokens).
- **Recalcul BGE :** L'embedding est recalculé en temps réel pour chaque fragment.
- **EOS Mastery :** Rosetta a appris à utiliser le token `</s>` (EOS) pour s'arrêter dès que le concept est exprimé, permettant enfin le décodage chirurgical de **mots isolés**.

## 🧪 Le Laboratoire d'Algèbre Latente
Rosetta démontre que l'espace BGE est un espace de calcul conceptuel :
- **Genre :** `BGE("Man")` vers `BGE("Woman")` bascule à $\alpha=0.50$.
- **Evolution :** `BGE("Boy") + BGE("Old")` $\rightarrow$ **"Old boy"**.
- **Localisation :** `Paris - France + Japan` $\rightarrow$ **"Tokyo"** (Weight 1.2).
- **Narrative LERP :** Transition fluide entre "Voiture rapide" et "Débris" générant spontanément le concept d'**"Accident"**.

## 🛠️ Laboratory Scripts
Le dépôt contient plusieurs outils pour explorer et diagnostiquer le modèle :
- `surgical_test.py` : Tests de précision (Property Swapping, Multi-Attribute Injection).
- `latent_walk.py` : Exploration par marche aléatoire autour d'une phrase cible.
- `latent_dark_side.py` : Décodage de 100 vecteurs aléatoires (Archéologie Latente).
- `latent_additive.py` : Tests d'additivité, de soustraction et d'extraction d'archétypes.
- `fix_ckpt.py` : Utilitaire pour corriger ou migrer les checkpoints entre epochs.

## 🚜 Training Status
- **Dataset :** Holy Trinity (Fineweb-Edu, Wikitext) - 20M segments.
- **Batch Size :** 1024 (Effective via 2x Gradient Accumulation).
- **Optimizer :** AdamW + OneCycleLR (Peak at 5e-4).
- **Accuracy :** Stable à ~70% sur le texte complexe, pics à 92% sur les fragments courts.

---
*Propulsé par la forge Rosetta et l'intuition de la recherche latente.*
