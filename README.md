# 🛰️ Reconnaissance d'Images Satellitaires avec PyTorch et Flask

Ce projet est une application de reconnaissance d'images satellitaires basée sur **PyTorch** et **Flask**.  
Il permet de classifier des images Sentinel-2 en différentes catégories (culture, forêt, zone urbaine, etc.) en utilisant un modèle **CNN** entraîné sur le dataset **EuroSAT**.  

---

## 📌 Fonctionnalités  

✅ **Upload d'une image** satellite via une interface web  
✅ **Classification automatique** de l'image dans l'une des 10 classes du dataset EuroSAT  
✅ **Affichage de la prédiction** avec une interface moderne  
✅ **Entraînement d'un modèle CNN** avec PyTorch  
✅ **Sauvegarde et réutilisation du modèle**  

---

## 🚀 Installation et Exécution  

### 1️⃣ Cloner le projet  
```bash
git clone https://github.com/seydousy/Reconnaissance_Images_Satellitaires.git
cd Reconnaissance_Images_Satellitaires
```
---
```
### 2️⃣ **Installer les dépendances**
Assure-toi d'avoir Python 3.8+ et exécute :
```
```bash
pip install -r requirements.txt```
---
### 3️⃣ Lancer l'application Flask
```
python app.py
```
---
### 4️⃣ Ouvrir dans le navigateur
```
Accède à l'application via :
👉 http://127.0.0.1:5000
```

