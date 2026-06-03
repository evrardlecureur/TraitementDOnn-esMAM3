import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
from math import pi

# ==============================================================================
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES (LECTURE 2)
# ==============================================================================

df = pd.read_csv("df_arabica_clean.csv")

# Harmonisation des méthodes de traitement
processing_mapping = {
    "Double Anaerobic Washed": "Washed / Wet",
    "Semi Washed": "Washed / Wet",
    "Honey,Mossto": "Pulped natural / honey",
    "Double Carbonic Maceration / Natural": "Natural / Dry",
    "Wet Hulling": "Washed / Wet",
    "Anaerobico 1000h": "Washed / Wet",
    "SEMI-LAVADO": "Natural / Dry"
}
df['Processing Method'] = df['Processing Method'].replace(processing_mapping).fillna("Washed / Wet")

# Correction de l'ID 99 (Pieds -> Mètres pour le Guatemala) et des autres altitudes
df.loc[df['ID'] == 99, 'Altitude'] = 5273 / 3.281  
df.loc[df['ID'] == 105, 'Altitude'] = 1800  
df.loc[df['ID'] == 180, 'Altitude'] = 1400  

def clean_altitude_range(range_value):
    if isinstance(range_value, str):
        range_value = range_value.replace(" ", "")
        if '-' in range_value:
            try:
                start, end = range_value.split('-')
                return (int(start) + int(end)) / 2
            except ValueError:
                return np.nan
        else:
            try:
                return int(range_value)
            except ValueError:
                return np.nan
    return range_value

df['Altitude'] = df['Altitude'].apply(clean_altitude_range)
df.loc[df['Altitude'] > 4000, 'Altitude'] = np.nan

# Calcul de l'Âge du Café (Jours écoulés entre la récolte et l'expiration)
df['Harvest Year'] = df['Harvest Year'].astype(str).str.split('/').str[0].str.strip()
df['Harvest Year'] = pd.to_datetime(df['Harvest Year'], format='%Y', errors='coerce')
df['Expiration'] = df['Expiration'].apply(lambda x: parser.parse(x) if pd.notnull(x) else np.nan)
df['Coffee Age'] = (df['Expiration'] - df['Harvest Year']).dt.days

# Création des tranches d'altitude (Binned Feature)
bins = [0, 1200, 1500, 1800, 4000]
labels = ['< 1200m', '1200m - 1500m', '1500m - 1800m', '> 1800m']
df['Altitude_Range'] = pd.cut(df['Altitude'], bins=bins, labels=labels)

# Sélection rigoureuse des colonnes à supprimer (on garde les défauts pour la corrélation)
columns_to_drop = [
    'ID','ICO Number','Owner','Region','Certification Contact','Certification Address',
    'Farm Name',"Lot Number","Mill","Producer",'Company','Expiration', 'Harvest Year',
    "Unnamed: 0",'Number of Bags','Bag Weight','In-Country Partner','Grading Date',
    'Status','Defects','Uniformity','Clean Cup','Sweetness','Certification Body'
]
existing_cols_to_drop = [c for c in columns_to_drop if c in df.columns]
df.drop(existing_cols_to_drop, axis=1, inplace=True)

# Sauvegarde de la base propre
df.to_csv("df_arabica_fully_cleaned_corrected.csv", index=False)


# ==============================================================================
# 2. VISUALISATION DES DONNÉES ET STORYTELLING (LES 4 GRAPHIQUES)
# ==============================================================================

sns.set_theme(style="whitegrid")

# Filtrage pour les graphiques croisés
process_counts = df['Processing Method'].value_counts()
valid_methods = process_counts[process_counts > 1].index
df_filtered = df[df['Processing Method'].isin(valid_methods)]

# ------------------------------------------------------------------------------
# GRAPHIQUE 1 : Matrice de Corrélation Élargie
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 8))
numeric_cols = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 
                'Total Cup Points', 'Altitude', 'Coffee Age', 'Quakers', 'Category Two Defects']
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
            vmin=-0.4, vmax=1, square=True, linewidths=.5)
plt.title("1. Matrice de Corrélation des Variables Numériques et Physiques", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("1_Matrice_de_correlation.png")
plt.show()


# ------------------------------------------------------------------------------
# GRAPHIQUE 2 : Boxplot + Stripplot (Distribution des scores par Altitude & Process)
# ------------------------------------------------------------------------------
plt.figure(figsize=(11, 6))
ax = sns.boxplot(data=df_filtered.dropna(subset=['Altitude_Range']), 
                 x='Altitude_Range', y='Total Cup Points', hue='Processing Method', 
                 palette="muted", boxprops=dict(alpha=0.4), showfliers=False)

sns.stripplot(data=df_filtered.dropna(subset=['Altitude_Range']), 
              x='Altitude_Range', y='Total Cup Points', hue='Processing Method', 
              dodge=True, palette="muted", alpha=0.8, linewidth=0.8, jitter=True, ax=ax)

# Nettoyage de la légende pour éviter les doublons de labels
handles, labels_leg = ax.get_legend_handles_labels()
plt.legend(handles[:3], labels_leg[:3], bbox_to_anchor=(1.05, 1), loc='upper left', title="Processing Method")

plt.title("2. Distribution Réelle des Scores par Tranche d'Altitude et de Traitement", fontsize=14)
plt.xlabel("Tranche d'Altitude", fontsize=12)
plt.ylabel("Score Final (Total Cup Points)", fontsize=12)
plt.tight_layout()
plt.savefig("2_Distribution_scores_altitude.png")
plt.show()


# ------------------------------------------------------------------------------
# GRAPHIQUE 3 : Radar Chart (Profil Gustatif Moyen par Variété)
# ------------------------------------------------------------------------------
df['Variety'] = df['Variety'].fillna('Unknown')
top_varieties = ['Gesha', 'Caturra', 'Typica']
df_variety_filtered = df[df['Variety'].isin(top_varieties)]

categories = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1] 

plt.figure(figsize=(8, 8), dpi=100)
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2) 
ax.set_theta_direction(-1) 

plt.xticks(angles[:-1], categories, size=11, color='darkslategray', fontweight='bold')
ax.set_rlabel_position(30)
plt.yticks([7.4, 7.6, 7.8, 8.0, 8.2], ["7.4", "7.6", "7.8", "8.0", "8.2"], color="grey", size=9)
plt.ylim(7.3, 8.3)
ax.grid(color='#CCCCCC', linestyle='--', linewidth=1)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, variety in enumerate(top_varieties):
    values = df_variety_filtered[df_variety_filtered['Variety'] == variety][categories].mean().values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=variety, 
            color=colors[idx], marker='o', markersize=7, markerfacecolor='white', markeredgewidth=2)
    ax.fill(angles, values, color=colors[idx], alpha=0.08)

plt.title("3. Profil Gustatif Moyen par Variété de Café (Top Variétés)", size=14, fontweight='bold', y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), frameon=False, fontsize=11)
plt.tight_layout()
plt.savefig("3_Graphique_radar.png")
plt.show()


# ------------------------------------------------------------------------------
# ALTERNATIVE GRAPHIQUE 4 : Droite de Régression Linéaire (Les Défauts vs Score Final)
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
# Idéal pour la Lecture 4 : montre une relation linéaire négative claire et exploitable
sns.regplot(data=df, x='Category Two Defects', y='Total Cup Points', 
            scatter_kws={'alpha':0.5, 'color':'#d62728'}, line_kws={'color':'black', 'linewidth':2.5})

plt.title("4. Introduction au Modèle : Impact des Défauts Physiques sur le Score Final", fontsize=14)
plt.xlabel("Nombre de Défauts de Catégorie 2 (Variable Explicative $X$)", fontsize=12)
plt.ylabel("Score Final (Variable Cible $Y$)", fontsize=12)
plt.tight_layout()
plt.savefig("4_Regression_defauts_vs_score.png")
plt.show()