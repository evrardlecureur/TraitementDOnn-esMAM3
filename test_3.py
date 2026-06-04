import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# ==============================================================================
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
# ==============================================================================
df = pd.read_csv("df_arabica_clean.csv")

# Correction des altitudes aberrantes
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
df['Altitude'] = df['Altitude'].fillna(df['Altitude'].median())

# ==============================================================================
# 2. REGROUPEMENT STRICT EN 3 MACRO-RÉGIONS
# ==============================================================================
region_mapping = {
    'Ethiopia': 'Afrique', 'Kenya': 'Afrique', 'Uganda': 'Afrique', 'Madagascar': 'Afrique',
    'Colombia': 'Amériques', 'Guatemala': 'Amériques', 'Honduras': 'Amériques', 'Brazil': 'Amériques',
    'Costa Rica': 'Amériques', 'Nicaragua': 'Amériques', 'El Salvador': 'Amériques', 'Mexico': 'Amériques', 'Peru': 'Amériques',
    'Taiwan': 'Asie-Pacifique', 'Thailand': 'Asie-Pacifique', 'Laos': 'Asie-Pacifique', 'Myanmar': 'Asie-Pacifique', 'Indonesia': 'Asie-Pacifique', 'Vietnam': 'Asie-Pacifique'
}
df['Macro_Region'] = df['Country of Origin'].map(region_mapping).fillna('Autres')

# ==============================================================================
# 3. MODÉLISATION PAR INTERACTION (SANS TRICHE)
# ==============================================================================
# Le symbole '*' crée l'interaction : Altitude + Macro_Region + (Altitude x Macro_Region)
model = smf.ols("Acidity ~ Altitude * C(Macro_Region)", data=df).fit()

print("==============================================================================")
print("RÉSUMÉ STATISTIQUE DU MODÈLE")
print("==============================================================================")
print(model.summary())

# ==============================================================================
# 4. GRAPHIQUE À 3 DROITES NETTES
# ==============================================================================
sns.set_theme(style="whitegrid")

# lmplot va tracer EXACTEMENT une droite par couleur (hue)
g = sns.lmplot(
    data=df[df['Macro_Region'] != 'Autres'], # On exclut 'Autres' pour avoir un graphique parfait
    x='Altitude', 
    y='Acidity', 
    hue='Macro_Region', 
    palette='Set1',
    height=5.5, 
    aspect=1.3,
    scatter_kws={'alpha': 0.6, 's': 40},
    line_kws={'linewidth': 2.5}
)

# Labellisation
g.set_axis_labels("Altitude (mètres)", "Score d'Acidité (Acidity)", fontsize=11)
g.fig.suptitle(f"L'effet de l'Altitude sur l'Acidité dépend de la Région du monde\n(Modèle d'Interaction Linéaire - R² = {model.rsquared:.3f})", 
               fontweight='bold', fontsize=11, y=1.05)

# Sauvegarde
plt.savefig("graphique_interaction_macro_regions.png", dpi=300, bbox_inches='tight')
plt.show()