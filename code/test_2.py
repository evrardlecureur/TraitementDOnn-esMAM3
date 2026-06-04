import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
import statsmodels.api as sm

# ==============================================================================
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
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

# Correction des altitudes (Guatemala, etc.)
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

# Calcul de l'Âge du Café (Jours écoulés entre la récolte et l'expiration)
df['Harvest Year'] = df['Harvest Year'].astype(str).str.split('/').str[0].str.strip()
df['Harvest Year'] = pd.to_datetime(df['Harvest Year'], format='%Y', errors='coerce')
df['Expiration'] = df['Expiration'].apply(lambda x: parser.parse(x) if pd.notnull(x) else np.nan)
df['Coffee Age'] = (df['Expiration'] - df['Harvest Year']).dt.days
df['Coffee Age'] = df['Coffee Age'].fillna(df['Coffee Age'].median())

# Simplification et regroupement de la couleur
color_mapping = {
    'green': 'Green', 'greenish': 'Greenish', 
    'bluish-green': 'Bluish/Blue-Green', 'blue-green': 'Bluish/Blue-Green',
    'yellow-green': 'Yellow-Green', 'yellow green': 'Yellow-Green', 
    'yellow- green': 'Yellow-Green', 'yello-green': 'Yellow-Green'
}
df['Color_Group'] = df['Color'].map(color_mapping).fillna('Other')

# Regroupement des pays d'origine (fréquence >= 5)
top_countries = df['Country of Origin'].value_counts()
top_countries = top_countries[top_countries >= 5].index
df['Country_Group'] = df['Country of Origin'].apply(lambda x: x if x in top_countries else 'Other')

# Regroupement des variétés botaniques (fréquence >= 5)
top_varieties = df['Variety'].fillna('unknown').value_counts()
top_varieties = top_varieties[top_varieties >= 5].index
df['Variety_Group'] = df['Variety'].fillna('unknown').apply(lambda x: x if x in top_varieties else 'Other')


# ==============================================================================
# 2. CONSTRUCTION DU MODÈLE DE RÉGRESSION MULTIPLE (HORS GOÛT)
# ==============================================================================

# Variables explicatives uniquement physiques, géographiques et techniques
numeric_features = ['Altitude', 'Coffee Age', 'Moisture Percentage', 'Category One Defects', 'Quakers', 'Category Two Defects']
categorical_features = ['Processing Method', 'Color_Group', 'Country_Group', 'Variety_Group']

df_reg = df[['Total Cup Points'] + numeric_features + categorical_features].dropna().copy()

# Encodage (One-Hot) des variables catégorielles
X = df_reg[numeric_features].copy()
for col in categorical_features:
    dummies = pd.get_dummies(df_reg[col], prefix=col, drop_first=True, dtype=float)
    X = pd.concat([X, dummies], axis=1)

y = df_reg['Total Cup Points']
X_with_const = sm.add_constant(X)

# Entraînement du modèle OLS
model = sm.OLS(y, X_with_const).fit()

# Calcul du score prédit par le modèle (Indice Composite)
df_reg['Predicted_Score'] = model.predict(X_with_const)


# ==============================================================================
# 3. GÉNÉRATION DU GRAPHIQUE UNIQUE (OPTIMISÉ CONTRE LES COUPURES)
# ==============================================================================

sns.set_theme(style="whitegrid")

# layout="constrained" ajuste automatiquement et proprement les éléments dans la zone d'image
fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

# Droite de régression
sns.regplot(data=df_reg, x='Predicted_Score', y='Total Cup Points', ax=ax,
            scatter_kws={'alpha': 0.6, 'color': '#2ca02c', 's': 45},
            line_kws={'color': '#d62728', 'linewidth': 2.2})

# Ligne de repère idéale (Y = X)
min_val = min(df_reg['Predicted_Score'].min(), df_reg['Total Cup Points'].min())
max_val = max(df_reg['Predicted_Score'].max(), df_reg['Total Cup Points'].max())
ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', alpha=0.7, label='Modèle Parfait (Y = X)')

# Titre clair et condensé avec le R²
ax.set_title(f"Score prédit avec les caractéristiques physique\n(R² = {model.rsquared:.3f})", 
             fontsize=13, fontweight='bold', pad=10)

# Axes simplifiés pour gagner de l'espace
ax.set_xlabel("Score prédit (Indice composite)", fontsize=11)
ax.set_ylabel("Score réel (Total Cup Points)", fontsize=11)

# Légende ancrée en haut à gauche pour ne pas déborder à droite
ax.legend(loc='upper left', frameon=True, facecolor='white', alpha=0.9)

# Sauvegarde propre avec une bonne résolution (300 DPI)
plt.savefig("score_predit_caracteristiques_physiques.png", dpi=300)
plt.show()