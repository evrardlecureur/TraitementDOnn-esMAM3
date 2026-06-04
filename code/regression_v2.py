import pandas as pd
import numpy as np
import statsmodels.api as sm
from dateutil import parser

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

# Nettoyage de l'Altitude
# Nettoyage de l'Altitude
df.loc[df['ID'] == 99, 'Altitude'] = str(5273 / 3.281)
df.loc[df['ID'] == 105, 'Altitude'] = "1800"
df.loc[df['ID'] == 180, 'Altitude'] = "1400"

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

# Calcul de l'Âge du Café
df['Harvest Year'] = df['Harvest Year'].astype(str).str.split('/').str[0].str.strip()
df['Harvest Year'] = pd.to_datetime(df['Harvest Year'], format='%Y', errors='coerce')
df['Expiration'] = df['Expiration'].apply(lambda x: parser.parse(x) if pd.notnull(x) else np.nan)
df['Coffee Age'] = (df['Expiration'] - df['Harvest Year']).dt.days

# Nettoyage des variables numériques supplémentaires
df['Moisture Percentage'] = pd.to_numeric(df['Moisture Percentage'], errors='coerce')
df['Category One Defects'] = pd.to_numeric(df['Category One Defects'], errors='coerce').fillna(0)
df['Category Two Defects'] = pd.to_numeric(df['Category Two Defects'], errors='coerce').fillna(0)
df['Quakers'] = pd.to_numeric(df['Quakers'], errors='coerce').fillna(0)

# Regroupement des Pays (Garder le Top 5, le reste en "Other")
top_countries = df['Country of Origin'].value_counts().nlargest(5).index
df['Country'] = df['Country of Origin'].where(df['Country of Origin'].isin(top_countries), 'Other')

# Regroupement des Variétés (Garder le Top 3 (ex: Gesha, Caturra, Typica), le reste en "Other")
top_varieties = df['Variety'].value_counts().nlargest(3).index
df['Variety_Group'] = df['Variety'].where(df['Variety'].isin(top_varieties), 'Other')


# ==============================================================================
# 2. PRÉPARATION DE LA RÉGRESSION MULTIPLE
# ==============================================================================

# Sélection des features strictes "hors goût"
features = [
    'Altitude', 'Coffee Age', 'Moisture Percentage', 
    'Category One Defects', 'Category Two Defects', 'Quakers',
    'Processing Method', 'Country', 'Variety_Group'
]

# Filtrage des lignes avec des NaN sur nos features critiques
df_reg = df.dropna(subset=['Total Cup Points', 'Altitude', 'Coffee Age', 'Moisture Percentage']).copy()

# Encodage One-Hot des variables catégorielles (Pays, Procédé, Variété)
# drop_first=True permet d'éviter le piège de la colinéarité parfaite
X_encoded = pd.get_dummies(df_reg[features], drop_first=True, dtype=float)

X = sm.add_constant(X_encoded)
y = df_reg['Total Cup Points']

# ==============================================================================
# 3. ENTRAÎNEMENT DU MODÈLE ET RÉSULTATS
# ==============================================================================

model = sm.OLS(y, X).fit()

print("="*60)
print("RÉSULTATS DE LA RÉGRESSION ÉLARGIE (SANS NOTES SENSORIELLES)")
print("="*60)
print(f"R^2 (Coefficient de détermination) : {model.rsquared:.4f}")
print(f"R^2 Ajusté                         : {model.rsquared_adj:.4f}")
print("-"*60)
print("COEFFICIENTS DES FEATURES :")
print(model.params.round(4).sort_values(ascending=False).to_string())
print("="*60)

# Si tu veux voir le rapport complet et détaillé de statsmodels, décommente la ligne ci-dessous :
# print(model.summary())
