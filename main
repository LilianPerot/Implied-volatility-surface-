import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import yfinance as yf
import math
from datetime import datetime

def black_scholes_price(S, K, T, risk_free_rate, sigma, option_type):
    """
    Calcule le prix théorique d'une option de type call ou put
    selon le modèle de Black-Scholes.

    Cette fonction est utilisée comme base pour l'estimation numérique
    de la volatilité implicite via inversion du modèle.
    
    S = prix actuel de l'actif sous-jacent
    K = strike de l'option
    T = maturité de l'option
    risk_free_rate = taux sans risque annuel
    sigma = volatilité implicite de l'actif 
    option_type = type de l'option "call" ou "put"
    """
    
# retourne NaN si T et sigma sont négatifs
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (math.log(S / K) + (risk_free_rate + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
# Fonction de répartition cumulative pour la loi normale 
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))

# pricing de l'option selon le type 
    if option_type == "call":
        return S * N(d1) - K * math.exp(-risk_free_rate * T) * N(d2)
    elif option_type == "put":
        return K * math.exp(-risk_free_rate * T) * N(-d2) - S * N(-d1)
    else:
        return np.nan

def implied_volatility(S, K, T, risk_free_rate, market_price, option_type, tol=1e-6, max_iter=100):
    """
    Estime la volatilité implicite d'une option par dichotomie 
    en utilisant la formule de Black-Scholes.

    market_price = prix observé de l'option
    tol = tolérance pour la convergence de la volatilité implicite
    max_iter = nombre maximum d'itérations pour la dichotomie
    """   
    sigma_low = 1e-6 # borne basse pour la dichotomie 
    sigma_high = 2.0 # borne haute pour la dichotomie
    for _ in range(max_iter):
        sigma_mid = 0.5 * (sigma_low + sigma_high)
        price = black_scholes_price(S, K, T, risk_free_rate, sigma_mid, option_type)
        if np.isnan(price): # NaN si aucun prix n'est calculable
            return np.nan
        if price > market_price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
        if abs(price - market_price) < tol:
            return sigma_mid
    return np.nan

def compute_implied_volatility(ticker, option_type, risk_free_rate):
    """
    Construit un DataFrame contenant les volatilités implicites de toutes les options disponibles
    pour un ticker donné, selon le type d'option et le taux sans risque.
    Le DataFrame contient: Strike, maturité, prix de marché, volatilité implicite, type et expiration

    ticker = code d'identification de l'actif sous-jacent dans Yahoo Finance
    """
# Récupération des données via Yfinance 
    asset = yf.Ticker(ticker)
    hist = asset.history(period="5d")["Close"]
    if hist.empty:
        raise ValueError(f"Aucune donnée trouvée pour le ticker {ticker}")
    spot_price = hist.iloc[-1] # sélectionne le dernier prix spot 
    iv_data = [] # liste pour le stockage
    for expiry in asset.options:
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
        T = (expiry_date - datetime.today()).days / 365 # convertit la date d'expiration en années jusqu'à maturité
        if T <= 0.05:    # filtre les maturités trop courtes qui faussent les résultats 
            continue
        chain = asset.option_chain(expiry)
        options = chain.calls if option_type == "call" else chain.puts
        for _, row in options.iterrows():
            K = row["strike"]
            bid = row["bid"]
            ask = row["ask"]
            if bid <= 0 or ask <= 0: # utilisation unique des options avec bid/ask positif
                continue
            market_price = 0.5*(bid + ask)
            if K < 0.5 * spot_price or K > 1.5 * spot_price:
                continue
            if market_price <= 0 or np.isnan(market_price):
                continue
                
        # lancement du calcul de la volatilité implicite
            iv = implied_volatility(S=spot_price, K=K, T=T, risk_free_rate=risk_free_rate, market_price=market_price, option_type=option_type)

        # Stockage des résultats du calcul
            iv_data.append({"ticker": ticker, "option_type": option_type, "expiry": expiry, "T": T, "strike": K,
                "option_price": market_price, "implied_vol": iv})
    df_iv = pd.DataFrame(iv_data)
    df_iv = df_iv.dropna()
    if df_iv.empty:
        raise ValueError(f"Aucune option trouvée pour le ticker {ticker} et le type d'option {option_type}")
    df_iv = df_iv[df_iv["implied_vol"] > 0] # garde seulement les résultats positifs
    df_iv["spot_price"] = spot_price
    return df_iv

def volatility_smile(ticker, option_type, T, risk_free_rate):
    """
    Trace et retourne le smile de volatilité pour un actif et une maturité spécifique.
    Retourne deux tuples: K_sorted et iv_sorted
    
    K_sorted = strikes triés par maturité
    iv_sorted = volatilité implicite correspondantes triées 
    """
    df_iv = compute_implied_volatility(ticker, option_type, risk_free_rate)
    tol = 1e-3
    df_T = df_iv[np.abs(df_iv["T"] - T) < tol] # selectionne l'option dans la marge de tolérance tol
    if df_T.empty:
        available_T = [round(t, 3) for t in sorted(df_iv['T'].unique())]
        raise ValueError(f"Aucune option trouvée pour la maturité T={T:.4f} ans. "
                         f"Maturités disponibles : {available_T}")
    K = df_T["strike"].values
    implied_vol = df_T["implied_vol"].values

# vérification de la conformité des données
    if len(K) == 0 and len(implied_vol) == 0:
        raise ValueError("K ou implied_vol est vide")
    if len(K) != len(implied_vol):
        raise ValueError("K et implied_vol doivent être de même longueur")
    if any([k <= 0 for k in K]):
        raise ValueError("Tous les strikes doivent être positifs")
    if any([iv <= 0 for iv in implied_vol]):
        raise ValueError("Toutes les volatilités implicites doivent être positives")
    if T <= 0:
        raise ValueError("La maturité T doit être positive")

# tri par strike
    K_array = np.array(K)
    iv_array = np.array(implied_vol)
    sorted_indices = np.argsort(K_array)
    K_sorted = K_array[sorted_indices]
    iv_sorted = iv_array[sorted_indices]
    plt.figure(figsize = (8,5))
    plt.plot(K_sorted, iv_sorted)
    plt.title(f"Smile de volatilité {ticker} pour T = {T: .2f} années")
    plt.xlabel("Strike (K)")
    plt.ylabel("Implied volatility")
    plt.grid()
    plt.savefig(f"Volatility_smile_{ticker}_{option_type}.png")
    plt.show()
    return K_sorted, iv_sorted 
    
def volatility_surface(ticker, option_type, risk_free_rate):
    """
    Construit et trace une surface de volatilité implicite pour un actif donné,
    en fonction des strikes et maturités disponibles.
    Affiche la surface 3D et sauvegarde le graphique sous le nom "Valatility_surface.png"
    """
    df_iv = compute_implied_volatility(ticker, option_type, risk_free_rate)

# vérification nombre de données minimum requis pour une surface
    unique_T = df_iv["T"].nunique()
    unique_strike = df_iv["strike"].nunique()
    if unique_T < 2:
        raise ValueError("Impossible de construire la surface de volatilité, une seule maturité est disponible")
    if unique_strike < 2:
        raise ValueError("Impossible de construire la surface de volatilité, un seul strike est disponible")
    
# pivot pour obtenir une matrice T x Strike
    df_surface = df_iv.pivot(index = "T", columns = "strike", values = "implied_vol")
    df_surface = df_surface.sort_index(axis = 0)
    df_surface = df_surface.sort_index(axis = 1)
    x = df_surface.columns.values # strikes
    y = df_surface.index.values # maturités
    z = df_surface.values # volatilités implicites
    x_grid, y_grid = np.meshgrid(x, y)
    fig = plt.figure(figsize =(9,7))
    axe = fig.add_subplot(111, projection = "3d")
    surface = axe.plot_surface(x_grid, y_grid, z, cmap = "viridis", edgecolor = "none")
    plt.title(f"Surface de volatilité {ticker}")
    axe.set_xlabel("Strike")
    axe.set_ylabel("Maturity")
    axe.set_zlabel("Implied volatility")
    fig.colorbar(surface, ax = axe, shrink = 0.4, aspect = 5)
    plt.savefig(f"Volatility_surface_{ticker}_{option_type}.png")
    plt.show()

def compute_greeks(ticker, risk_free_rate, option_type):
    """
    Calcule les grecs (Delta, Gamma, Vega, Theta) pour toutes les options disponibles
    d'un actif, pour toutes les maturités.
    Sauvegarde les données dnas un fichier CSV.
    """
    
    df_iv = compute_implied_volatility(ticker, option_type, risk_free_rate)
    S = df_iv["spot_price"].iloc[0]      # récupération du prix spot
    for T in sorted(df_iv["T"].unique()):
        df_T = df_iv[df_iv["T"] == T]
        df_T = df_T[df_T["option_type"] == option_type]
        strikes = df_T["strike"].values
        implied_vol_greeks = df_T["implied_vol"].values
        S_vector = np.full_like(strikes, S, dtype = float)

    # calcul de d1, d2 et N (répartition cumulative pour la loi normale) pour le calcul des grecs
        d1 = (np.log(S_vector / strikes) + (risk_free_rate + 0.5 * implied_vol_greeks**2) * T) / (implied_vol_greeks * np.sqrt(T))
        d2 = d1 - implied_vol_greeks * np.sqrt(T)
        N = np.vectorize(lambda x: 0.5 * (1 + math.erf(x / np.sqrt(2))))
    # calcul de delta
        if option_type == "call":
            delta = N(d1)
        else:
            delta = N(d1) - 1
    # calcul de gamma
        gamma = np.exp(-0.5 * d1**2) / (S_vector * implied_vol_greeks * np.sqrt(2 * np.pi * T))
    # calcul de vega
        vega = S_vector * np.sqrt(T) * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    # calcul de theta
        if option_type == "call":
            theta = - (S_vector * implied_vol_greeks * np.exp(-0.5 * d1**2)) / (2 * np.sqrt(2 * np.pi * T)) - risk_free_rate * strikes * np.exp(-risk_free_rate*T) * N(d2)
        else:
            theta = - (S_vector * implied_vol_greeks * np.exp(-0.5 * d1**2)) / (2 * np.sqrt(2 * np.pi * T)) + risk_free_rate * strikes * np.exp(-risk_free_rate*T) * N(-d2)
        df_greeks = pd.DataFrame({"strike": strikes, "T": T, "implied_vol": implied_vol_greeks, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta})
        df_greeks.to_csv("greeks.csv", sep = ',')
        return

def main():
    """
    Cette fonction sert d'interface utilisateur CLI.
    Elle :
    - demande les paramètres nécessaires à l’analyse de la volatilité (ticker, type d’option, taux sans risque)
    - permet de sélectionner la fonctionnalité souhaitée :
        • calcul et affichage du smile de volatilité
        • construction de la surface de volatilité implicite
        • calcul des grecs
    - appelle les fonctions correspondantes en fonction du choix effectué.

    La fonction assure une validation minimale des entrées utilisateur
    et lève des erreurs explicites en cas de choix invalide.

    Aucune valeur n’est retournée : les résultats sont affichés sous forme de graphiques
    et/ou exportés dans des fichiers (CSV, PNG).
    """
# demande des inputs pour les fonctions de base (black_scholes_price, implied_volatility et compute_implied_volatility)
    ticker = input("choisissez un actif à analyser (exemple: 'NVDA') : ")
    option_type = input("Choisissez le type de l'option 'call' ou 'put'")
    if option_type not in ['call', 'put']:
        raise ValueError("L'option doit être de type 'call' ou 'put'")
    risk_free_rate = float(input("Entrez le taux sans risque exemple: (0.05 pour 5%) utilisation du taux OAT 10 ans conseillé"))
# choix de la fonction et demande d'inputs spécifiques si besoin
    fonction_choice = input("choisir une fonctionnalité: 'smile', 'surface', 'greeks'")
    if fonction_choice == "smile":
        T = float(input("Choisir la maturité à laquelle le smile sera calculé"))
        volatility_smile(ticker, option_type, T, risk_free_rate)
    elif fonction_choice == 'surface':
        volatility_surface(ticker, option_type, risk_free_rate)
    elif fonction_choice == 'greeks':
        compute_greeks(ticker, risk_free_rate, option_type)
    else:
        raise ValueError("le type de fonction n'est pas correct")
        
