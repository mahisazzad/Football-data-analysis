import pandas as pd, numpy as np, json
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

np.random.seed(42)
df = pd.read_csv("carries_final.csv")
df["line_break_2d"] = pd.to_numeric(df["line_break_2d"], errors="coerce")
df["progressive_carry"] = df["progressive_carry"].astype(int)
df["under_pressure"] = df["under_pressure"].astype(int)
df = df.dropna(subset=["defender_proximity_m", "line_break_2d", "pitch_control_gained"]).copy()
df["line_break_2d"] = df["line_break_2d"].astype(int)

FEAT6 = ["carry_distance_m", "carry_duration_s", "carry_speed_ms", "defender_proximity_m", "under_pressure", "progressive_carry"]

ROSTER1 = ['Lionel Andrés Messi Cuccittini','Frenkie de Jong','Pedro González López','Sergio Busquets i Burgos',
           'Ousmane Dembélé','Antoine Griezmann','Philippe Coutinho Correia','Miralem Pjanić',
           'Vinícius José Paixão de Oliveira Júnior','Toni Kroos','Marcos Llorente Moreno','João Félix Sequeira',
           'Luka Modrić','Iago Aspas Juncal','Karim Benzema']
SHORT1 = {'Lionel Andrés Messi Cuccittini':'Messi','Frenkie de Jong':'de Jong','Pedro González López':'Pedri',
    'Sergio Busquets i Burgos':'Busquets','Ousmane Dembélé':'Dembélé','Antoine Griezmann':'Griezmann',
    'Philippe Coutinho Correia':'Coutinho','Miralem Pjanić':'Pjanić',
    'Vinícius José Paixão de Oliveira Júnior':'Vinícius Jr','Toni Kroos':'Kroos','Marcos Llorente Moreno':'Llorente',
    'João Félix Sequeira':'João Félix','Luka Modrić':'Modrić','Iago Aspas Juncal':'Aspas','Karim Benzema':'Benzema'}
ROSTER2 = ['Lionel Andrés Messi Cuccittini','Kylian Mbappé Lottin','Neymar da Silva Santos Junior',
           'Ángel Fabián Di María Hernández','Marco Verratti','Idrissa Gana Gueye','Danilo Luís Hélio Pereira',
           'Georginio Wijnaldum','Dimitri Payet','Seko Fofana']
SHORT2 = {'Lionel Andrés Messi Cuccittini':'Messi (PSG)','Kylian Mbappé Lottin':'Mbappé','Neymar da Silva Santos Junior':'Neymar',
          'Ángel Fabián Di María Hernández':'Di María','Marco Verratti':'Verratti','Idrissa Gana Gueye':'Gueye',
          'Danilo Luís Hélio Pereira':'Danilo Pereira','Georginio Wijnaldum':'Wijnaldum','Dimitri Payet':'Payet','Seko Fofana':'Fofana'}

def coef_table(fit, names):
    ci = fit.conf_int()
    out = []
    for n in names:
        out.append({
            "term": n, "coef": round(float(fit.params[n]), 4), "se": round(float(fit.bse[n]), 4),
            "z_or_t": round(float(fit.tvalues[n]), 3), "p_value": round(float(fit.pvalues[n]), 4),
            "ci_low": round(float(ci.loc[n, 0]), 4), "ci_high": round(float(ci.loc[n, 1]), 4),
            "significant_95": bool(fit.pvalues[n] < 0.05),
        })
    return out

results = {}

# ===================== 1. OLS: pitch_control_gained ~ 6 features (continuous outcome, robust SE) =====================
for season in df["season"].unique():
    d = df[df["season"] == season]
    X = sm.add_constant(d[FEAT6])
    y = d["pitch_control_gained"]
    fit = sm.OLS(y, X).fit(cov_type="HC3")  # heteroskedasticity-robust SEs
    tab = coef_table(fit, ["const"] + FEAT6)
    results[f"ols_pitch_control_{season}"] = {"r_squared": round(fit.rsquared, 4), "n": int(fit.nobs), "table": tab}
    print(f"\n=== OLS pitch_control_gained ~ 6 features [{season}] (HC3 robust SE) ===")
    print(f"R^2 = {fit.rsquared:.4f}  n = {int(fit.nobs)}")
    for row in tab:
        print(f"  {row['term']:<24s} coef={row['coef']:+.4f}  SE={row['se']:.4f}  p={row['p_value']:.4f}  "
              f"95% CI=[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  {'*' if row['significant_95'] else ''}")

# ===================== 2. Logit: line_break_2d ~ 6 features (baseline, both seasons) =====================
for season in df["season"].unique():
    d = df[df["season"] == season]
    X = sm.add_constant(d[FEAT6])
    y = d["line_break_2d"]
    fit = sm.Logit(y, X).fit(disp=0)
    tab = coef_table(fit, ["const"] + FEAT6)
    results[f"logit_baseline_{season}"] = {"pseudo_r2": round(fit.prsquared, 4), "n": int(fit.nobs), "table": tab}
    print(f"\n=== Logit line_break_2d ~ 6 features [{season}] ===")
    print(f"Pseudo R^2 = {fit.prsquared:.4f}  n = {int(fit.nobs)}")
    for row in tab:
        print(f"  {row['term']:<24s} coef={row['coef']:+.4f}  SE={row['se']:.4f}  p={row['p_value']:.4f}  "
              f"95% CI=[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  {'*' if row['significant_95'] else ''}")

# ===================== 3. Logit with player fixed effects, formal inference (both seasons) =====================
def fixed_effects_inference(season_tag, roster, short_map, min_n=10):
    d = df[df["season"] == season_tag].copy()
    d["short_name"] = d["player"].map(short_map)
    d = d[d["player"].isin(roster)]
    counts = d["short_name"].value_counts()
    keep = counts[counts >= min_n].index.tolist()
    d = d[d["short_name"].isin(keep)]
    baseline_player = sorted(keep)[0]
    dummies = pd.get_dummies(d["short_name"], prefix="player", drop_first=True).astype(int)
    X = pd.concat([d[FEAT6].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = d["line_break_2d"].reset_index(drop=True)
    fit = sm.Logit(y, X).fit(disp=0, maxiter=200)
    player_terms = [c for c in X.columns if c.startswith("player_")]
    tab = coef_table(fit, player_terms)
    for row in tab:
        row["player"] = row["term"].replace("player_", "")
        row["n"] = int(counts[row["player"]])
    print(f"\n=== Logit + player fixed effects [{season_tag}] (baseline: {baseline_player}) ===")
    print(f"Pseudo R^2 = {fit.prsquared:.4f}  n = {int(fit.nobs)}")
    for row in sorted(tab, key=lambda r: -r["coef"]):
        sig = "***" if row["p_value"]<0.01 else ("**" if row["p_value"]<0.05 else ("*" if row["p_value"]<0.10 else ""))
        print(f"  {row['player']:<16s} coef={row['coef']:+.4f}  SE={row['se']:.4f}  p={row['p_value']:.4f}  "
              f"95% CI=[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  n={row['n']:<5d} {sig}")
    return {"baseline_player": baseline_player, "pseudo_r2": round(fit.prsquared, 4), "n": int(fit.nobs), "table": tab}

results["fe_season1"] = fixed_effects_inference("LaLiga_2020_21", ROSTER1, SHORT1)
results["fe_season2"] = fixed_effects_inference("Ligue1_2021_22", ROSTER2, SHORT2)

# ===================== 4. Bootstrap CI for GBM AUC (no classical parameters to test) =====================
def bootstrap_auc(d, feat_cols, target, n_boot=300):
    X, y = d[feat_cols].values, d[target].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    gb.fit(Xtr, ytr)
    proba = gb.predict_proba(Xte)[:, 1]
    point_auc = roc_auc_score(yte, proba)
    n = len(yte)
    rng = np.random.RandomState(42)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(yte[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(yte[idx], proba[idx]))
    boot_aucs = np.array(boot_aucs)
    return {"point_auc": round(float(point_auc), 4),
            "boot_mean": round(float(boot_aucs.mean()), 4),
            "ci_low": round(float(np.percentile(boot_aucs, 2.5)), 4),
            "ci_high": round(float(np.percentile(boot_aucs, 97.5)), 4)}

for season in df["season"].unique():
    d = df[df["season"] == season]
    r5 = bootstrap_auc(d, FEAT6[:5], "line_break_2d")
    r6 = bootstrap_auc(d, FEAT6, "line_break_2d")
    results[f"gbm_boot_5feat_{season}"] = r5
    results[f"gbm_boot_6feat_{season}"] = r6
    print(f"\n=== GBM bootstrap AUC 95% CI [{season}] ===")
    print(f"  5-feat: AUC={r5['point_auc']}  95% CI=[{r5['ci_low']}, {r5['ci_high']}]")
    print(f"  6-feat: AUC={r6['point_auc']}  95% CI=[{r6['ci_low']}, {r6['ci_high']}]")

json.dump(results, open("inference_results.json", "w"), indent=2)
print("\nSaved inference_results.json")
