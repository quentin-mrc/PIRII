import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def get_continuous_segments(mask):
    """Transforme un masque booléen en liste de segments continus."""
    if not mask.any(): return []
    padded = np.concatenate(([False], mask.values, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    segments = []
    for s, e in zip(starts, ends):
        segments.append(range(s, e))
    return segments

def merge_close_events(events, gap_threshold=50):
    """Fusionne les segments proches."""
    if not events: return []
    merged = [events[0]]
    for current in events[1:]:
        previous = merged[-1]
        gap = current.start - previous.stop
        if gap < gap_threshold:
            new_event = range(previous.start, current.stop)
            merged[-1] = new_event
        else:
            merged.append(current)
    return merged

def group_dirac_indices(indices, max_gap=5):
    """Regroupe les points diracs contigus."""
    if len(indices) == 0: return 0, []
    indices = np.sort(indices)
    events = []
    current_event = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] - indices[i-1] <= max_gap:
            current_event.append(indices[i])
        else:
            events.append(current_event)
            current_event = [indices[i]]
    events.append(current_event)
    return len(events), events

def detect_cusum_recenter(x, threshold=8, drift=1.0, cool_down=50):
    x = np.atleast_1d(x).astype('float64')
    gp, gn = np.zeros(x.size), np.zeros(x.size)
    ta = []
    
    # --- Estimation du bruit global (MAD) ---
    # On utilise la médiane globale pour estimer le "bruit" car elle n'est pas polluée par les sauts
    global_median = np.median(x)
    mad = np.median(np.abs(x - global_median))
    std_est = 1.4826 * mad
    
    if std_est == 0: 
        std_est = 1e-6 if np.mean(x) == 0 else abs(np.mean(x)) * 0.01

    # --- Initialisation de la référence ---
    reference_mean = np.mean(x[:50]) 
    
    last_detection_idx = -cool_down - 1
    
    i = 1
    while i < x.size:
        # A. Si on est en période de "Cool Down" (juste après un saut)
        if i < last_detection_idx + cool_down:
            gp[i], gn[i] = 0, 0
            
            # Si on arrive EXACTEMENT à la fin du cool down
            if i == last_detection_idx + cool_down - 1:
                # C'EST ICI QUE LA MAGIE OPÈRE :
                # On regarde où on a atterri (moyenne locale des derniers points du cool down)
                # Et on dit au CUSUM : "Ceci est ta nouvelle référence (zéro)"
                new_plateau_start = max(0, i - 20)
                reference_mean = np.mean(x[new_plateau_start:i+1])
            
            i += 1
            continue

        # B. Calcul standard CUSUM normalisé par rapport à la RÉFÉRENCE COURANTE
        # (x[i] - reference_mean) : C'est l'écart au plateau actuel
        s_val = (x[i] - reference_mean) / std_est
        
        gp[i] = max(0, gp[i-1] + s_val - drift)
        gn[i] = min(0, gn[i-1] + s_val + drift)
        
        # C. Détection
        if gp[i] > threshold or gn[i] < -threshold:
            ta.append(i)
            last_detection_idx = i
            
            # Reset immédiat des accumulateurs
            gp[i], gn[i] = 0, 0
            
            # On ne met pas à jour la reference_mean ici ! 
            # On attend la fin du cool_down pour voir où le signal se stabilise.
            
        i += 1
            
    return ta


file_name = "NavData2.csv"
output_folder = "Resultats_Figures"

# Création du dossier de sauvegarde s'il n'existe pas
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cols_to_analyze = [
    "VPL", "HPL", "PDOP", "Pdop", "NavState", "ClkBias",
    "AgcLevels1", "LlhPosX", "LlhPosY", "LlhPosZ",
    "EcefPosX", "EcefPosY", "EcefPosZ"
]

# --- REGLAGES FINS ---

# STEPS (Gradient)
step_window = 50
step_thresh_factor = 1.5   
step_min_amp_factor = 0.5 

# RAMPES
ramp_thresh_factor = 0.1   
ramp_min_len = 60          

# DIRACS
dirac_window = 31
dirac_sigma = 4
dirac_min_amp_factor = 0.2

# CUSUM (recentré)
# Threshold : 6.0 ou 8.0 (plus bas = plus sensible, car le re-centrage évite les faux positifs)
cusum_threshold = 8.0  
cusum_drift = 1.0       
cusum_cool_down = 60    # Temps nécessaire pour que le filtre du récepteur se stabilise après un saut

try:
    print(f"Lecture du fichier {file_name}...")
    ref_df = pd.read_csv(file_name)
    
    if 'TowMs' in ref_df.columns:
        ref_df['Time_s'] = ref_df['TowMs'] / 1000
    else:
        ref_df['Time_s'] = ref_df.index

    available_cols = [col for col in cols_to_analyze if col in ref_df.columns]
    
    print("--- DÉBUT DE L'ANALYSE ---")

    for col in available_cols:
        signal = ref_df[col].dropna().astype(float)
        if len(signal) <= 100: continue
        
        time = ref_df.loc[signal.index, 'Time_s']
        global_std = np.std(signal)
        
        # --- LISSAGE ---
        sig_smooth = signal.rolling(window=step_window, center=True, min_periods=1).mean()
        grad = np.gradient(sig_smooth)
        grad_abs = np.abs(grad)
        grad_std = np.std(grad)
        
        step_thresh = grad_std * step_thresh_factor if grad_std > 0 else 0.001
        ramp_thresh = grad_std * ramp_thresh_factor if grad_std > 0 else 0.0005

        # --- STEPS (GRADIENT) ---
        raw_step_mask = grad_abs > step_thresh
        raw_step_segments = get_continuous_segments(pd.Series(raw_step_mask, index=signal.index))
        merged_step_segments = merge_close_events(raw_step_segments, gap_threshold=100)
        
        steps_up, steps_down = [], []
        step_indices = set()
        
        for seg in merged_step_segments:
            if len(seg) < 2: continue
            amp = sig_smooth.iloc[seg[-1]] - sig_smooth.iloc[seg[0]]
            if abs(amp) >= (global_std * step_min_amp_factor):
                step_indices.update(list(seg))
                if amp > 0: steps_up.append(seg)
                else: steps_down.append(seg)

        # --- RAMPES ---
        raw_ramp_mask = grad_abs > ramp_thresh
        raw_ramp_segments = get_continuous_segments(pd.Series(raw_ramp_mask, index=signal.index))
        ramps = []
        for seg in raw_ramp_segments:
            if len(seg) >= ramp_min_len:
                overlap = step_indices.intersection(set(seg))
                if len(overlap) < (len(seg) * 0.5):
                    ramps.append(seg)

        # --- DIRACS ---
        rolling_median = signal.rolling(window=dirac_window, center=True).median()
        diff = np.abs(signal - rolling_median)
        mad = diff.rolling(window=dirac_window, center=True).median()
        local_thresh = dirac_sigma * 1.4826 * mad.replace(0, 0.0001)
        dirac_noise_floor = global_std * dirac_min_amp_factor
        
        is_dirac = (diff > local_thresh) & (diff > dirac_noise_floor)
        dirac_indices = signal.index[is_dirac].tolist()
        num_diracs, dirac_groups = group_dirac_indices(dirac_indices, max_gap=5)
        dirac_points = [diff.loc[g].idxmax() for g in dirac_groups]

        cusum_points_idx = detect_cusum_recenter(sig_smooth.values, 
                                                 threshold=cusum_threshold, 
                                                 drift=cusum_drift, 
                                                 cool_down=cusum_cool_down)
        cusum_pandas_indices = [signal.index[i] for i in cusum_points_idx]

        nb_evts = len(steps_up) + len(steps_down) + len(ramps) + num_diracs + len(cusum_pandas_indices)
        
        if nb_evts > 0:
            print(f"-> {col}: {len(cusum_pandas_indices)} CUSUM Events detected")
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Signaux
            ax.plot(time, signal, color='lightgray', alpha=1.0, linewidth=1, label='Signal Brut')
            ax.plot(time, sig_smooth, color='gray', alpha=0.5, linewidth=1.5, label='Signal Lissé')
            
            # Rampes
            l_added = False
            for r in ramps:
                l = "Rampe" if not l_added else ""
                ax.plot(time.iloc[r], sig_smooth.iloc[r], color='orange', linewidth=4, alpha=0.7, label=l)
                l_added = True

            # Steps Gradient
            l_added = False
            for s in steps_up:
                l = "Step Up (Grad)" if not l_added else ""
                ax.plot(time.iloc[s], sig_smooth.iloc[s], color='magenta', linewidth=3, label=l)
                l_added = True
            l_added = False
            for s in steps_down:
                l = "Step Down (Grad)" if not l_added else ""
                ax.plot(time.iloc[s], sig_smooth.iloc[s], color='cyan', linewidth=3, label=l)
                l_added = True
                
            # CUSUM 
            l_added = False
            for c_idx in cusum_pandas_indices:
                l = "Détection CUSUM" if not l_added else ""
                ax.axvline(x=time.loc[c_idx], color='green', linestyle='--', linewidth=2, alpha=1.0, label=l)
                l_added = True
                
            # Diracs
            l_added = False
            for idx in dirac_points:
                l = "Dirac" if not l_added else ""
                ax.plot(time.loc[idx], signal.loc[idx], color='red', marker='x', markersize=10, markeredgewidth=2.5, linestyle='None', label=l)
                l_added = True
            
            ax.set_title(f"Analyse Multi-Défauts : {col}")
            ax.set_xlabel("Temps (s)")
            ax.set_ylabel("Amplitude")
            ax.legend(loc='upper right', framealpha=0.9)
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            save_path = os.path.join(output_folder, f"Analyse_{col}.png")
            plt.savefig(save_path, dpi=150)
            print(f"   Figure enregistrée : {save_path}")
            plt.close(fig)

except FileNotFoundError:
    print(f"ERREUR : Le fichier '{file_name}' est introuvable.")
except Exception as e:
    print(f"ERREUR INATTENDUE : {e}")