import subprocess
import os
import time
import csv

def run_endurance_trial(trial_id, lr, weight_decay, sem_weight, pct_start):
    print(f"\n🏃 ENDURANCE DUEL {trial_id}: LR={lr}, WD={weight_decay}, SEM={sem_weight}, PCT={pct_start}")
    
    cmd = [
        "./venv/bin/python", "train_monster.py",
        "--lr", str(lr),
        "--weight-decay", str(weight_decay),
        "--sem-weight", str(sem_weight),
        "--pct-start", str(pct_start),
        "--epochs", "5",
        "--data-dir", "/home/nini/Model_training/data/monster_chunks",
        "--trial"
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    all_acc = []
    last_metrics = {}
    
    try:
        for line in process.stdout:
            # On cherche notre nouveau format : STEP=100 L_CE=...
            if "STEP=" in line:
                try:
                    import re
                    metrics = re.findall(r'(\w+)=([\d\.]+)', line)
                    m_dict = {k: float(v) for k, v in metrics}
                    
                    if 'Acc' in m_dict:
                        all_acc.append(m_dict['Acc'])
                    last_metrics = m_dict
                    
                    # On affiche la progression en direct pour nous rassurer
                    print(f"   Step {int(m_dict['STEP']):04d} | Acc: {m_dict['Acc']*100:.2f}% | L_SEM: {m_dict['L_SEM']:.4f}", end='\r')
                    
                    # On laisse tourner 2000 steps (env 4 chunks)
                    if m_dict['STEP'] >= 2000:
                        process.terminate()
                        break
                except:
                    pass
    except:
        process.kill()

    print("\n") # New line after live progress
    if len(all_acc) > 5:
        # Pente calculée sur les 10 dernières mesures (1000 derniers steps)
        # Pour voir si ça s'essouffle ou si ça accélère
        recent_acc = all_acc[-10:]
        slope = (recent_acc[-1] - recent_acc[0]) / 1000.0
        avg_acc = (sum(all_acc) / len(all_acc)) * 100
        last_acc = all_acc[-1] * 100
        
        print(f"📊 Results: Avg Acc={avg_acc:.2f}% | Final Acc={last_acc:.2f}% | Slope={slope:.8f}")
        last_metrics['Avg_Acc'] = avg_acc
        last_metrics['Final_Acc'] = last_acc
        last_metrics['Slope'] = slope
        return last_metrics
    return {}, 0

def main():
    # --- DUEL FINAL ---
    duelists = [
        {"name": "Champion (Vitesse)", "lr": 1.2e-3, "wd": 0.03, "sw": 0.7, "ps": 0.1},
        {"name": "Challenger (Stable)", "lr": 8e-4, "wd": 0.02, "sw": 0.7, "ps": 0.2},
        {"name": "Aggressive High-WD", "lr": 1.5e-3, "wd": 0.05, "sw": 0.8, "ps": 0.1}
    ]

    for d in duelists:
        res = run_endurance_trial(d["name"], d["lr"], d["wd"], d["sw"], d["ps"])
        print(f"✅ {d['name']} - AVG Acc: {res.get('Avg_Acc', 0):.2f}% | Final Acc: {res.get('Acc')}% | GN: {res.get('GN')}")

if __name__ == "__main__":
    main()
