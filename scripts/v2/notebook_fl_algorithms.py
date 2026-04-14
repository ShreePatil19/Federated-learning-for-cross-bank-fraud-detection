"""
FL Benchmarking — FL ALGORITHMS
Shared across all model notebooks.
Run notebook_base.py first to define all dependencies.
"""

# ══════════════════════════════════════════════════════════════════
#  AGGREGATION
# ══════════════════════════════════════════════════════════════════

def fedavg_aggregate(local_weights, local_sizes):
    total     = sum(local_sizes)
    new_state = copy.deepcopy(local_weights[0])
    for key in new_state:
        new_state[key] = sum(
            local_weights[i][key] * (local_sizes[i] / total)
            for i in range(len(local_sizes))
        )
    return new_state


# ══════════════════════════════════════════════════════════════════
#  FEDAVG / FEDPROX / PERSFL
# ══════════════════════════════════════════════════════════════════

def run_fedavg(global_model, bank_data, X_te, y_te,
               fl_algo, privacy_mode, path, name, master_csv,
               start_round=1):
    is_lr    = isinstance(global_model, LRWrapper)
    is_perfl = (fl_algo == "PersFL")
    mu       = HP["FEDPROX_MU"] if fl_algo == "FedProx" else 0.0
    n_banks  = len(bank_data)

    # PersFL: save personal models to disk instead of keeping in memory
    personal_model_paths = {}
    if is_perfl and not is_lr:
        for i in range(n_banks):
            pm      = copy.deepcopy(global_model)
            pm_path = os.path.join(path, f"personal_bank{i}.pt")
            torch.save(pm.state_dict(), pm_path)
            personal_model_paths[i] = pm_path
            del pm
        torch.cuda.empty_cache()

    csv_path    = os.path.join(path, f"results_{name}.csv")
    best_auc, best_f1, best_round = 0.0, 0.0, 1
    rounds_95   = None
    all_metrics = []
    bank_aucs   = []

    for rnd in range(start_round, HP["FL_ROUNDS"] + 1):
        t0            = time.time()
        local_weights = []
        local_sizes   = []

        for bank_id, (X_b, y_b) in enumerate(bank_data):
            X_r, y_r = smote_bank(X_b, y_b)
            if is_lr:
                lm = copy.deepcopy(global_model)
                local_train_lr(lm, X_r, y_r)
                local_weights.append(lm.get_params())
                local_sizes.append(len(X_b))
                del lm
            else:
                lm     = copy.deepcopy(global_model)
                loader = make_loader(X_r, y_r)
                gm_ref = copy.deepcopy(global_model) if mu > 0 else None
                lm, _  = local_train_nn(
                    lm, loader, HP["LOCAL_EPOCHS"], HP["LR"],
                    privacy_mode, global_model=gm_ref, mu=mu
                )
                local_weights.append(copy.deepcopy(lm.state_dict()))
                local_sizes.append(len(X_b))
                del lm, loader
                if gm_ref is not None:
                    del gm_ref
                torch.cuda.empty_cache()

        # Aggregate
        if is_lr:
            total = sum(local_sizes)
            global_model.set_params({
                "coef": sum(
                    local_weights[i]["coef"] * (local_sizes[i] / total)
                    for i in range(n_banks)
                ),
                "intercept": sum(
                    local_weights[i]["intercept"] * (local_sizes[i] / total)
                    for i in range(n_banks)
                )
            })
        else:
            new_state = fedavg_aggregate(local_weights, local_sizes)
            global_model.load_state_dict(new_state)
            del new_state

        del local_weights
        torch.cuda.empty_cache()

        # PersFL fine-tuning — one model at a time, save to disk
        if is_perfl and not is_lr:
            for bank_id, (X_b, y_b) in enumerate(bank_data):
                X_r, y_r = smote_bank(X_b, y_b)
                pm       = copy.deepcopy(global_model)
                loader   = make_loader(X_r, y_r)
                pm, _    = local_train_nn(
                    pm, loader, HP["FINETUNE_EPOCHS"], HP["FINETUNE_LR"], "NoDP"
                )
                torch.save(pm.state_dict(), personal_model_paths[bank_id])
                del pm, loader
                torch.cuda.empty_cache()

        # Evaluate
        probs = global_model.predict_proba(X_te) if is_lr else get_probs_nn(global_model, X_te)
        m     = compute_metrics(y_te, probs)
        loss  = float(-(y_te * np.log(probs + 1e-9) +
                        (1 - y_te) * np.log(1 - probs + 1e-9)).mean())

        # Per-bank fairness
        bank_aucs = []
        for i, (X_b, y_b) in enumerate(bank_data):
            if len(np.unique(y_b)) < 2:
                bank_aucs.append(None)
                continue
            if is_perfl and not is_lr and i in personal_model_paths:
                eval_m = copy.deepcopy(global_model)
                eval_m.load_state_dict(
                    torch.load(personal_model_paths[i], map_location=DEVICE, weights_only=False)
                )
                p = get_probs_nn(eval_m, X_b)
                del eval_m
                torch.cuda.empty_cache()
            elif is_lr:
                p = global_model.predict_proba(X_b)
            else:
                p = get_probs_nn(global_model, X_b)
            bank_aucs.append(round(roc_auc_score(y_b, p), 6))

        sigma, jfi = fairness_metrics(bank_aucs)

        if m["auc"] > best_auc:
            best_auc, best_f1, best_round = m["auc"], m["f1"], rnd
        if rounds_95 is None and m["auc"] >= 0.95 * best_auc:
            rounds_95 = rnd

        row = {"round": rnd, **m, "sigma_auc": sigma, "jfi": jfi,
               "loss": round(loss, 6), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        append_csv_row(csv_path, row, write_header=(rnd == start_round))
        all_metrics.append(row)
        save_checkpoint(path, name, rnd, global_model, m, is_lr=is_lr)

        print(f"  Round {rnd:03d}/{HP['FL_ROUNDS']:03d} | "
              f"AUC: {m['auc']:.4f} | F1: {m['f1']:.4f} | "
              f"R@1%FPR: {m['recall_1fpr']:.4f} | P@K: {m['prec_at_k']:.4f} | "
              f"KS: {m['ks_stat']:.4f} | σ: {sigma:.4f} | Loss: {loss:.4f}")

    return global_model, all_metrics, best_auc, best_f1, best_round, rounds_95, bank_aucs


# ══════════════════════════════════════════════════════════════════
#  SCAFFOLD
# ══════════════════════════════════════════════════════════════════

def run_scaffold(global_model, bank_data, X_te, y_te,
                 privacy_mode, path, name, master_csv,
                 start_round=1):
    n_banks  = len(bank_data)
    c_global = [torch.zeros_like(p.data).cpu() for p in global_model.parameters()]
    c_locals = [[torch.zeros_like(p.data).cpu() for p in global_model.parameters()]
                for _ in range(n_banks)]

    csv_path    = os.path.join(path, f"results_{name}.csv")
    best_auc, best_f1, best_round = 0.0, 0.0, 1
    rounds_95   = None
    all_metrics = []
    bank_aucs   = []

    for rnd in range(start_round, HP["FL_ROUNDS"] + 1):
        t0            = time.time()
        local_weights = []
        local_sizes   = []
        delta_c_list  = []

        for bank_id, (X_b, y_b) in enumerate(bank_data):
            X_r, y_r = smote_bank(X_b, y_b)
            lm       = copy.deepcopy(global_model)
            loader   = make_loader(X_r, y_r)
            lm, aux  = local_train_nn(
                lm, loader, HP["LOCAL_EPOCHS"], HP["LR"], privacy_mode,
                c_local=c_locals[bank_id], c_global=c_global
            )
            if "new_c_local" in aux:
                delta = [nc - oc for nc, oc in zip(aux["new_c_local"], c_locals[bank_id])]
                c_locals[bank_id] = aux["new_c_local"]
                delta_c_list.append(delta)
            local_weights.append(copy.deepcopy(lm.state_dict()))
            local_sizes.append(len(X_b))
            del lm, loader, aux
            torch.cuda.empty_cache()

        new_state = fedavg_aggregate(local_weights, local_sizes)
        global_model.load_state_dict(new_state)
        del new_state, local_weights
        torch.cuda.empty_cache()

        if delta_c_list:
            for j in range(len(c_global)):
                c_global[j] = c_global[j] + sum(
                    delta_c_list[i][j] for i in range(len(delta_c_list))
                ) / n_banks
        del delta_c_list

        probs = get_probs_nn(global_model, X_te)
        m     = compute_metrics(y_te, probs)
        loss  = float(-(y_te * np.log(probs + 1e-9) +
                        (1 - y_te) * np.log(1 - probs + 1e-9)).mean())

        bank_aucs = []
        for i, (X_b, y_b) in enumerate(bank_data):
            if len(np.unique(y_b)) < 2:
                bank_aucs.append(None)
                continue
            p = get_probs_nn(global_model, X_b)
            bank_aucs.append(round(roc_auc_score(y_b, p), 6))

        sigma, jfi = fairness_metrics(bank_aucs)

        if m["auc"] > best_auc:
            best_auc, best_f1, best_round = m["auc"], m["f1"], rnd
        if rounds_95 is None and m["auc"] >= 0.95 * best_auc:
            rounds_95 = rnd

        row = {"round": rnd, **m, "sigma_auc": sigma, "jfi": jfi,
               "loss": round(loss, 6), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        append_csv_row(csv_path, row, write_header=(rnd == start_round))
        all_metrics.append(row)
        save_checkpoint(path, name, rnd, global_model, m)

        print(f"  Round {rnd:03d}/{HP['FL_ROUNDS']:03d} | "
              f"AUC: {m['auc']:.4f} | F1: {m['f1']:.4f} | "
              f"R@1%FPR: {m['recall_1fpr']:.4f} | P@K: {m['prec_at_k']:.4f} | "
              f"KS: {m['ks_stat']:.4f} | σ: {sigma:.4f} | Loss: {loss:.4f}")

    return global_model, all_metrics, best_auc, best_f1, best_round, rounds_95, bank_aucs


# ══════════════════════════════════════════════════════════════════
#  FEDNOVA
# ══════════════════════════════════════════════════════════════════

def run_fednova(global_model, bank_data, X_te, y_te,
                privacy_mode, path, name, master_csv,
                start_round=1):
    n_banks     = len(bank_data)
    csv_path    = os.path.join(path, f"results_{name}.csv")
    best_auc, best_f1, best_round = 0.0, 0.0, 1
    rounds_95   = None
    all_metrics = []
    bank_aucs   = []

    for rnd in range(start_round, HP["FL_ROUNDS"] + 1):
        t0           = time.time()
        global_state = copy.deepcopy(global_model.state_dict())
        norm_grads   = []
        tau_list     = []
        local_sizes  = []

        for bank_id, (X_b, y_b) in enumerate(bank_data):
            X_r, y_r = smote_bank(X_b, y_b)
            lm       = copy.deepcopy(global_model)
            loader   = make_loader(X_r, y_r)
            lm, aux  = local_train_nn(
                lm, loader, None, HP["LR"], privacy_mode,
                use_steps=True, n_steps=HP["LOCAL_STEPS"]
            )
            norm_grads.append(aux["norm_grad"])
            tau_list.append(aux["tau"])
            local_sizes.append(len(X_b))
            del lm, loader, aux
            torch.cuda.empty_cache()

        total_w   = sum(local_sizes[i] * tau_list[i] for i in range(n_banks))
        new_state = copy.deepcopy(global_state)
        for key in new_state:
            if new_state[key].dtype == torch.float32:
                agg = sum(
                    norm_grads[i][key].to(DEVICE) *
                    (local_sizes[i] * tau_list[i] / total_w)
                    for i in range(n_banks)
                )
                new_state[key] = global_state[key].to(DEVICE) + agg
            else:
                new_state[key] = sum(
                    norm_grads[i][key].to(DEVICE) * (local_sizes[i] / sum(local_sizes))
                    for i in range(n_banks)
                )
        global_model.load_state_dict(new_state)
        del norm_grads, new_state, global_state
        torch.cuda.empty_cache()

        probs = get_probs_nn(global_model, X_te)
        m     = compute_metrics(y_te, probs)
        loss  = float(-(y_te * np.log(probs + 1e-9) +
                        (1 - y_te) * np.log(1 - probs + 1e-9)).mean())

        bank_aucs = []
        for i, (X_b, y_b) in enumerate(bank_data):
            if len(np.unique(y_b)) < 2:
                bank_aucs.append(None)
                continue
            p = get_probs_nn(global_model, X_b)
            bank_aucs.append(round(roc_auc_score(y_b, p), 6))

        sigma, jfi = fairness_metrics(bank_aucs)

        if m["auc"] > best_auc:
            best_auc, best_f1, best_round = m["auc"], m["f1"], rnd
        if rounds_95 is None and m["auc"] >= 0.95 * best_auc:
            rounds_95 = rnd

        row = {"round": rnd, **m, "sigma_auc": sigma, "jfi": jfi,
               "loss": round(loss, 6), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        append_csv_row(csv_path, row, write_header=(rnd == start_round))
        all_metrics.append(row)
        save_checkpoint(path, name, rnd, global_model, m)

        print(f"  Round {rnd:03d}/{HP['FL_ROUNDS']:03d} | "
              f"AUC: {m['auc']:.4f} | F1: {m['f1']:.4f} | "
              f"R@1%FPR: {m['recall_1fpr']:.4f} | P@K: {m['prec_at_k']:.4f} | "
              f"KS: {m['ks_stat']:.4f} | σ: {sigma:.4f} | Loss: {loss:.4f}")

    return global_model, all_metrics, best_auc, best_f1, best_round, rounds_95, bank_aucs


# ══════════════════════════════════════════════════════════════════
#  RUN ONE COMBINATION
# ══════════════════════════════════════════════════════════════════

def run_combination(fl_algo, ml_model, privacy_mode,
                    X_tr, X_te, y_tr, y_te,
                    combo_idx, total_combos, master_csv):
    gc.collect()
    torch.cuda.empty_cache()

    path, name = combo_dir(fl_algo, ml_model, privacy_mode)

    print(f"\n{'='*60}")
    print(f"  {fl_algo} + {ml_model} + {privacy_mode}  "
          f"({combo_idx}/{total_combos})")
    print(f"{'='*60}")

    start_round, ckpt = find_latest_checkpoint(path, name)
    start_round += 1

    n_features   = X_tr.shape[1]
    bank_data    = non_iid_split(X_tr, y_tr)
    global_model = build_model(ml_model, n_features)

    if ckpt is not None:
        if isinstance(global_model, LRWrapper):
            global_model.set_params(ckpt["model_params"])
        else:
            global_model.load_state_dict(
                {k: v.to(DEVICE) for k, v in ckpt["model_state"].items()}
            )

    if isinstance(global_model, LRWrapper) and fl_algo in ["SCAFFOLD", "FedNova"]:
        print(f"  SKIP: LR not compatible with {fl_algo}")
        return None

    t_start = time.time()

    if fl_algo in ["FedAvg", "FedProx", "PersFL"]:
        _, all_m, best_auc, best_f1, best_round, rounds_95, bank_aucs = run_fedavg(
            global_model, bank_data, X_te, y_te,
            fl_algo, privacy_mode, path, name, master_csv,
            start_round=start_round
        )
    elif fl_algo == "SCAFFOLD":
        _, all_m, best_auc, best_f1, best_round, rounds_95, bank_aucs = run_scaffold(
            global_model, bank_data, X_te, y_te,
            privacy_mode, path, name, master_csv,
            start_round=start_round
        )
    elif fl_algo == "FedNova":
        if isinstance(global_model, LRWrapper):
            print("  SKIP: LR not compatible with FedNova")
            return None
        _, all_m, best_auc, best_f1, best_round, rounds_95, bank_aucs = run_fednova(
            global_model, bank_data, X_te, y_te,
            privacy_mode, path, name, master_csv,
            start_round=start_round
        )

    total_time = round(time.time() - t_start, 1)
    final_m    = all_m[-1] if all_m else {}
    sigma, jfi = fairness_metrics(bank_aucs)

    save_summary(path, name, best_round, best_auc, best_f1, final_m, bank_aucs)

    master_row = {
        "fl_algorithm"      : fl_algo,
        "ml_model"          : ml_model,
        "privacy_mode"      : privacy_mode,
        "best_auc"          : best_auc,
        "best_auc_round"    : best_round,
        "final_auc"         : final_m.get("auc", 0),
        "best_f1"           : best_f1,
        "best_recall_1fpr"  : max((r.get("recall_1fpr", 0) for r in all_m), default=0),
        "ks_stat"           : final_m.get("ks_stat", 0),
        "sigma_auc"         : sigma,
        "jfi"               : jfi,
        "rounds_to_95pct"   : rounds_95,
        "total_time_seconds": total_time,
    }
    append_master(master_row, master_csv)

    # Cleanup
    del global_model, bank_data
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n  ✓ Done | Best AUC: {best_auc:.4f} @ round {best_round} | Time: {total_time}s")
    return master_row


# FIX 5: Separate CUDA RuntimeError handling from all other exceptions
# so a CUDA crash on one combo doesn't silently swallow the real error.
# The broad "except Exception" in the original masked every failure with
# the same message, making it impossible to distinguish CUDA arch errors
# from logic bugs or OOM.
def run_combination_safe(fl_algo, ml_model, privacy_mode,
                         X_tr, X_te, y_tr, y_te,
                         combo_idx, total_combos, master_csv):
    gc.collect()
    torch.cuda.empty_cache()
    try:
        result = run_combination(
            fl_algo, ml_model, privacy_mode,
            X_tr, X_te, y_tr, y_te,
            combo_idx, total_combos, master_csv
        )
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"  ⚠ CUDA error on {fl_algo}+{ml_model}+{privacy_mode} — retrying on CPU...")
            # Move computation to CPU for this combo only, then restore global DEVICE
            global DEVICE
            _prev_device = DEVICE
            DEVICE = torch.device("cpu")
            try:
                result = run_combination(
                    fl_algo, ml_model, privacy_mode,
                    X_tr, X_te, y_tr, y_te,
                    combo_idx, total_combos, master_csv
                )
            except Exception as e2:
                print(f"  ✗ FAILED even on CPU: {fl_algo} + {ml_model} + {privacy_mode} | Error: {e2}")
                result = None
            finally:
                DEVICE = _prev_device  # always restore, even if CPU retry fails
        else:
            print(f"  ✗ FAILED: {fl_algo} + {ml_model} + {privacy_mode} | Error: {e}")
            result = None
    except Exception as e:
        print(f"  ✗ FAILED: {fl_algo} + {ml_model} + {privacy_mode} | Error: {e}")
        result = None
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return result
