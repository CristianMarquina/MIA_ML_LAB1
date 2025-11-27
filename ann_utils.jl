using Flux
using JLD2, FileIO
using Statistics
using Random
using Printf
using CategoricalArrays
using Plots


"""
    load_and_preprocess(filename::String)
"""
function load_and_preprocess(filename::String)
    println(" Loading file: $filename ...")
    data = load(filename)

    #Extract
    x_train_raw = copy(data["x_train"])
    y_train_raw = copy(data["y_train"])
    x_val_raw   = copy(data["x_val"])    
    y_val_raw   = copy(data["y_val"])   
    x_test_raw  = copy(data["x_test"])
    y_test_raw  = copy(data["y_test"])
    
    # Transpose Inputs
    x_train = Float32.(permutedims(x_train_raw))
    x_val   = Float32.(permutedims(x_val_raw)) 
    x_test  = Float32.(permutedims(x_test_raw))
    
    # Target Encoding
    classes = [0, 1, 2, 3, 4]
    
    # One-Hot
    y_train_encoded = oneHotEncoding(y_train_raw, classes)
    y_val_encoded   = oneHotEncoding(y_val_raw, classes)  
    y_test_encoded  = oneHotEncoding(y_test_raw, classes)

    y_train_flux = Float32.(permutedims(y_train_encoded))
    
    return (
        x_train=x_train, y_train_flux=y_train_flux,
        x_val=x_val,     y_val_encoded=y_val_encoded,  
        x_test=x_test,   y_test_encoded=y_test_encoded
    )
end

"""
    generate_clinical_plots(metrics, approach_name)

Generates clinical visualizations:
1. Confusion Matrix Heatmap with EXPLICIT COUNTS in each cell.
2. Sensitivity vs Specificity Balance Chart.
"""
function generate_clinical_plots(metrics, approach_name)
    
    # --- PLOT 1: CONFUSION MATRIX HEATMAP ---
    cm = metrics.confusion_matrix
    n_classes = size(cm, 1)
    
    # Generar las anotaciones (Texto con el número)
    # cm[i,j] es el valor (recuento) en la fila i (Real), columna j (Predicha)
    # Posición (j, i) pone el texto en la coordenada correcta del gráfico
    anns = [(j, i, text(string(cm[i,j]), 10, :black, :center)) 
            for i in 1:n_classes, j in 1:n_classes]
    
    p1 = heatmap(cm, 
        title="$approach_name\nConfusion Matrix",
        xlabel="Predicted Class", 
        ylabel="True Class",
        color=:blues,
        yflip=true, # Pone la clase 0 arriba (estándar visual)
        annotations=vec(anns), # <--- vec() asegura que Plots lo lea bien
        aspect_ratio=:equal
    )

    # --- PLOT 2: BALANCE SENSITIVITY VS SPECIFICITY ---
    sens = metrics.per_class_metrics.sensitivities
    spec = metrics.per_class_metrics.specificities
    
    classes = ["0", "1", "2", "3", "4"]
    data = [sens spec]
    
    p2 = bar(classes, data,
        title="Clinical Balance per Class",
        label=["Sensitivity (Recall)" "Specificity"],
        xlabel="Class (Severity)",
        ylabel="Score (0-1)",
        lw=0,
        alpha=0.8,
        color=[:green :red],
        legend=:bottomright
    )
    
    # --- DISPLAY COMBINED (VERTICAL STACK) ---
    display(plot(p1, p2, layout=(2, 1), size=(800, 1000), margin=5Plots.mm))
end

function run_topology_search(dataset, approach_name, topologies)
    
    println("\n  Starting Topology Search for: $approach_name")
    
    x_train = dataset.x_train
    y_train_flux = dataset.y_train_flux
    x_val = dataset.x_val
    y_val_encoded = dataset.y_val_encoded
    n_inputs  = size(x_train, 1)
    n_outputs = 5
    epochs    = 1000
    
    results = []

    for (i, topology) in enumerate(topologies)
        
        # Build
        model = buildClassANN(n_inputs, topology, n_outputs)
        loss(m, x, y) = Flux.crossentropy(m(x), y)
        opt_state = Flux.setup(Flux.Adam(0.01), model)
        
        #  Train
        start_time = time()
        for epoch in 1:epochs
            Flux.train!(loss, model, [(x_train, y_train_flux)], opt_state)
        end
        elapsed = time() - start_time
        
        # Evaluation
        raw_preds_val = model(x_val)
        y_pred_val_bool = classifyOutputs(permutedims(raw_preds_val))
        
        # Get metrics
        metrics_val = confusionMatrix(y_pred_val_bool, y_val_encoded)
        sens_val = metrics_val.sensitivity # Criterio de selección
        
        # Save
        push!(results, (topology, sens_val, elapsed, model))
    end
    
    # Get the best
    sort!(results, by = x -> x[2], rev = true)
    
    winner = results[1]
    (best_topo, best_sens_val, best_time, best_model) = winner
    
    println("\n     WINNER SELECTED (Based on Validation): $best_topo")
    println("      Validation Sensitivity: $(round(best_sens_val, digits=4))")

    println("   Running Final Audit on TEST SET...")
    
    x_test = dataset.x_test
    y_test_encoded = dataset.y_test_encoded
    
    # Predecir en Test con el ganador
    raw_preds_test = best_model(x_test)
    y_pred_test_bool = classifyOutputs(permutedims(raw_preds_test))
    
    # Metrics with test data
    metrics_test = confusionMatrix(y_pred_test_bool, y_test_encoded)
    
    acc_test  = metrics_test.accuracy * 100
    f1_test   = metrics_test.f_score
    sens_test = metrics_test.sensitivity
    spec_test = metrics_test.specificity
    prec_test = metrics_test.ppv
    
    println("  FINAL TEST RESULTS:")
    println("  Sens: $(round(sens_test, digits=4)) | Acc: $(round(acc_test, digits=2))% | F1: $(round(f1_test, digits=3))")

    return (best_topo, acc_test, f1_test, sens_test, spec_test, prec_test, best_time, best_model, metrics_test)
end


function save_approach_results(winner_tuple, dataset, approach_name)
    println("\n Saving artifacts for $approach_name...")
    
    (topo, acc, f1, sens, spec, prec, time, model) = winner_tuple
    clean_name = replace(approach_name, " " => "")
    classes_ordered = [0, 1, 2, 3, 4]

    # Save Model
    file_model = "Modelo_ANN_$(clean_name)_Final.jld2"
    JLD2.save(file_model, "ann_model", model)
    println("Model Saved: $file_model")
    
    # Generate Predictions 
    raw_preds = model(dataset.x_test)
    y_pred_vec = Flux.onecold(raw_preds, classes_ordered)
    
    y_test_T = permutedims(dataset.y_test_encoded)
    y_test_vec = Flux.onecold(y_test_T, classes_ordered)
    
    # Save Preds
    save_dict_preds = Dict(
        "modelo_info" => "ANN Topo:$topo ($clean_name)",
        "y_pred"      => y_pred_vec,
        "y_test"      => y_test_vec,
        "accuracy"    => acc / 100.0
    )
    file_preds = "Predicciones_ANN_$(clean_name).jld2"
    JLD2.save(file_preds, "ANN_preds", save_dict_preds)
    println(" Preds Saved: $file_preds")
end

"""
    export_standardized_results(winner_tuple, dataset, approach_name)

Exports the results in the STRICT standardized dictionary format for the Ensemble.
Converts Multiclass ANN outputs (0-4) to Binary (0=Healthy, 1=Sick) as required.

Uses 'unit4-metrics.jl' functions where applicable, but recalculates binary metrics
to ensure 'recall', 'precision', etc. refer specifically to the Positive Class (Sick).
"""
function export_standardized_results(winner_tuple, dataset, approach_name)
    
    (topo, _, _, _, _, _, time_sec, model) = winner_tuple
    
    clean_name = replace(approach_name, " " => "") 
    model_id = "ANN_$(clean_name)_Topo$topo"
    
    println("  Standardizing results for Ensemble ($model_id)...")

   
    raw_probs = model(dataset.x_test)
    probs_healthy = raw_probs[1, :] 
    y_scores = 1.0f0 .- probs_healthy # Vector de Float32
    
    y_pred_binary = Int.(y_scores .>= 0.5)
    
    y_test_flux = permutedims(dataset.y_test_encoded)
    true_indices = Flux.onecold(y_test_flux) # Devuelve 1..5
    
    y_test_binary = [idx == 1 ? 0 : 1 for idx in true_indices]

    y_p_bool = Bool.(y_pred_binary)
    y_t_bool = Bool.(y_test_binary)
    
    (bin_acc, bin_err, bin_sens, bin_spec, bin_ppv, bin_npv, bin_f1, bin_cm) = confusionMatrix(y_p_bool, y_t_bool)
    
    calc_auc = 0.0 

    # build dic
    resultados_modelo = Dict(
        "nombre_modelo" => model_id,
        "y_test"        => y_test_binary,    
        "y_pred"        => y_pred_binary,    
        "y_scores"      => y_scores,         
        "metricas"      => Dict(
            "accuracy"          => bin_acc,
            "recall"            => bin_sens, #
            "specificity"       => bin_spec,
            "precision"         => bin_ppv,  
            "f1_score"          => bin_f1,
            "auc_roc"           => calc_auc,
            "training_time_sec" => time_sec
        )
    )

 
    filename = "Resultados_ANN_$(clean_name).jld2"
    
    JLD2.save(filename, "resultados_modelo", resultados_modelo)
    
    println("   EXPORT SUCCESS: $filename")
    println("    Métricas Binarias -> Acc: $(round(bin_acc, digits=3)) | F1: $(round(bin_f1, digits=3))")
    println(resultados_modelo)
end
function process_single_approach(filepath, approach_name, topologies_list)
    println("\n======================================================================")
    println("🚀 PROCESSING APPROACH: $approach_name")
    println("======================================================================")
    
    # 1. Load (Train/Val/Test)
    dataset = load_and_preprocess(filepath)
    
    
    # 2. Train & Search (Use Val to pick winner, Test to report)
    winner_tuple = run_topology_search(dataset, approach_name, topologies_list)

    metrics_object = winner_tuple[9]
    
    # 3. GENERAR GRÁFICOS (Aquí es donde se llama ahora)
    println("\n  Generating Clinical Plots for $approach_name...")
    generate_clinical_plots(metrics_object, approach_name)

    export_standardized_results(winner_tuple, dataset, approach_name)

    save_approach_results(winner_tuple, dataset, approach_name)
    
    # 3. Save (Winner model & Test predictions)
    save_approach_results(winner_tuple, dataset, approach_name)
    
    println("\nCOMPLETED: $approach_name")
    return winner_tuple
end