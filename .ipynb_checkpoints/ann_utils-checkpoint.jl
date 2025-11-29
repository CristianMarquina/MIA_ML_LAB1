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

    println("\n    Print Confusion Matrix")
    printConfusionMatrix(y_pred_test_bool, y_test_encoded)
    
    acc_test  = metrics_test.accuracy * 100
    f1_test   = metrics_test.f_score
    sens_test = metrics_test.sensitivity
    spec_test = metrics_test.specificity
    prec_test = metrics_test.ppv
    
    println("  FINAL TEST RESULTS:")
    println("  Sens: $(round(sens_test, digits=4)) | Acc: $(round(acc_test, digits=2))% | F1: $(round(f1_test, digits=3))")

    return (best_topo, acc_test, f1_test, sens_test, spec_test, prec_test, best_time, best_model, metrics_test)
end

# ==============================================================================
# RUN GRID SEARCH (Topology + Hyperparameters)
# ==============================================================================
function run_grid_search_v2(dataset, approach_name, topologies, learning_rates)
    
    println("\n   🧠 Starting Grid Search (Topology + Learning Rate) for: $approach_name")
    
    # Desempaquetar datos
    x_train = dataset.x_train
    y_train_flux = dataset.y_train_flux
    x_val = dataset.x_val
    
    # Crear y_val_flux si no existe
    y_val_flux = Float32.(permutedims(dataset.y_val_encoded))
    
    x_test = dataset.x_test
    y_test_encoded = dataset.y_test_encoded
    y_test_flux = Float32.(permutedims(y_test_encoded))
    
    results = []
    
    total_iterations = length(topologies) * length(learning_rates)
    counter = 0

    # --- DOBLE BUCLE: TOPOLOGÍAS x LEARNING RATES ---
    for topo in topologies
        for lr in learning_rates
            counter += 1
            
            # Start timer
            start_time = time()
            
            # Le pasamos el 'lr' actual en cada vuelta
            ann, train_hist, val_hist, test_hist = trainClassANN(
                topo,
                (x_train, y_train_flux);
                validationDataset = (x_val, y_val_flux),
                testDataset = (x_test, y_test_flux),
                learningRate = lr,          
                maxEpochs = 1000,
                maxEpochsVal = 100,          # Early Stopping
                showText = false
            )
            
            elapsed = time() - start_time
            
            # Evaluar en Validación para elegir al ganador
            raw_preds_val = ann(x_val)
            y_pred_val_bool = classifyOutputs(permutedims(raw_preds_val))
            metrics_val = confusionMatrix(y_pred_val_bool, dataset.y_val_encoded)
            sens_val = metrics_val.sensitivity

           
            
            # Print de progreso limpio
            # @printf("   [%d/%d] Topo: %-10s | LR: %.3f | Val Sens: %.4f | Time: %.2fs\n", 
            #         counter, total_iterations, string(topo), lr, sens_val, elapsed)
            
            # Guardamos el LR también en los resultados
            push!(results, (topo, lr, sens_val, elapsed, ann, train_hist, val_hist))
        end
    end
    
    # --- SELECCIONAR GANADOR ---
    # Ordenamos por Sensibilidad en Validación (índice 3)
    sort!(results, by = x -> x[3], rev = true)
    
    winner = results[1]
    (best_topo, best_lr, best_sens_val, best_time, best_model, best_train_hist, best_val_hist) = winner
    
    println("\n      🏆 WINNER FOUND:")
    println("         Topology:      $best_topo")
    println("         Learning Rate: $best_lr")
    println("         Val Sens:      $(round(best_sens_val, digits=4))")

    # --- AUDITORÍA FINAL (TEST) ---
    println("      ⚖️  Running Final Audit on TEST SET...")
    raw_preds_test = best_model(x_test)
    y_pred_test_bool = classifyOutputs(permutedims(raw_preds_test))
    
    metrics_test = confusionMatrix(y_pred_test_bool, y_test_encoded)
    
    acc_test  = metrics_test.accuracy * 100
    f1_test   = metrics_test.f_score
    sens_test = metrics_test.sensitivity
    spec_test = metrics_test.specificity
    prec_test = metrics_test.ppv
    
    println("      ✅ FINAL TEST RESULTS:")
    println("         Sens: $(round(sens_test, digits=4)) | Acc: $(round(acc_test, digits=2))% | F1: $(round(f1_test, digits=3))")

    # Devolvemos la tupla (añadimos best_lr para que quede registrado)
    return (best_topo, acc_test, f1_test, sens_test, spec_test, prec_test, best_time, best_model, metrics_test, best_train_hist, best_val_hist, best_lr)
end

# ==============================================================================
# RUN GRID SEARCH V2 (Compatible con Unit 3)
# ==============================================================================
# ==============================================================================
# RUN GRID SEARCH V3 (Topology + Learning Rate + Boolean Fix for trainClassANN)
# ==============================================================================
function run_grid_search_v3(dataset, approach_name, topologies, learning_rates)
    
    println("\n   🧠 Starting Grid Search (Topology + Learning Rate) for: $approach_name")
    
    # 1. Desempaquetar datos
    # x_train: Matrix{Float32} (Features x Samples)
    x_train = dataset.x_train
    
    # --- FIX DE TIPOS CRÍTICO PARA trainClassANN ---
    # trainClassANN exige targets Booleanos (BitMatrix) en la tupla de dataset.
    # dataset.y_train_flux es Float32, así que NO lo usamos aquí.
    # Usamos dataset.y_train_encoded (BitMatrix) y lo transponemos para que sea (Classes x Samples).
    y_train_bool = permutedims(dataset.y_train_encoded)
    
    # Lo mismo para Validación
    x_val = dataset.x_val
    y_val_bool = permutedims(dataset.y_val_encoded) # Versión Booleana para validationDataset
    
    # Lo mismo para Test
    x_test = dataset.x_test
    y_test_bool = permutedims(dataset.y_test_encoded) # Versión Booleana para testDataset
    
    
    results = []
    total_iterations = length(topologies) * length(learning_rates)
    counter = 0

    # --- DOBLE BUCLE: TOPOLOGÍAS x LEARNING RATES ---
    for topo in topologies
        for lr in learning_rates
            counter += 1
            
            start_time = time()
            
            # Llamamos a trainClassANN pasando los targets BOOLEANOS
            # (x_train, y_train_bool) cumple con la firma Tuple{Real, Bool}
            ann, train_hist, val_hist, test_hist = trainClassANN(
                topo,
                (x_train, y_train_bool);           # Training (Bool)
                validationDataset = (x_val, y_val_bool), # Validation (Bool)
                testDataset = (x_test, y_test_bool),     # Test (Bool)
                learningRate = lr,          
                maxEpochs = 1000,
                maxEpochsVal = 100,          
                showText = false # Silenciar logs internos para no saturar la pantalla
            )
            
            elapsed = time() - start_time
            
            # --- EVALUACIÓN (Usando Validación para elegir ganador) ---
            # Predecimos sobre el set de validación
            raw_preds_val = ann(x_val)
            # Convertimos probabilidades a booleanos (Clasificación dura)
            y_pred_val_bool = classifyOutputs(permutedims(raw_preds_val))
            
            # Calculamos métricas
            metrics_val = confusionMatrix(y_pred_val_bool, dataset.y_val_encoded)
            sens_val = metrics_val.sensitivity
            
            # Log de progreso simple
             @printf("      [%d/%d] Topo: %-10s | LR: %.3f | Val Sens: %.4f | Time: %.2fs\n", 
                    counter, total_iterations, string(topo), lr, sens_val,  elapsed)
            
            # Guardamos todo el historial
            push!(results, (topo, lr, sens_val, elapsed, ann, train_hist, val_hist))
        end
    end
    
    # --- SELECCIONAR GANADOR ---
    # Ordenamos por Sensibilidad en Validación (índice 3)
    sort!(results, by = x -> x[3], rev = true)
    
    winner = results[1]
    (best_topo, best_lr, best_sens_val, best_time, best_model, best_train_hist, best_val_hist) = winner
    
    println("\n      🏆 WINNER FOUND:")
    println("         Topology:      $best_topo")
    println("         Learning Rate: $best_lr")
    println("         Val Sens:      $(round(best_sens_val, digits=4))")

    # --- AUDITORÍA FINAL (TEST) ---
    println("      ⚖️  Running Final Audit on TEST SET...")
    
    # Predicción final en Test con el mejor modelo
    raw_preds_test = best_model(x_test)
    y_pred_test_bool = classifyOutputs(permutedims(raw_preds_test))
    
    # Métricas Finales Reales (Unit 4)
    metrics_test = confusionMatrix(y_pred_test_bool, dataset.y_test_encoded)
    
    acc_test  = metrics_test.accuracy * 100
    f1_test   = metrics_test.f_score
    sens_test = metrics_test.sensitivity
    spec_test = metrics_test.specificity
    prec_test = metrics_test.ppv
    
    println("      ✅ FINAL TEST RESULTS:")
    println("         Sens: $(round(sens_test, digits=4)) | Acc: $(round(acc_test, digits=2))% | F1: $(round(f1_test, digits=3))")

    # Devolvemos una tupla masiva con todo lo necesario para graficar y guardar
    return (best_topo, acc_test, f1_test, sens_test, spec_test, prec_test, best_time, best_model, metrics_test, best_train_hist, best_val_hist, best_lr)
end

function run_grid_search_v3(dataset, approach_name, topologies, learning_rates)
    
    println("\n   🧠 Starting Grid Search (Topology + Learning Rate) for: $approach_name")
    
    # 1. Desempaquetar datos (Vienen como Samples x Features)
    x_train = dataset.x_train
    y_train_bool = dataset.y_train_encoded # BitMatrix (Samples x Classes)
    
    x_val = dataset.x_val
    y_val_bool = dataset.y_val_encoded
    
    x_test = dataset.x_test
    y_test_bool = dataset.y_test_encoded
    
    results = []
    total_iterations = length(topologies) * length(learning_rates)
    counter = 0

    # --- DOBLE BUCLE: TOPOLOGÍAS x LEARNING RATES ---
    for topo in topologies
        for lr in learning_rates
            counter += 1
            start_time = time()
            
            # A. ENTRENAMIENTO (trainClassANN maneja la transposición interna)
            # Le pasamos (Samples x Features) y (Samples x Classes)
            ann, train_hist, val_hist, test_hist = trainClassANN(
                topo,
                (x_train, y_train_bool);           
                validationDataset = (x_val, y_val_bool),
                testDataset = (x_test, y_test_bool),
                learningRate = lr,          
                maxEpochs = 1000,
                maxEpochsVal = 100,          
                showText = false 
            )
            
            elapsed = time() - start_time
            
            # B. EVALUACIÓN (AQUÍ SÍ TRANSPONEMOS MANUALMENTE PARA FLUX)
            # Flux necesita (Features x Samples), así que usamos permutedims(x)'
            # O simplemente x_val' (la comilla transpone)
            
            # Predicción: model(x_val') -> Devuelve (Classes x Samples)
            raw_preds_val = ann(permutedims(x_val)) 
            
            # Clasificación: Transponemos de vuelta a (Samples x Classes) para métricas
            y_pred_val_bool = classifyOutputs(permutedims(raw_preds_val))
            
            metrics_val = confusionMatrix(y_pred_val_bool, dataset.y_val_encoded)
            sens_val = metrics_val.sensitivity
            
            @printf("      [%d/%d] Topo: %-10s | LR: %.3f | Val Sens: %.4f |  Time: %.2fs\n", 
                    counter, total_iterations, string(topo), lr, sens_val,   elapsed)
            
            push!(results, (topo, lr, sens_val, elapsed, ann, train_hist, val_hist))
        end
    end
    
    # ... (El resto de la selección del ganador sigue igual) ...
    sort!(results, by = x -> x[3], rev = true)
    winner = results[1]
    (best_topo, best_lr, best_sens_val, best_time, best_model, best_train_hist, best_val_hist) = winner
    
    println("\n      🏆 WINNER FOUND: $best_topo (Sens: $best_sens_val)")

    # --- AUDITORÍA FINAL ---
    # Transponemos x_test para Flux
    raw_preds_test = best_model(permutedims(x_test))
    y_pred_test_bool = classifyOutputs(permutedims(raw_preds_test))
    
    metrics_test = confusionMatrix(y_pred_test_bool, dataset.y_test_encoded)
    acc_test  = metrics_test.accuracy * 100
    f1_test   = metrics_test.f_score
    sens_test = metrics_test.sensitivity
    spec_test = metrics_test.specificity
    prec_test = metrics_test.ppv
    
    println("      ✅ FINAL TEST RESULTS: Sens $(round(sens_test, digits=4)) | Acc $(round(acc_test, digits=2))%")

    return (best_topo, acc_test, f1_test, sens_test, spec_test, prec_test, best_time, best_model, metrics_test, best_train_hist, best_val_hist, best_lr)
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

"""
    export_standardized_results_v2(winner_tuple, dataset, approach_name)

Exports the results in the STRICT standardized dictionary format for the Ensemble.
Converts Multiclass ANN outputs (0-4) to Binary (0=Healthy, 1=Sick) as required.
Includes fix for matrix dimensionality (Transposition for Flux).
"""
function export_standardized_results_v2(winner_tuple, dataset, approach_name)
    
    # 1. Desempaquetar
    # Recuerda que winner_tuple tiene muchas cosas, pero el modelo está en la posición 8
    # (Topo, Acc, F1, Sens, Spec, Prec, Time, MODEL, ...)
    (topo, _, _, _, _, _, time_sec, model) = winner_tuple
    
    clean_name = replace(approach_name, " " => "") 
    model_id = "ANN_$(clean_name)_Topo$topo"
    
    println("   📦 Standardizing results V2 for Ensemble ($model_id)...")

    # -----------------------------------------------------------
    # 2. GENERAR PREDICCIONES (FIX: TRANSPOSICIÓN)
    # -----------------------------------------------------------
    
    # IMPORTANTE: Transponemos x_test para que Flux reciba (Features x Samples)
    x_test_flux = permutedims(dataset.x_test)
    raw_probs = model(x_test_flux)
    
    # Probabilidad de estar enfermo = 1.0 - Probabilidad de estar Sano (Clase 0, fila 1)
    probs_healthy = raw_probs[1, :] 
    y_scores = 1.0f0 .- probs_healthy 
    
    # Clase Binaria (0 o 1)
    y_pred_binary = Int.(y_scores .>= 0.5)
    
    # Ground Truth Binario
    y_test_flux_target = permutedims(dataset.y_test_encoded)
    true_indices = Flux.onecold(y_test_flux_target) # 1..5
    y_test_binary = [idx == 1 ? 0 : 1 for idx in true_indices]

    # -----------------------------------------------------------
    # 3. MÉTRICAS BINARIAS
    # -----------------------------------------------------------
    y_p_bool = Bool.(y_pred_binary)
    y_t_bool = Bool.(y_test_binary)
    
    (bin_acc, bin_err, bin_sens, bin_spec, bin_ppv, bin_npv, bin_f1, bin_cm) = confusionMatrix(y_p_bool, y_t_bool)
    
    calc_auc = 0.0 

    # -----------------------------------------------------------
    # 4. GUARDADO
    # -----------------------------------------------------------
    resultados_modelo = Dict(
        "nombre_modelo" => model_id,
        "y_test"        => y_test_binary,    
        "y_pred"        => y_pred_binary,    
        "y_scores"      => y_scores,         
        "metricas"      => Dict(
            "accuracy"          => bin_acc,
            "recall"            => bin_sens, 
            "specificity"       => bin_spec,
            "precision"         => bin_ppv,  
            "f1_score"          => bin_f1,
            "auc_roc"           => calc_auc,
            "training_time_sec" => time_sec
        )
    )

    filename = "Resultados_ANN_$(clean_name).jld2"
    JLD2.save(filename, "resultados_modelo", resultados_modelo)
    
    println("      ✅ EXPORT SUCCESS: $filename")
    println("         Binary Metrics -> Acc: $(round(bin_acc, digits=3)) | F1: $(round(bin_f1, digits=3))")
end

"""
    save_approach_results_v2(winner_tuple, dataset, approach_name)

Saves the physical ANN model and raw predictions.
Includes fix for matrix dimensionality (Transposition for Flux).
"""
function save_approach_results_v2(winner_tuple, dataset, approach_name)
    println("\n   💾 Saving artifacts V2 for $approach_name...")
    
    (topo, acc, f1, sens, spec, prec, time, model) = winner_tuple
    clean_name = replace(approach_name, " " => "")
    classes_ordered = [0, 1, 2, 3, 4]

    # A. Save Physical Model
    file_model = "Modelo_ANN_$(clean_name)_Final.jld2"
    JLD2.save(file_model, "ann_model", model)
    println("      ✅ Model Saved: $file_model")
    
    # B. Generate Predictions (FIX: TRANSPOSICIÓN)
    x_test_flux = permutedims(dataset.x_test)
    raw_preds = model(x_test_flux)
    y_pred_vec = Flux.onecold(raw_preds, classes_ordered)
    
    y_test_T = permutedims(dataset.y_test_encoded)
    y_test_vec = Flux.onecold(y_test_T, classes_ordered)
    
    # C. Save Predictions Dictionary
    save_dict_preds = Dict(
        "modelo_info" => "ANN Topo:$topo ($clean_name)",
        "y_pred"      => y_pred_vec,
        "y_test"      => y_test_vec,
        "accuracy"    => acc / 100.0
    )
    file_preds = "Predicciones_ANN_$(clean_name).jld2"
    JLD2.save(file_preds, "ANN_preds", save_dict_preds)
    println("      ✅ Preds Saved: $file_preds")
end
# ==============================================================================
# RUN GRID SEARCH V3 (CORREGIDO: SIN TRANSPONER TARGETS)
# ==============================================================================
function run_grid_search_v4(dataset, approach_name, topologies, learning_rates)
    
    println("\n   🧠 Starting Grid Search (Topology + Learning Rate) for: $approach_name")
    
    # 1. Desempaquetar datos (Ya vienen como Samples x Features)
    x_train = dataset.x_train
    
    # --- CORRECCIÓN AQUÍ: NO USAR PERMUTEDIMS ---
    # trainClassANN espera que X e Y tengan el mismo número de filas (samples).
    # Como x_train ya tiene filas=samples, y_train_encoded también debe tenerlo.
    y_train_bool = dataset.y_train_encoded # BitMatrix (Samples x Classes)
    
    x_val = dataset.x_val
    y_val_bool = dataset.y_val_encoded # Sin permutedims
    
    x_test = dataset.x_test
    y_test_bool = dataset.y_test_encoded # Sin permutedims
    
    
    results = []
    total_iterations = length(topologies) * length(learning_rates)
    counter = 0

    # --- DOBLE BUCLE: TOPOLOGÍAS x LEARNING RATES ---
    for topo in topologies
        for lr in learning_rates
            counter += 1
            start_time = time()
            
            # A. ENTRENAMIENTO (trainClassANN maneja la transposición interna)
            ann, train_hist, val_hist, test_hist = trainClassANN(
                topo,
                (x_train, y_train_bool);           
                validationDataset = (x_val, y_val_bool),
                testDataset = (x_test, y_test_bool),
                learningRate = lr,          
                maxEpochs = 1000,
                maxEpochsVal = 100,          
                showText = false 
            )
            
            elapsed = time() - start_time
            
            # B. EVALUACIÓN (AQUÍ SÍ TRANSPONEMOS MANUALMENTE PARA FLUX)
            # Flux necesita (Features x Samples), así que usamos permutedims(x)
            
            # Predicción: model(permutedims(x)) -> Devuelve (Classes x Samples)
            raw_preds_val = ann(permutedims(x_val)) 
            
            # Clasificación: Transponemos de vuelta a (Samples x Classes) para métricas
            y_pred_val_bool = classifyOutputs(permutedims(raw_preds_val))
            
            metrics_val = confusionMatrix(y_pred_val_bool, dataset.y_val_encoded)
            sens_val = metrics_val.sensitivity
            
            # Guardamos
            push!(results, (topo, lr, sens_val, elapsed, ann, train_hist, val_hist))
        end
    end
    
    # --- SELECCIONAR GANADOR ---
    sort!(results, by = x -> x[3], rev = true)
    winner = results[1]
    (best_topo, best_lr, best_sens_val, best_time, best_model, best_train_hist, best_val_hist) = winner
    
    println("\n      🏆 WINNER FOUND: $best_topo (Sens: $best_sens_val)")

    # --- AUDITORÍA FINAL ---
    # Transponemos x_test para Flux
    raw_preds_test = best_model(permutedims(x_test))
    y_pred_test_bool = classifyOutputs(permutedims(raw_preds_test))
    
    metrics_test = confusionMatrix(y_pred_test_bool, dataset.y_test_encoded)
    acc_test  = metrics_test.accuracy * 100
    f1_test   = metrics_test.f_score
    sens_test = metrics_test.sensitivity
    spec_test = metrics_test.specificity
    prec_test = metrics_test.ppv
    
    println("      ✅ FINAL TEST RESULTS: Sens $(round(sens_test, digits=4)) | Acc $(round(acc_test, digits=2))%")

    return (best_topo, acc_test, f1_test, sens_test, spec_test, prec_test, best_time, best_model, metrics_test, best_train_hist, best_val_hist, best_lr)
end