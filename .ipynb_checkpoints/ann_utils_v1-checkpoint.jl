using Flux
using JLD2, FileIO
using Statistics
using Random
using Printf
using CategoricalArrays
using Plots


"""
    generate_clinical_plots(metrics, approach_name)

Generates and displays clinical visualizations using Plots.jl:
1. Confusion Matrix Heatmap with explicit counts per cell.
2. Clinical Balance Chart comparing Sensitivity and Specificity per class.

# Arguments
- `metrics`: Metrics object containing confusion matrix and per-class stats.
- `approach_name`: String identifier for the approach (e.g., "MinMax").
"""
function generate_clinical_plots(metrics, approach_name)
    
    # Plot 1: Confusion Matrix Heatmap 
    cm = metrics.confusion_matrix
    n_classes = size(cm, 1)
    
    # Generate text annotations for the heatmap cells
    anns = [(j, i, text(string(cm[i,j]), 10, :black, :center)) 
            for i in 1:n_classes, j in 1:n_classes]
    
    p1 = heatmap(cm, 
        title="$approach_name\nConfusion Matrix",
        xlabel="Predicted Class", 
        ylabel="True Class",
        color=:blues,
        yflip=true, # Ensure Class 0 is at the top
        annotations=vec(anns),
        aspect_ratio=:equal
    )

    # Plot 2: Sensitivity vs Specificity Balance
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
    
    # Display combined plot (Vertical Layout)
    display(plot(p1, p2, layout=(2, 1), size=(800, 1000), margin=5Plots.mm))
end

"""
    run_topology_search(dataset, approach_name, topologies)

Legacy function for topology search without hyperparameter tuning.
Trains models for each topology and selects the winner based on Validation Sensitivity.
"""
function run_topology_search(dataset, approach_name, topologies)
    
    println("\n Starting Topology Search for: $approach_name")
    
    x_train = dataset.x_train
    y_train_flux = dataset.y_train_flux
    x_val = dataset.x_val
    y_val_encoded = dataset.y_val_encoded
    n_inputs  = size(x_train, 1)
    n_outputs = 5
    epochs    = 1000
    
    results = []

    for (i, topology) in enumerate(topologies)
        
        # Build Model
        model = buildClassANN(n_inputs, topology, n_outputs)
        loss(m, x, y) = Flux.crossentropy(m(x), y)
        opt_state = Flux.setup(Flux.Adam(0.01), model)
        
        # Training Loop
        start_time = time()
        for epoch in 1:epochs
            Flux.train!(loss, model, [(x_train, y_train_flux)], opt_state)
        end
        elapsed = time() - start_time
        
        # Validation Evaluation
        raw_preds_val = model(x_val)
        y_pred_val_bool = classifyOutputs(permutedims(raw_preds_val))
        
        metrics_val = confusionMatrix(y_pred_val_bool, y_val_encoded)
        sens_val = metrics_val.sensitivity 
        
        push!(results, (topology, sens_val, elapsed, model))
    end
    
    # Select Winner (Sort by Sensitivity)
    sort!(results, by = x -> x[2], rev = true)
    
    winner = results[1]
    (best_topo, best_sens_val, best_time, best_model) = winner
    
    println("\n      WINNER SELECTED (Based on Validation): $best_topo")
    println("      Validation Sensitivity: $(round(best_sens_val, digits=4))")

    println("   Running Final Audit on TEST SET...")
    
    x_test = dataset.x_test
    y_test_encoded = dataset.y_test_encoded
    
    # Final Test Prediction
    raw_preds_test = best_model(x_test)
    y_pred_test_bool = classifyOutputs(permutedims(raw_preds_test))
    
    # Final Metrics Calculation
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

"""
    run_grid_search_v4(dataset, approach_name, topologies, learning_rates)

Performs an exhaustive Grid Search over ANN topologies and Learning Rates.
Incorporates correct type handling for the training function (Boolean Targets).
Selects the best model based on Validation Sensitivity and performs a final audit on the Test set.

# Returns
A tuple containing the best model configuration, performance metrics, and history.
"""
function run_grid_search_v4(dataset, approach_name, topologies, learning_rates)
    
    println("\n   Starting Grid Search (Topology + Learning Rate) for: $approach_name")
    
    # Unpack Data
    x_train = dataset.x_train
    
    # Target Handling Fix: 'trainClassANN' requires BitMatrix (Boolean) targets
    # We use 'y_train_encoded' which is BitMatrix, instead of 'y_train_flux' (Float32)
    y_train_bool = dataset.y_train_encoded 
    
    x_val = dataset.x_val
    y_val_bool = dataset.y_val_encoded 
    
    x_test = dataset.x_test
    y_test_bool = dataset.y_test_encoded 
    
    results = []
    total_iterations = length(topologies) * length(learning_rates)
    counter = 0

    # --- Nested Loop: Topologies x Learning Rates ---
    for topo in topologies
        for lr in learning_rates
            counter += 1
            start_time = time()
            
            # Train ANN using the provided training function
            # Note: Input data is passed as (Samples x Features) as required by 'trainClassANN'
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
            
            # --- Evaluation on Validation Set ---
            # Transpose input for Flux prediction: (Features x Samples)
            raw_preds_val = ann(permutedims(x_val)) 
            
            # Transpose output back for metrics: (Samples x Classes)
            y_pred_val_bool = classifyOutputs(permutedims(raw_preds_val))
            
            metrics_val = confusionMatrix(y_pred_val_bool, dataset.y_val_encoded)
            sens_val = metrics_val.sensitivity
            
            @printf("      [%d/%d] Topo: %-10s | LR: %.3f | Val Sens: %.4f | Time: %.2fs\n", 
                    counter, total_iterations, string(topo), lr, sens_val, elapsed)
            
            push!(results, (topo, lr, sens_val, elapsed, ann, train_hist, val_hist))
        end
    end
    
    # --- Select Winner ---
    # Sort results by Validation Sensitivity (Index 3)
    sort!(results, by = x -> x[3], rev = true)
    
    winner = results[1]
    (best_topo, best_lr, best_sens_val, best_time, best_model, best_train_hist, best_val_hist) = winner
    
    println("\n      WINNER FOUND: $best_topo (Sens: $best_sens_val)")

    # --- Final Audit on Test Set ---
    println("      Running Final Audit on TEST SET...")
    
    # Transpose Test data for Flux prediction
    raw_preds_test = best_model(permutedims(x_test))
    y_pred_test_bool = classifyOutputs(permutedims(raw_preds_test))
    
    metrics_test = confusionMatrix(y_pred_test_bool, dataset.y_test_encoded)
    
    acc_test  = metrics_test.accuracy * 100
    f1_test   = metrics_test.f_score
    sens_test = metrics_test.sensitivity
    spec_test = metrics_test.specificity
    prec_test = metrics_test.ppv
    
    println("      FINAL TEST RESULTS: Sens $(round(sens_test, digits=4)) | Acc $(round(acc_test, digits=2))%")

    return (best_topo, acc_test, f1_test, sens_test, spec_test, prec_test, best_time, best_model, metrics_test, best_train_hist, best_val_hist, best_lr)
end

"""
    run_grid_search_v3(dataset, approach_name, topologies, learning_rates)

Performs an exhaustive Grid Search over ANN topologies and Learning Rates.
It handles type compatibility for the training function (Boolean Targets) and manages matrix transpositions for Flux.
Selects the best model based on Validation Sensitivity and performs a final audit on the Test set.

# Returns
A tuple containing the best model configuration, performance metrics, and history.
"""
function run_grid_search_v3(dataset, approach_name, topologies, learning_rates)
    
    println("\n   Starting Grid Search (Topology + Learning Rate) for: $approach_name")
    
    # 1. Unpack Data
    x_train = dataset.x_train
    
    # Target Handling: 'trainClassANN' requires BitMatrix (Boolean) targets
    # We pass 'y_train_encoded' which is BitMatrix. 
    # Note: 'trainClassANN' expects (Samples x Features/Classes) and handles transposition internally.
    y_train_bool = dataset.y_train_encoded 
    
    x_val = dataset.x_val
    y_val_bool = dataset.y_val_encoded 
    
    x_test = dataset.x_test
    y_test_bool = dataset.y_test_encoded 
    
    results = []
    total_iterations = length(topologies) * length(learning_rates)
    counter = 0

    # --- Nested Loop: Topologies x Learning Rates ---
    for topo in topologies
        for lr in learning_rates
            counter += 1
            start_time = time()
            
            # Train ANN using the provided training function
            # Input data is passed as (Samples x Features) and (Samples x Classes)
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
            
            # --- Evaluation on Validation Set ---
            # Transpose input for Flux prediction: (Features x Samples)
            raw_preds_val = ann(permutedims(x_val)) 
            
            # Transpose output back for metrics: (Samples x Classes)
            y_pred_val_bool = classifyOutputs(permutedims(raw_preds_val))
            
            metrics_val = confusionMatrix(y_pred_val_bool, dataset.y_val_encoded)
            sens_val = metrics_val.sensitivity
            
            @printf("      [%d/%d] Topo: %-10s | LR: %.3f | Val Sens: %.4f | Time: %.2fs\n", 
                    counter, total_iterations, string(topo), lr, sens_val, elapsed)
            
            push!(results, (topo, lr, sens_val, elapsed, ann, train_hist, val_hist))
        end
    end
    
    # Sort results by Validation Sensitivity (Index 3)
    sort!(results, by = x -> x[3], rev = true)
    
    winner = results[1]
    (best_topo, best_lr, best_sens_val, best_time, best_model, best_train_hist, best_val_hist) = winner
    
    println("\n      WINNER FOUND: $best_topo (Sens: $best_sens_val)")

    # --- Final Audit on Test Set ---
    println("      Running Final Audit on TEST SET...")
    
    # Transpose Test data for Flux prediction
    raw_preds_test = best_model(permutedims(x_test))
    y_pred_test_bool = classifyOutputs(permutedims(raw_preds_test))
    
    # Calculate Final Metrics (Unit 4)
    metrics_test = confusionMatrix(y_pred_test_bool, dataset.y_test_encoded)
    
    acc_test  = metrics_test.accuracy * 100
    f1_test   = metrics_test.f_score
    sens_test = metrics_test.sensitivity
    spec_test = metrics_test.specificity
    prec_test = metrics_test.ppv
    
    println("      FINAL TEST RESULTS: Sens $(round(sens_test, digits=4)) | Acc $(round(acc_test, digits=2))%")

    return (best_topo, acc_test, f1_test, sens_test, spec_test, prec_test, best_time, best_model, metrics_test, best_train_hist, best_val_hist, best_lr)
end

"""
    save_approach_results_v2(winner_tuple, dataset, approach_name)

Saves the best trained ANN model and the corresponding test predictions to .jld2 files.
Includes necessary matrix transpositions for Flux compatibility.
"""
function save_approach_results_v2(winner_tuple, dataset, approach_name)
    println("\n Saving artifacts for $approach_name...")
    
    (topo, acc, f1, sens, spec, prec, time, model) = winner_tuple
    clean_name = replace(approach_name, " " => "")
    classes_ordered = [0, 1, 2, 3, 4]

    # Save Model File
    file_model = "Modelo_ANN_$(clean_name)_Final.jld2"
    JLD2.save(file_model, "ann_model", model)
    println("   Model Saved: $file_model")
    
    # Generate Predictions for Export
    x_test_flux = permutedims(dataset.x_test)
    raw_preds = model(x_test_flux)
    y_pred_vec = Flux.onecold(raw_preds, classes_ordered)
    
    y_test_T = permutedims(dataset.y_test_encoded)
    y_test_vec = Flux.onecold(y_test_T, classes_ordered)
    
    # Save Predictions Dictionary
    save_dict_preds = Dict(
        "modelo_info" => "ANN Topo:$topo ($clean_name)",
        "y_pred"      => y_pred_vec,
        "y_test"      => y_test_vec,
        "accuracy"    => acc / 100.0
    )
    file_preds = "Predicciones_ANN_$(clean_name).jld2"
    JLD2.save(file_preds, "ANN_preds", save_dict_preds)
    println("   Preds Saved: $file_preds")
end

"""
    export_standardized_results_v2(winner_tuple, dataset, approach_name)

Exports results in a standardized format for Ensemble integration.
Converts Multiclass ANN outputs (0-4) to Binary (0=Healthy, 1=Sick) and recalculates binary metrics.
"""
function export_standardized_results_v2(winner_tuple, dataset, approach_name)
    
    (topo, _, _, _, _, _, time_sec, model) = winner_tuple
    
    clean_name = replace(approach_name, " " => "") 
    model_id = "ANN_$(clean_name)_Topo$topo"
    
    println("   Standardizing results for Ensemble ($model_id)...")

    # Transpose input for Flux prediction
    x_test_flux = permutedims(dataset.x_test)
    raw_probs = model(x_test_flux)
    
    # Calculate probability of being Sick (1 - Prob(Healthy))
    # Class 0 corresponds to the first row (index 1)
    probs_healthy = raw_probs[1, :] 
    y_scores = 1.0f0 .- probs_healthy
    
    # Binarize Predictions (Threshold 0.5)
    y_pred_binary = Int.(y_scores .>= 0.5)
    
    # Binarize Ground Truth
    y_test_flux_target = permutedims(dataset.y_test_encoded)
    true_indices = Flux.onecold(y_test_flux_target) # Returns 1..5
    y_test_binary = [idx == 1 ? 0 : 1 for idx in true_indices]

    # Convert to Boolean for Metrics Calculation
    y_p_bool = Bool.(y_pred_binary)
    y_t_bool = Bool.(y_test_binary)
    
    # Calculate Binary Metrics using unit4
    (bin_acc, bin_err, bin_sens, bin_spec, bin_ppv, bin_npv, bin_f1, bin_cm) = confusionMatrix(y_p_bool, y_t_bool)
    
    calc_auc = 0.0 

    # Build Results Dictionary
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
    
    println("      EXPORT SUCCESS: $filename")
    println("      Binary Metrics -> Acc: $(round(bin_acc, digits=3)) | F1: $(round(bin_f1, digits=3))")
end