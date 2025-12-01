function stratified_holdOut(targets::AbstractVector, p_val::Real, p_test::Real; seed::Int=1234) 
    Random.seed!(seed)
    N = length(targets)
    classes = unique(targets)
    
    # Final Indices
    idx_train = Int[]
    idx_val   = Int[]
    idx_test  = Int[]
    
    # Loop iterating over all classes
    for c in classes
        # Get the index for a given class
        idx_class = findall(x -> x == c, targets)
        n_class = length(idx_class)
        
        # Shuffle inside the class
        shuffle!(idx_class)
        
        # Measure how many are going into each set
        n_val  = round(Int, n_class * p_val)
        n_test = round(Int, n_class * p_test)
        n_train = n_class - n_val - n_test
        
        # Distibute the data
        
        append!(idx_val,   idx_class[1:n_val])
        append!(idx_test,  idx_class[n_val+1 : n_val+n_test])
        append!(idx_train, idx_class[n_val+n_test+1 : end])
    end
    
    # Shuffle final indices, avoiding order by class
    shuffle!(idx_train)
    shuffle!(idx_val)
    shuffle!(idx_test)
    
    return idx_train, idx_val, idx_test
end


function load_and_clean_data(path::String)
    println(">>> Loading data from: $path")
    
    # 1. Load data
    data = DataFrame()
    try
        data = CSV.read(path, DataFrame)
    catch e
        error("Error while loading file. Check path.\nDetails: $e")
    end

    # 2. Drop irrelevant features
    select!(data, Not([:id, :dataset])) 

    println("  Original Size: $(size(data))")

    # ---------------------------------------------------------
    # 3. Mange Nulls (CAT) -> "missingval"
    # ---------------------------------------------------------
    cat_col_null = [:fbs, :restecg, :exang, :slope, :thal, :ca]
    
    for col in cat_col_null
        
        data[!, col] = replace(data[!, col], missing => "missingval")
    end
    println(" Categorical Null values replaced with ---> 'missingval'.")

    # ---------------------------------------------------------
    # 4. manage Nulls (NUM) -> Drop rows if null < given percentage
    # ---------------------------------------------------------
    
    rows = nrow(data)
    desc_df = describe(data, :nmissing)
    aux_miss = DataFrame(
        Column = desc_df.variable, 
        Percent = (desc_df.nmissing ./ rows) .* 100
    )

    #  Drop rows if null < given percentage
    cols_to_clean_df = @rsubset(aux_miss, 0 < :Percent < 7.5)
    cols_to_clean = Symbol.(cols_to_clean_df.Column)

    if !isempty(cols_to_clean)
        dropmissing!(data, cols_to_clean)
        println("  Deleted rows in features: $cols_to_clean")
    end

    # ---------------------------------------------------------
    # 5. Final sort
    # ---------------------------------------------------------
    num_col = [:age, :trestbps, :chol, :thalch, :oldpeak]
    cat_col = [:sex, :cp, :fbs, :restecg, :exang, :slope, :ca, :thal]
    target_col = :num
    # disallowmissing, transform clean features to single types
    data = select(data, num_col, cat_col, target_col)
    disallowmissing!(data) 

    println("  Final shape: $(size(data))")
    println("------------------------")

    return data, num_col, cat_col, target_col
end


function prepare_data(clean_data::DataFrame, 
                                    num_col::Vector{Symbol}, #name of the numerical features
                                    cat_col::Vector{Symbol}, #name of the categorical features
                                    target_col::Symbol; #name of the target feature
                                    Pval::Real=0.15, #percent for split  val set
                                    Ptest::Real=0.15, #percent for split test set
                                    norm_method::Symbol=:minmax) #normalization method, either :minmax or :zscore
    
    println("\n--- init Preprocess ---")
    println("   Normalization: $norm_method")

    # --- 1. Data Split (HoldOut) ---
    rows, columns = size(clean_data)
    N = rows
    
    (train_indices, val_indices, test_indices) = stratified_holdOut(data[!, target_col], Pval, Ptest; seed = 1234)
    
    train_data = clean_data[train_indices, :]
    val_data = clean_data[val_indices, :]
    test_data = clean_data[test_indices, :]
    println("    Stratigfied HoldOut split: $(size(train_data,1)) train, $(size(val_data,1)) val, $(size(test_data,1)) test")

    # --- 2. Features/Target Split ---
    x_train_df = select(train_data, Not(target_col))
    y_train_vec = train_data[!, target_col]
    x_val_df = select(val_data, Not(target_col))
    y_val_vec = val_data[!, target_col]
    x_test_df = select(test_data, Not(target_col))
    y_test_vec = test_data[!, target_col]

    # --- 3. Normalization of numerical features ---
    println("    Normalizing numerical features...")
    x_train_num_mat = Matrix{Float64}(x_train_df[!, num_col])
    x_test_num_mat = Matrix{Float64}(x_test_df[!, num_col])
    x_val_num_mat = Matrix{Float64}(x_val_df[!, num_col])
    
    norm_param = nothing #Init the variable 

    if norm_method == :minmax
        norm_param = calculateMinMaxNormalizationParameters(x_train_num_mat)
        normalizeMinMax!(x_train_num_mat, norm_param)
        normalizeMinMax!(x_test_num_mat, norm_param)
        normalizeMinMax!(x_val_num_mat, norm_param)
    elseif norm_method == :zscore
        norm_param = calculateZeroMeanNormalizationParameters(x_train_num_mat)
        normalizeZeroMean!(x_train_num_mat, norm_param)
        normalizeZeroMean!(x_test_num_mat, norm_param)
        normalizeZeroMean!(x_val_num_mat, norm_param)
    else
        error("Normalization method not clear: '$norm_method' . Use :minmax or :zscore.")
    end
    println("    ...Normalization completed.")

    # --- 4. One-Hot Encoding Categorial features ---
    println("    Encoding categorical features (OHE)...")
    
    x_train_cat_mat = BitArray{2}(undef, size(x_train_df, 1), 0)
    x_test_cat_mat  = BitArray{2}(undef, size(x_test_df, 1), 0)
    x_val_cat_mat = BitArray{2}(undef, size(x_val_df, 1), 0)
    
    ohe_classes_map = Dict{Symbol, Vector{Any}}() # Store classes

    for col in cat_col
        feature_train = x_train_df[!, col]
        feature_test  = x_test_df[!, col]
        feature_val = x_val_df[!, col]
        
        learn_classes = unique(feature_train)
        
        # Manage unseen clases, ex. if missingval is only present in tets and validation due to the randomness in split
        for val in unique(vcat(feature_test, feature_val))
            if !(val in learn_classes)
                push!(learn_classes, val)
                println("        -> Warning: Feature '$col': Class '$val' aded (Not present in train).")
            end
        end
        
        ohe_classes_map[col] = learn_classes # Save the classes
        
        encoded_train = oneHotEncoding(feature_train, learn_classes)
        encoded_test  = oneHotEncoding(feature_test, learn_classes)
        encoded_val   = oneHotEncoding(feature_val, learn_classes)
        
        x_train_cat_mat = hcat(x_train_cat_mat, encoded_train)
        x_test_cat_mat  = hcat(x_test_cat_mat, encoded_test)
        x_val_cat_mat   = hcat(x_val_cat_mat, encoded_val) 
    end
    println("    ...OHE completed.")

    # --- 5. Combine the matrices ---
    println("    Concatenate numerical and categorical matrices...")
    x_train_final = hcat(x_train_num_mat, x_train_cat_mat)
    x_test_final  = hcat(x_test_num_mat, x_test_cat_mat)
    x_val_final = hcat(x_val_num_mat, x_val_cat_mat)
    
    # --- 6. Process the targets ---
    target_classes = sort(unique(clean_data[!, target_col]))
    println("    Classes stored for the target: $target_classes")
    
    # (SVM, DT, kNN)
    y_train_cat = MLJ.categorical(y_train_vec)
    y_test_cat = MLJ.categorical(y_test_vec)
    y_val_cat = MLJ.categorical(y_val_vec)
    
    # For ANN (OHE)
    y_train_ohe = oneHotEncoding(y_train_vec, target_classes)
    y_test_ohe  = oneHotEncoding(y_test_vec, target_classes)
    y_val_ohe   = oneHotEncoding(y_val_vec, target_classes)
    
    println("--- PREPROCESS END SUCCESFULLY ---")

    # --- 7. Return data ---
    return (
        x_train = x_train_final,
        y_train_cat = y_train_cat, # For MLJ
        y_train_ohe = y_train_ohe, # For ANN
        
        x_val = x_val_final,
        y_val_cat = y_val_cat,     # For MLJ
        y_val_ohe = y_val_ohe,     # For ANN
        
        x_test = x_test_final,
        y_test_cat = y_test_cat,   # For MLJ
        y_test_ohe = y_test_ohe,   # For ANN
        
        norm_params = norm_param,
        ohe_classes = ohe_classes_map
    )
end

# Función auxiliar para entrenar y evaluar un SVM
function run_svm_experiment(x_train, y_train, x_test, y_test, approach_name, configs)
    println("\n=================================================================")
    println(" EXPERIMENT: $approach_name (Manual Train/Test)")
    println("=================================================================")
    
    # -----------------------------------------------------------
    # 1. LIMPIEZA DE ETIQUETAS (Categorical -> Int)
    # -----------------------------------------------------------
    get_val(x) = (typeof(x) <: CategoricalValue) ? unwrap(x) : x
    y_train_clean = vec(Int.(get_val.(y_train)))
    y_test_clean  = vec(Int.(get_val.(y_test)))
    
    # -----------------------------------------------------------
    # 2. CORRECCIÓN DE DIMENSIONES (SOLUCIÓN AL ERROR)
    # -----------------------------------------------------------
    # LIBSVM necesita (Features x Samples).
    # Si detectamos que las muestras (filas) coinciden con el largo de Y, transponemos.
    x_train_mat = Matrix(x_train) # Aseguramos que sea matriz
    x_test_mat  = Matrix(x_test)

    if size(x_train_mat, 2) != length(y_train_clean) && size(x_train_mat, 1) == length(y_train_clean)
        println("   [Auto-Fix] Transponiendo matriz para formato LIBSVM (Features x Samples)...")
        x_train_clean = Float64.(permutedims(x_train_mat))
        x_test_clean  = Float64.(permutedims(x_test_mat))
    else
        x_train_clean = Float64.(x_train_mat)
        x_test_clean  = Float64.(x_test_mat)
    end
    
    println("   Dimensiones finales X_train: $(size(x_train_clean))")

    # -----------------------------------------------------------
    # 3. CÁLCULO DE PESOS
    # -----------------------------------------------------------
    classes = sort(unique(y_train_clean))
    n_samples = length(y_train_clean)
    n_classes = length(classes)
    class_weights = Dict{Int64, Float64}()
    
    for c in classes
        n_class = count(==(c), y_train_clean)
        raw_weight = n_samples / (n_classes * n_class)
        if occursin("ICA", approach_name); raw_weight = sqrt(raw_weight); end
        class_weights[Int64(c)] = raw_weight
    end
    
    results = []

    # -----------------------------------------------------------
    # 4. BUCLE DE ENTRENAMIENTO
    # -----------------------------------------------------------
    for (i, conf) in enumerate(configs)
        # Desempaquetado seguro
        if length(conf) == 6
            (name, k_type, c_val, g_val, deg_val, coef_val) = conf
        elseif length(conf) == 5
            (name, k_type, c_val, g_val, deg_val) = conf
            coef_val = 0.0
        end
        
        print(" Testing Config $i/$(length(configs)): $name ... ")
        start_time = time()
        
        # Entrenar (usando x_train_clean transpuesto)
        model = svmtrain(x_train_clean, y_train_clean, 
                         kernel = k_type, 
                         cost = c_val, 
                         gamma = g_val, 
                         degree = deg_val,
                         coef0 = coef_val,
                         weights = class_weights,
                         shrinking = true) 
        
        train_time = time() - start_time
        
        # Predecir
        y_hat, _ = svmpredict(model, x_test_clean)
        y_hat_int = Int.(y_hat)
        
        # Evaluar
        metrics = confusionMatrix(y_hat_int, y_test_clean, classes; weighted=false)
        
        # Extraer métricas de forma segura
        sens = hasproperty(metrics, :sensitivity) ? metrics.sensitivity : metrics[3]
        acc  = hasproperty(metrics, :accuracy)    ? metrics.accuracy    : metrics[1]
        f1   = hasproperty(metrics, :f_score)     ? metrics.f_score     : metrics[7]
        
        acc_percent = acc <= 1.0 ? acc * 100 : acc

        push!(results, (
            config = conf,
            sens = sens,
            acc = acc_percent,
            f1 = f1,
            time = train_time,
            model = model
        ))
        
        println("Done. Sens: $(round(sens, digits=4)) | Acc: $(round(acc_percent, digits=2))%")
    end
    
    # Ordenar
    sort!(results, by = x -> x.sens, rev = true)
    winner = results[1]

    return winner
end

function fit_preprocess(x_num::Matrix, x_cat::DataFrame)
    # Num
    norm_params = calculateMinMaxNormalizationParameters(x_num)
    
    # Cat
    ohe_classes = Dict{Symbol, Vector{Any}}()
    ohe_modes = Dict{Symbol, Any}() 
    
    for col in names(x_cat)
        col_sym = Symbol(col)
        vals = x_cat[!, col]
        ohe_classes[col_sym] = unique(vals)
        
        counts = Dict{Any, Int}()
        for v in vals
            counts[v] = get(counts, v, 0) + 1
        end
        ohe_modes[col_sym] = argmax(counts) 
    end
    
    return (norm_params, ohe_classes, ohe_modes)
end

function apply_preprocess(x_num::Matrix, x_cat::DataFrame, params)
    (norm_params, ohe_classes, ohe_modes) = params
    
    # Num
    x_num_proc = normalizeMinMax(copy(x_num), norm_params)
    
    # Cat
    list_ohe = []
    for col_str in names(x_cat)
        col = Symbol(col_str)
        data_col = copy(x_cat[!, col])
        classes = ohe_classes[col]
        mode_val = ohe_modes[col]
        
        #This ensures  OHE always deliver the same column size
        for i in 1:length(data_col)
            if !(data_col[i] in classes)
                data_col[i] = mode_val
            end
        end
        
        push!(list_ohe, oneHotEncoding(data_col, classes))
    end
    
    x_cat_proc = hcat(list_ohe...)
    return hcat(x_num_proc, x_cat_proc)
end

function universalCrossValidation1(
        modelType::Symbol,
        hyperparams::Dict,
        data_num::DataFrame, 
        data_cat::DataFrame, 
        targets::AbstractArray, 
        folds_indices::Array{Int64,1})
    
    num_folds = maximum(folds_indices)
    classes = unique(targets)
    accuracies = zeros(num_folds)
    
    for k in 1:num_folds
        
        idx_train_global = folds_indices .!= k
        idx_test         = folds_indices .== k

        
        x_tr_raw_num = Matrix{Float64}(data_num[idx_train_global, :])
        x_tr_raw_cat = data_cat[idx_train_global, :]
        y_tr_global  = targets[idx_train_global]
        
        x_te_raw_num = Matrix{Float64}(data_num[idx_test, :])
        x_te_raw_cat = data_cat[idx_test, :]
        y_test       = targets[idx_test]
        
     
        X_train_final, X_test_final = nothing, nothing
        y_train_final, y_test_final = nothing, nothing
        
        # =========================================================
        # (ANN) -> Nested Split + Float32
        # =========================================================
        if modelType == :ANN
            # 1. Split for Early Stopping
            val_ratio = get(hyperparams, :validationRatio, 0.2)
            (idx_tr_real, idx_val) = holdOut(length(y_tr_global), val_ratio)
          
            x_tr_real_num = x_tr_raw_num[idx_tr_real, :]
            x_tr_real_cat = x_tr_raw_cat[idx_tr_real, :]
            y_tr_real     = y_tr_global[idx_tr_real]
            
            x_val_num     = x_tr_raw_num[idx_val, :]
            x_val_cat     = x_tr_raw_cat[idx_val, :]
            y_val         = y_tr_global[idx_val]
            
            # 2. Preprocessing 
            params_proc = fit_preprocess(x_tr_real_num, x_tr_real_cat)
            
            # 3.(Train Real, Val, Test Global) -> Float32
            X_tr_proc  = Float32.(apply_preprocess(x_tr_real_num, x_tr_real_cat, params_proc))
            X_val_proc = Float32.(apply_preprocess(x_val_num, x_val_cat, params_proc))
            X_te_proc  = Float32.(apply_preprocess(x_te_raw_num, x_te_raw_cat, params_proc))
            
            # 4. Targets OHE
            y_tr_ohe  = oneHotEncoding(y_tr_real, classes)
            y_val_ohe = oneHotEncoding(y_val, classes)
            y_te_ohe  = oneHotEncoding(y_test, classes)
            
            # 5. train ANN
            topology = get(hyperparams, :topology, [10])
            lr       = get(hyperparams, :learningRate, 0.01)
            epochs   = get(hyperparams, :maxEpochs, 1000)
            
            ann, _, _, _ = trainClassANN(
                topology, (X_tr_proc, y_tr_ohe);
                validationDataset = (X_val_proc, y_val_ohe),
                learningRate = lr, maxEpochs = epochs, maxEpochsVal = 20, showText = false
            )
            
            # 6. EVAL
            test_out = ann(X_te_proc')'
            accuracies[k] = accuracy(test_out, y_te_ohe)
            
        # =========================================================
        # MODELS -> Simple Split + Table
        # =========================================================
        else
            # 1. Preprocessing
            params_proc = fit_preprocess(x_tr_raw_num, x_tr_raw_cat)
            
            # 2. Transform
            X_tr_mat = apply_preprocess(x_tr_raw_num, x_tr_raw_cat, params_proc)
            X_te_mat = apply_preprocess(x_te_raw_num, x_te_raw_cat, params_proc)
            
            # 3. Prepare data for MLJ (Table + Categorical)
            X_train_table = MLJ.table(X_tr_mat)
            X_test_table  = MLJ.table(X_te_mat)
            y_train_cat   = MLJ.categorical(y_tr_global)
            y_test_cat    = MLJ.categorical(y_test)
            
            # 4. Config
            model = nothing
            if modelType == :SVC
                # Extraer params
                C_val = get(hyperparams, :C, 1.0)
                g_val = get(hyperparams, :gamma, 0.0) # 0.0 suele indicar auto/default en LIBSVM wrapper si no se especifica
                d_val = get(hyperparams, :degree, 3)
                c0_val = get(hyperparams, :coef0, 0.0)
                k_str = get(hyperparams, :kernel, "rbf")
    
                # Mapear string a Kernel de LIBSVM
                k_type = if k_str == "linear" LIBSVM.Kernel.Linear
                         elseif k_str == "poly" LIBSVM.Kernel.Polynomial
                         elseif k_str == "sigmoid" LIBSVM.Kernel.Sigmoid
                         else LIBSVM.Kernel.RadialBasis end
                
                model = SVMClassifier(kernel=k_type, cost=Float64(C_val), gamma=Float64(g_val), degree=Int32(d_val), coef0=Float64(c0_val))
                
            elseif modelType == :DT
                depth = get(hyperparams, :max_depth, 5)
                model = DTClassifier(max_depth=Int(depth), rng=Random.MersenneTwister(1234))
            elseif modelType == :KNN
                K = get(hyperparams, :K, 5)
                model = kNNClassifier(K=Int(K))
            end
            
            # 5. Train and predict
            mach = machine(model, X_train_table, y_train_cat)
            MLJ.fit!(mach, verbosity=0)
            y_pred = MLJ.predict(mach, X_test_table)
            
            # 6. Evaluate
            final_pred = (modelType == :SVC) ? y_pred : mode.(y_pred)
            accuracies[k] = MLJ.accuracy(final_pred, y_test_cat)
        end
    end
    
    return mean(accuracies), std(accuracies)
end

"""
    fit_preprocess_zscore(x_num, x_cat)

Learns the transformation parameters from the training set:
1. Calculates Mean and Std for Z-Score normalization (numerical features).
2. Identifies unique Classes and the Mode for One-Hot Encoding (categorical features).
   The Mode is used later to impute unknown categories in Validation/Test sets.
"""
function fit_preprocess_zscore(x_num::Matrix, x_cat::DataFrame)
    # 1. Numerical: Z-Score Parameters (Mean, Std)
    # Note: PCA requires data centered at 0 with unit variance.
    norm_params = calculateZeroMeanNormalizationParameters(x_num)
    
    # 2. Categorical: Store Classes and Mode
    ohe_classes = Dict{Symbol, Vector{Any}}()
    ohe_modes = Dict{Symbol, Any}()
    
    for col in names(x_cat)
        col_sym = Symbol(col)
        vals = x_cat[!, col]
        ohe_classes[col_sym] = unique(vals)
        
        # Calculate Mode manually (to avoid dependency on StatsBase)
        counts = Dict{Any, Int}()
        for v in vals 
            counts[v] = get(counts, v, 0) + 1 
        end
        # argmax on the dictionary returns the key with the highest value
        ohe_modes[col_sym] = argmax(counts) 
    end
    
    return (norm_params, ohe_classes, ohe_modes)
end

"""
    apply_preprocess_zscore(x_num, x_cat, params)

Applies the learned transformations to a dataset (Train, Val, or Test):
1. Normalizes numerical columns using Z-Score.
2. Applies One-Hot Encoding. If a value was not present in the training set,
   it is replaced by the Mode of the training set (Imputation) to preserve dimensions.
"""
function apply_preprocess_zscore(x_num::Matrix, x_cat::DataFrame, params)
    (norm_params, ohe_classes, ohe_modes) = params
    
    # 1. Numerical: Apply Z-Score
    # Use copy() to avoid modifying the original matrix in place
    x_num_proc = normalizeZeroMean(copy(x_num), norm_params)
    
    # 2. Categorical: Apply OHE with Mode Imputation
    list_ohe = []
    for col_str in names(x_cat)
        col = Symbol(col_str)
        data_col = copy(x_cat[!, col]) # Copy to avoid altering original dataframe
        
        classes = ohe_classes[col]
        mode_val = ohe_modes[col]
        
        # CRITICAL LOGIC: Handling Unknown Classes
        # If a value in Val/Test is not in the learned 'classes' from Train,
        # replace it with the Mode. This ensures OHE always yields the same number of columns.
        for i in 1:length(data_col)
            if !(data_col[i] in classes)
                data_col[i] = mode_val
            end
        end
        
        push!(list_ohe, oneHotEncoding(data_col, classes))
    end
    
    # Concatenate Numerical and One-Hot Encoded Categorical matrices
    return hcat(x_num_proc, hcat(list_ohe...))
end

"""
    universalCrossValidation_PCA(modelType, hyperparams, data_num, data_cat, targets, folds_indices)

Performs Cross-Validation incorporating PCA.
Ensures zero data leakage by fitting Z-Score, OHE, and PCA only on the training fold.
"""
function universalCrossValidation_PCA(
        modelType::Symbol,
        hyperparams::Dict,
        data_num::DataFrame, 
        data_cat::DataFrame, 
        targets::AbstractArray, 
        folds_indices::Array{Int64,1})
    
    num_folds = maximum(folds_indices)
    classes = unique(targets)
    accuracies = zeros(num_folds)
    
    # Extract PCA component number (default to 17 based on previous scree plot)
    n_pca = get(hyperparams, :pca_components, 17)

    for k in 1:num_folds
        # --- A. FOLD SEPARATION (Global Train vs. Fold Test) ---
        idx_tr_glob = folds_indices .!= k
        idx_test    = folds_indices .== k
        
        # Raw Data for this fold
        x_tr_raw_num = Matrix{Float64}(data_num[idx_tr_glob, :])
        x_tr_raw_cat = data_cat[idx_tr_glob, :]
        y_tr_glob    = targets[idx_tr_glob]
        
        x_te_raw_num = Matrix{Float64}(data_num[idx_test, :])
        x_te_raw_cat = data_cat[idx_test, :]
        y_test       = targets[idx_test]
        
        # =========================================================
        # BRANCH A: ARTIFICIAL NEURAL NETWORKS (ANN)
        # Requires: Nested Split (Early Stopping), Float32, OHE Targets
        # =========================================================
        if modelType == :ANN
            # 1. Internal Split for Early Stopping (Nested CV)
            val_ratio = get(hyperparams, :validationRatio, 0.2)
            # Use holdOut from unit3
            (idx_real, idx_val) = holdOut(length(y_tr_glob), val_ratio) 
            
            # Sub-sets (Real Train vs Internal Validation)
            x_tr_real_num = x_tr_raw_num[idx_real, :]
            x_tr_real_cat = x_tr_raw_cat[idx_real, :]
            y_tr_real     = y_tr_glob[idx_real]
            
            x_val_num     = x_tr_raw_num[idx_val, :]
            x_val_cat     = x_tr_raw_cat[idx_val, :]
            y_val         = y_tr_glob[idx_val]
            
            # 2. Learn Preprocessing (Z-Score/OHE) on Real Train
            params_proc = fit_preprocess_zscore(x_tr_real_num, x_tr_real_cat)
            
            # 3. Apply Preprocessing (Hybrid Matrices)
            X_tr_hyb  = apply_preprocess_zscore(x_tr_real_num, x_tr_real_cat, params_proc)
            X_val_hyb = apply_preprocess_zscore(x_val_num, x_val_cat, params_proc)
            X_te_hyb  = apply_preprocess_zscore(x_te_raw_num, x_te_raw_cat, params_proc)
            
            # 4. Learn PCA on Real Train
            # Transpose input because MultivariateStats expects (features x samples)
            pca_m = MultivariateStats.fit(MultivariateStats.PCA, X_tr_hyb'; maxoutdim=n_pca)
            
            # 5. Project PCA -> Convert to Float32 for Flux
            X_tr_pca  = Float32.(MultivariateStats.transform(pca_m, X_tr_hyb')')
            X_val_pca = Float32.(MultivariateStats.transform(pca_m, X_val_hyb')')
            X_te_pca  = Float32.(MultivariateStats.transform(pca_m, X_te_hyb')')
            
            # 6. Targets OHE
            y_tr_ohe  = oneHotEncoding(y_tr_real, classes)
            y_val_ohe = oneHotEncoding(y_val, classes)
            y_te_ohe  = oneHotEncoding(y_test, classes)
            
            # 7. Train ANN
            topo = get(hyperparams, :topology, [10])
            lr   = get(hyperparams, :learningRate, 0.01)
            eps  = get(hyperparams, :maxEpochs, 1000)
            
            ann, _, _, _ = trainClassANN(
                topo, (X_tr_pca, y_tr_ohe);
                validationDataset = (X_val_pca, y_val_ohe),
                learningRate = lr, maxEpochs = eps, maxEpochsVal = 20, showText = false
            )
            
            # 8. Evaluate
            out = ann(X_te_pca')'
            accuracies[k] = accuracy(out, y_te_ohe)
            
        # =========================================================
        # BRANCH B: CLASSICAL MODELS (SVM, DT, KNN)
        # Requires: Simple Split, Table format, Categorical Targets
        # =========================================================
        else
            # 1. Learn Preprocessing on Global Train
            params_proc = fit_preprocess_zscore(x_tr_raw_num, x_tr_raw_cat)
            
            # 2. Apply Preprocessing
            X_tr_hyb = apply_preprocess_zscore(x_tr_raw_num, x_tr_raw_cat, params_proc)
            X_te_hyb = apply_preprocess_zscore(x_te_raw_num, x_te_raw_cat, params_proc)
            
            # 3. Learn PCA on Global Train
            pca_m = MultivariateStats.fit(MultivariateStats.PCA, X_tr_hyb'; maxoutdim=n_pca)
            
            # 4. Project PCA -> Convert to Table for MLJ
            X_tr_pca = MLJ.table(MultivariateStats.transform(pca_m, X_tr_hyb')')
            X_te_pca = MLJ.table(MultivariateStats.transform(pca_m, X_te_hyb')')
            
            y_tr_cat = MLJ.categorical(y_tr_glob)
            y_te_cat = MLJ.categorical(y_test)
            
            # 5. Configure Model
            model = nothing
            if modelType == :SVC
                # Extraer parámetros con valores por defecto seguros
                C_val = get(hyperparams, :C, 1.0)
                g_val = get(hyperparams, :gamma, 1.0/n_pca) # 1/features es default usual
                d_val = get(hyperparams, :degree, 3)
                c0_val = get(hyperparams, :coef0, 0.0)
                k_str = get(hyperparams, :kernel, "rbf")
                
                # Mapear String -> LIBSVM Kernel
                k_type = if k_str=="linear" LIBSVM.Kernel.Linear
                         elseif k_str=="poly" LIBSVM.Kernel.Polynomial
                         elseif k_str=="sigmoid" LIBSVM.Kernel.Sigmoid
                         else LIBSVM.Kernel.RadialBasis end
                
                model = SVMClassifier(kernel=k_type, cost=Float64(C_val), gamma=Float64(g_val), degree=Int32(d_val), coef0=Float64(c0_val))
                
            elseif modelType == :DT
                d = get(hyperparams, :max_depth, 5)
                model = DTClassifier(max_depth=Int(d), rng=Random.MersenneTwister(1234))
            elseif modelType == :KNN
                K = get(hyperparams, :K, 5)
                model = kNNClassifier(K=Int(K))
            end
            
            # 6. Train
            mach = machine(model, X_tr_pca, y_tr_cat)
            MLJ.fit!(mach, verbosity=0)
            
            # 7. Evaluate
            y_pred = MLJ.predict(mach, X_te_pca)
            final_pred = (modelType == :SVC) ? y_pred : mode.(y_pred)
            accuracies[k] = MLJ.accuracy(final_pred, y_te_cat)
        end
    end
    
    return mean(accuracies), std(accuracies)
end