
#load the previous functions
include("unit2-multilayer-perceptron.jl");
include("unit3-overfitting.jl");
include("unit4-metrics.jl");
include("unit5-crossvalidation.jl");
#load the models
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0;
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0;
DTClassifier = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0;


function modelCrossValidation(
        modelType::Symbol, modelHyperparameters::Dict,
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        crossValidationIndices::Array{Int64,1})
    
    inputs, targets = dataset

    if modelType == :ANN || modelType == :DoME
        
        topology = get(modelHyperparameters, :topology) do
            haskey(modelHyperparameters, "topology") ? 
                modelHyperparameters["topology"] : 
                error("Hyperparameter :topology not found for :$modelType")
        end
        
        #mapping keywords
        numExecutions = get(modelHyperparameters, :repetitions, 
            get(modelHyperparameters, "repetitions", 1))

        transferFunctions = get(modelHyperparameters, :transferFunctions, 
            get(modelHyperparameters, "transferFunctions", fill(σ, length(topology) + 1)))

        maxEpochs = get(modelHyperparameters, :maxEpochs, 
            get(modelHyperparameters, "maxEpochs", 1000))

        minLoss = get(modelHyperparameters, :minLoss, 
            get(modelHyperparameters, "minLoss", 0.0))

        learningRate = get(modelHyperparameters, :learningRate, 
            get(modelHyperparameters, "learningRate", 0.01))

        validationRatio = get(modelHyperparameters, :validationRatio, 
            get(modelHyperparameters, "validationRatio", 0.0))

        maxEpochsVal = get(modelHyperparameters, :patience, 
            get(modelHyperparameters, "patience", 20)) # patience = maxEpochsVal
        
        #call the previous developed function
        return ANNCrossValidation(
            topology,              
            dataset,                
            crossValidationIndices; 
           
            numExecutions = numExecutions,
            transferFunctions = transferFunctions,
            maxEpochs = maxEpochs,
            minLoss = minLoss,
            learningRate = learningRate,
            validationRatio = validationRatio,
            maxEpochsVal = maxEpochsVal
        )
    end
    

    # -----------------------------------------------------------------
    # MLJ MODELS (SVM, DT, kNN)
    # -----------------------------------------------------------------
    
    targets_str = string.(targets) 
    classes = unique(targets_str)
    numClasses = length(classes)
    numFolds = maximum(crossValidationIndices)

    accuracies = zeros(numFolds)
    error_rates = zeros(numFolds)
    recalls_macro = zeros(numFolds)
    specificities_macro = zeros(numFolds)
    ppvs_macro = zeros(numFolds)
    npvs_macro = zeros(numFolds)
    f1s_macro = zeros(numFolds)
    
    globalConfusionMatrix = zeros(Int64, numClasses, numClasses)

    # Crossvalidation loop
    for k in 1:numFolds
        
        trainIndices = crossValidationIndices .!= k
        testIndices = crossValidationIndices .== k
        
        Xtrain = inputs[trainIndices, :]
        ytrain = targets_str[trainIndices]
        
        Xtest = inputs[testIndices, :]
        ytest = targets_str[testIndices]
        
        
        model = nothing 
        
        if modelType == :SVC
            C = get(modelHyperparameters, :C, get(modelHyperparameters, "C", 1.0))
            kernel_str = get(modelHyperparameters, :kernel, get(modelHyperparameters, "kernel", "rbf"))
            degree = get(modelHyperparameters, :degree, get(modelHyperparameters, "degree", 3))
            gamma_str = get(modelHyperparameters, :gamma, get(modelHyperparameters, "gamma", "auto"))
            coef0 = get(modelHyperparameters, :coef0, get(modelHyperparameters, "coef0", 0.0))

            kernel = LIBSVM.Kernel.RadialBasis #Default
            if kernel_str == "linear"
                kernel = LIBSVM.Kernel.Linear
            elseif kernel_str == "poly"
                kernel = LIBSVM.Kernel.Polynomial
            elseif kernel_str == "sigmoid"
                kernel = LIBSVM.Kernel.Sigmoid
            end
            
            num_features = size(Xtrain, 2)
            gamma = (gamma_str == "auto") ? (1.0 / num_features) : parse(Float64, string(gamma_str))

            model = SVMClassifier(
                kernel = kernel,
                cost = Float64(C),
                gamma = Float64(gamma),
                degree = Int32(degree), 
                coef0 = Float64(coef0)  
            )
            
        elseif modelType == :DecisionTreeClassifier
            max_depth = get(modelHyperparameters, :max_depth, get(modelHyperparameters, "max_depth", -1))
            
            model = DTClassifier(
                max_depth = Int(max_depth),
                rng = Random.MersenneTwister(1) 
            )
            
        elseif modelType == :KNeighborsClassifier
            n_neighbors = get(modelHyperparameters, :n_neighbors, get(modelHyperparameters, "n_neighbors", 5))
            
            model = kNNClassifier(
                K = Int(n_neighbors) 
            )
            
        else
            error("modelType desconocido: $modelType.")
        end
        
        #Create the machine and train 
        mach = machine(model, MLJ.table(Xtrain), categorical(ytrain))
        MLJ.fit!(mach, verbosity=0)

        # Predict
        y_raw = MLJ.predict(mach, MLJ.table(Xtest))
        
        y = nothing
        if modelType == :SVC
            #this model returns the predicted class 
            y = y_raw
        else
            # the other models return the probabilities of correspondicng to each class, so we select the mode (highest prob. value)
            y = mode.(y_raw)
        end
        
        y = convert(Vector{String}, y)
        
        # Group metrics
        cm = confusionMatrix(y, ytest, classes) 
        
        globalConfusionMatrix .+= cm.confusion_matrix
        
        accuracies[k] = cm.accuracy
        error_rates[k] = cm.error_rate
        
        recalls_macro[k] = mean(cm.per_class_metrics.sensitivities)
        specificities_macro[k] = mean(cm.per_class_metrics.specificities)
        ppvs_macro[k] = mean(cm.per_class_metrics.ppvs)
        npvs_macro[k] = mean(cm.per_class_metrics.npvs) 
        f1s_macro[k] = mean(cm.per_class_metrics.f_scores)

    end # K-fold loop end
    
    # 6. Calcular media y desviación estándar de las métricas
    return (
        (mean(accuracies), std(accuracies)),
        (mean(error_rates), std(error_rates)),
        (mean(recalls_macro), std(recalls_macro)),
        (mean(specificities_macro), std(specificities_macro)),
        (mean(ppvs_macro), std(ppvs_macro)),
        (mean(npvs_macro), std(npvs_macro)),
        (mean(f1s_macro), std(f1s_macro)),
        globalConfusionMatrix
    )
end