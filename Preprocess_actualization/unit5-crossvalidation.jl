
# Make results reproducible by setting a fixed random seed
Random.seed!(1234);

# ---------------------------------------------------------
# Unit 5 Cross-validation
# ---------------------------------------------------------

function crossvalidation(N::Int64, k::Int64)

    # New vector with k sorted elements
    folds = 1:k

    # Get the number of repetitions
    num_repeat = Int(ceil(N/k))
    
    # Take the first N values of the vector
    indices = repeat(folds, num_repeat)[1:N]

    # Shuffled vector of size N 
    shuffle!(indices)

    return indices
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    """
    Function to handle the preservation of the distribution of positive and negative 
    instances across folds.
    """

    N = size(targets, 1)
    # Note: zeros(N) is slightly slower than Array{} due to initialization
    indices = Array{Int64,1}(undef, N)
    
    # ---------------------------------------------------------
    # Positive instances
    # ---------------------------------------------------------
    num_positive_inst = sum(targets)
    # Generate fold assignments for the positive instances to ensure stratified sampling
    cross_vector = crossvalidation(num_positive_inst, k)
    # Use the boolean mask targets to select the elements of index_vector to change
    indices[targets] = cross_vector

    # ---------------------------------------------------------
    # Negative instances
    # ---------------------------------------------------------
    num_negative_inst = sum(.!targets)
    cross_vector = crossvalidation(num_negative_inst, k)
    indices[.!targets] = cross_vector

    return indices
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)

    N = size(targets, 1) # num_instances
    num_classes = size(targets, 2)

    # Create an empty indices vector to store fold assignments
    indices = Array{Int64,1}(undef, N)
    for col in 1:num_classes
        num_instances_class = sum(targets[:, col])
        # Modify the indices of class 'col' where targets is true
        indices[targets[:,col]] = crossvalidation(num_instances_class, k)
    end

    return indices
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)

    return crossvalidation(oneHotEncoding(targets), k)
end;


function ANNCrossValidation(topology::AbstractArray{<:Int,1},
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        crossValidationIndices::Array{Int64,1};
        numExecutions::Int=50,
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
        validationRatio::Real=0, maxEpochsVal::Int=20)

    inputs, targets = dataset
    classes = unique(targets)
    numClasses = size(classes, 1)
    targets = oneHotEncoding(targets, classes)

    numFolds = maximum(crossValidationIndices)

    testAccuracies = Array{Float64,1}(undef, numFolds)
    testErrorRate = Array{Float64,1}(undef, numFolds)
    testSensitivity = Array{Float64,1}(undef, numFolds)
    testSpecificity = Array{Float64,1}(undef, numFolds)
    testPPV = Array{Float64,1}(undef, numFolds)
    testNPV = Array{Float64,1}(undef, numFolds)
    testF1 = Array{Float64,1}(undef, numFolds)
    
    testConfusionMatrix = zeros(numClasses, numClasses)

    for i in 1:numFolds
        boolean_mask_test = (crossValidationIndices .== i)
        boolean_mask_train = .!(crossValidationIndices .== i)

        trainingInputs = inputs[boolean_mask_train, :]
        testInputs = inputs[boolean_mask_test, :]
        trainingTargets = targets[boolean_mask_train, :]
        testTargets = targets[boolean_mask_test, :]

        testAccuraciesEachRepetition = Array{Float64,1}(undef, numExecutions)
        testErrorRateEachRepetition = Array{Float64,1}(undef, numExecutions)
        testSensitivityEachRepetition = Array{Float64,1}(undef, numExecutions)
        testSpecificityEachRepetition = Array{Float64,1}(undef, numExecutions)
        testPPVEachRepetition = Array{Float64,1}(undef, numExecutions)
        testNPVEachRepetition = Array{Float64,1}(undef, numExecutions)
        testF1EachRepetition = Array{Float64,1}(undef, numExecutions)
        testConfusionMatrixEachRepetition = zeros((numClasses, numClasses, numExecutions)) 

        if validationRatio > 0
            # Adjust validation ratio
            numClassesTrainingVal = size(trainingInputs,1)
            #validationRatioAdj = validationRatio * numClasses/numClassesTrainingVal
            # The given validationRatio is to apply over the training subset, not the whole dataset
            (trainingIndices, valIndices) = holdOut(numClassesTrainingVal, validationRatio)

            trainingInputs = trainingInputs[trainingIndices, :]
            trainingTargets = trainingTargets[trainingIndices, :]
            valInputs = trainingInputs[valIndices, :]
            valTargets = trainingTargets[valIndices, :]
        else
            # If validationRatio is 0
            valInputs = Array{eltype(inputs),2}(undef,0,0)
            valTargets = falses(0, 0)
        end
        
        for ex in 1:numExecutions
            ann, t, v, s = trainClassANN(
                topology, 
                (trainingInputs, trainingTargets); 
                validationDataset=(valInputs, valTargets),
                testDataset=(testInputs, testTargets),
                transferFunctions=transferFunctions,
                maxEpochs=maxEpochs,
                minLoss=minLoss,
                learningRate=learningRate,
                maxEpochsVal=maxEpochsVal)
            
            testOutputs = ann(testInputs')'
            metrics = confusionMatrix(testOutputs, testTargets)
            testAccuraciesEachRepetition[ex] = metrics.accuracy
            testErrorRateEachRepetition[ex] = metrics.error_rate
            testSensitivityEachRepetition[ex] = metrics.sensitivity
            testSpecificityEachRepetition[ex] = metrics.specificity
            testPPVEachRepetition[ex] = metrics.ppv
            testNPVEachRepetition[ex] = metrics.npv
            testF1EachRepetition[ex] = metrics.f_score
        
            testConfusionMatrixEachRepetition[:, :, ex] = metrics.confusion_matrix
        end
        testAccuracies[i] = mean(testAccuraciesEachRepetition)
        testErrorRate[i] = mean(testErrorRateEachRepetition)
        testSensitivity[i] = mean(testSensitivityEachRepetition)
        testSpecificity[i] = mean(testSpecificityEachRepetition)
        testPPV[i] = mean(testPPVEachRepetition)
        testNPV[i] = mean(testNPVEachRepetition)
        testF1[i] = mean(testF1EachRepetition)
        meanConfusionMatrix3D = mean(testConfusionMatrixEachRepetition, dims=3)
        meanConfusionMatrix2D = dropdims(meanConfusionMatrix3D, dims=3)
        testConfusionMatrix .+= meanConfusionMatrix2D
    end
    results = (
        (mean(testAccuracies), std(testAccuracies)),     # Accuracy (mean, std)
        (mean(testErrorRate), std(testErrorRate)),       # Error Rate (mean, std)
        (mean(testSensitivity), std(testSensitivity)),   # Sensitivity (mean, std)
        (mean(testSpecificity), std(testSpecificity)),   # Specificity (mean, std)
        (mean(testPPV), std(testPPV)),                   # PPV (mean, std)
        (mean(testNPV), std(testNPV)),                   # NPV (mean, std)
        (mean(testF1), std(testF1)),                     # F1-score (mean, std)
        testConfusionMatrix                              # Global test confusion matrix
    )
    return results
end