
# One-Hot Encoding

function oneHotEncoding(feature::AbstractArray{<:Any,1},      
        classes::AbstractArray{<:Any,1})
    
    # First, we are going to set a line as defense to check values    
    @assert(all([in(value, classes) for value in feature]))
    
    num_patterns = length(feature)
    num_classes = length(classes)
    
    # Second defensive statement, check the number of classes
    @assert(num_classes>1)
    
    # Length of the vector of unique elements in classes
    if num_classes == 2
        # If feature[x] == 1, it returns true.
        boolean_vector = (feature .== classes[1])
        return reshape(boolean_vector, (:,1))
    else
        boolean_matrix = BitArray{2}(undef, num_patterns, num_classes)
        for idx = 1:num_classes
            boolean_matrix[:, idx] = (feature .== classes[idx])
        end
        return boolean_matrix
    end
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

function oneHotEncoding(feature::AbstractArray{Bool,1})
    # Input is like feature = [true, false, true, true]
    return reshape(feature, (:, 1))
end

# Normalization parameters
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})

    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})

    return mean(dataset, dims=1), std(dataset, dims=1)
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})

    min = normalizationParameters[1]
    max = normalizationParameters[2]
    dataset .= (dataset .- min) ./ (max .- min)
    # Eliminate any attribute that does not add information
    dataset[:, vec(min.==max)] .= 0;
    return dataset
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2},      
                normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 

    normalizeMinMax!(copy(dataset), normalizationParameters)
end    

function normalizeMinMax(dataset::AbstractArray{<:Real,2})

    normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset))
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},      
                        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    
    mean = normalizationParameters[1]
    std = normalizationParameters[2]
    dataset .= (dataset .- mean)./ std
    # Remove any attribute that does not have information
    dataset[:, vec(std.==0)] .= 0
    return dataset
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset))
end

function normalizeZeroMean( dataset::AbstractArray{<:Real,2},      
                            normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    normalizeZeroMean!(copy(dataset), normalizationParameters)
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}) 
    normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset))
end


# Classify outputs

function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
                        threshold::Real=0.5) 
    
    num_columns = size(outputs, 2)
    @assert(num_columns!=2)
    
    if num_columns == 1
        classified_outputs = outputs .>= threshold
        return classified_outputs
    else
        # Look for the maximum value using the findmax function
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2)
        # Set up the boolean matrix to everything false while 
        classified_outputs = falses(size(outputs))
        classified_outputs[indicesMaxEachInstance] .= true
        # Defensive check if all patterns are in a single class
        @assert(all(sum(classified_outputs, dims=2).==1))        
        return classified_outputs
    end
end

# Accuracy
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    return mean(outputs .== targets)
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert(all(size(outputs).==size(targets)))
    num_columns = size(targets, 2) #length(targets[1,:])
    if num_columns == 1
        return accuracy(outputs[:,1], targets[:,1])
    else
        correct_rows = all(outputs .== targets, dims=2)
        return mean(correct_rows)
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
                threshold::Real=0.5)

    # Convert the 1D real vector to a 1D boolean vector
    boolean_outputs = outputs .>= threshold
    return accuracy(boolean_outputs, targets)
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
                threshold::Real=0.5)
    
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets)
    end
end

# Build ANN
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 

    # Keep track of the number of inputs for the next layer
    numInputsLayer = numInputs

    # Empty Chain to hold the layers of the ANN
    ann = Chain();

    # Add hidden layers
    for (i, numOutputsLayer) in enumerate(topology)      
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]));      
        numInputsLayer = numOutputsLayer; 
    end
    #Final output layer
    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
    else 
        # Multi-class classification: output layer N neurons and no activation function
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax);
    end
    return ann
end