
function holdOut(N::Int, P::Real)
    # Ensure that P is between 0 and 1
    @assert ((P>=0.) & (P<=1.))

    # Number of training patterns
    num_train = round(Int, N*(1-P))

    # Random permutation of all indices
    indices = randperm(N)

    # Split the shuffled indices into training and testing sets
    train_indices = indices[1:num_train]
    test_indices = indices[num_train+1:end]

    return (train_indices, test_indices)
end;

function holdOut(N::Int, Pval::Real, Ptest::Real) 
    # Ensure that Ptest + Pval is between 0 and 1
    @assert (Ptest + Pval <= 1.);

    # Get test indices
    (train_val_indices, test_indices) = holdOut(N, Ptest)
    # Get training and validation indices
    # Adjust Pval
    N_train_val = length(train_val_indices)
    Pval_adjusted = Pval * N/N_train_val
    (train_indices, val_indices) = holdOut(N_train_val, Pval_adjusted)

    return (train_indices, val_indices, test_indices)
end;

function trainClassANN(topology::AbstractArray{<:Int,1},  
            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
            validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
            maxEpochsVal::Int=20, showText::Bool=false) 
      
    # 1. PREPARE DATA AND MODEL
    # Get inputs and targets with the patterns in the rows
    # The transposition for Flux will be applied later
    inputsTrain, targetsTrain = trainingDataset
    inputsVal, targetsVal = validationDataset
    inputsTest, targetsTest = testDataset

    # Check that we have the same number of patterns between sets
    @assert(size(inputsTrain, 1)==size(targetsTrain, 1));
    @assert(size(inputsVal, 1)==size(targetsVal, 1));
    @assert(size(inputsTest, 1)==size(targetsTest, 1));

    # Check that the number of columns matches between sets
    !isempty(inputsVal) && @assert(size(inputsTrain, 2)==size(inputsVal, 2));
    !isempty(inputsTest) && @assert(size(inputsTrain, 2)==size(inputsTest, 2));
    
    # Number of inputs/outputs is the number of columns 
    num_inputs = size(inputsTrain, 2)
    num_outputs = size(targetsTrain, 2)

    # Define model
    ann = buildClassANN(num_inputs, topology, num_outputs; transferFunctions=transferFunctions)

    # 2. DEFINE TRAINING PARAMETERS
    # Get loss function 
    loss(m, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(m(x),y) : Losses.crossentropy(m(x),y)
    # Set optimizer and learningRate
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Arrays to store loss values
    # Note: [] is Vector{Any}, while Float32[] is Vector{Float32} (more efficient)
    train_loss_history = Float32[]
    val_loss_history = Float32[]
    test_loss_history = Float32[]

    # Include cycle 0
    current_train_loss = loss(ann, inputsTrain', targetsTrain')
    push!(train_loss_history, current_train_loss)
    
    if !isempty(inputsVal)
        current_val_loss = loss(ann, inputsVal', targetsVal')
        push!(val_loss_history, current_val_loss)
    end
    
    if !isempty(inputsTest)
        current_test_loss = loss(ann, inputsTest', targetsTest')
        push!(test_loss_history, current_test_loss)
    end
    
    # Best validation loss set to Inf because a new loss value could be <0
    best_val_loss = Inf 
    epochs_without_improvement = 0
    # Deepcopy makes a fully independent copy, whereas copy stores pointers
    best_ann = deepcopy(ann)

    # 3. RUN TRAINING LOOP
    # Train ANN
    for epoch in 1:maxEpochs
        # Train for one epoch
        Flux.train!(loss, ann, [(inputsTrain', targetsTrain')], opt_state)
        
        # Calculate losses
        current_train_loss = loss(ann, inputsTrain', targetsTrain')
        push!(train_loss_history, current_train_loss)
        if showText && (epoch % 20 == 0)
            # Print loss values if showText and epoch is a multiple of 20
            @printf("Epoch %d/%d - Training Loss: %.4f", epoch, maxEpochs, current_train_loss)
        end
        if !isempty(inputsVal)
            current_val_loss = loss(ann, inputsVal', targetsVal')
            push!(val_loss_history, current_val_loss)
            if showText && (epoch % 20 == 0)
                @printf(" - Validation Loss: %.4f", current_val_loss)
            end
            if current_val_loss < best_val_loss
                best_val_loss = current_val_loss
                best_ann = deepcopy(ann)
                epochs_without_improvement = 0
            else
                epochs_without_improvement +=1
            end
        end

        if !isempty(inputsTest)
            current_test_loss = loss(ann, inputsTest', targetsTest')
            push!(test_loss_history, current_test_loss)
            if showText && (epoch % 20 == 0)
                @printf(" - Test Loss: %.4f", current_test_loss)
            end
        end
        if showText && (epoch % 20 == 0)
            @printf("\n")
        end

        
        # EARLY STOPPING: validation set
        if epochs_without_improvement >= maxEpochsVal
            if showText
                println("Stopping at epoch $epoch: Validation loss has not improved for $maxEpochsVal epochs.")
            end
            break
        end       
        
        # EARLY STOPPING: no validation set
        if current_train_loss < minLoss
            if showText
                println("Stopping at epoch $epoch due to reaching minimum loss.")
            end
            break
        end

        if epoch == maxEpochs
            if showText
                println("Stopping due to reaching maximum number of epochs.")
            end
        end
    end
        
    # 4. RETURN TRAINED MODEL AND LOSS HISTORY
    final_ann = isempty(inputsVal) ? ann : best_ann
    return final_ann, train_loss_history, val_loss_history, test_loss_history
end;

function trainClassANN(topology::AbstractArray{<:Int,1},  
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
        maxEpochsVal::Int=20, showText::Bool=false)

    inputsTrain, targetsTrain = trainingDataset
    inputsVal, targetsVal = validationDataset
    inputsTest, targetsTest = testDataset

    vectTrainingDataset = (inputsTrain, reshape(targetsTrain, (:, 1)))
    vectValidationDataset = (inputsVal, reshape(targetsVal, (:, 1)))
    vectTestDataset = (inputsTest, reshape(targetsTest, (:, 1)))
    return trainClassANN(
        topology,  
        vectTrainingDataset; 
        validationDataset=vectValidationDataset,     
        testDataset=vectTestDataset, 
        transferFunctions=transferFunctions, 
        maxEpochs=maxEpochs, 
        minLoss=minLoss, 
        learningRate=learningRate,  
        maxEpochsVal=maxEpochsVal, 
        showText=showText) 
end;