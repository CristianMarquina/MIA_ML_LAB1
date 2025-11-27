

# ---------------------------------------------------------
# Unit 4.1 Two-class classification
# ---------------------------------------------------------

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

    true_positives = sum(outputs .& targets)
    false_positives = sum(outputs .& .!targets)
    false_negatives = sum(.!outputs .& targets)
    true_negatives = sum(.!outputs .& .!targets)

    # Metrics
    accuracy = (true_negatives + true_positives) / length(outputs)
    error_rate = (false_positives + false_negatives) / length(outputs)

    if true_negatives != length(outputs)
        denominator_sensitivity = false_negatives + true_positives
        denominator_positive_predict_val = true_positives + false_positives
        sensitivity = denominator_sensitivity != 0 ? true_positives / (denominator_sensitivity) : 0
        positive_predict_val = denominator_positive_predict_val != 0 ? true_positives / denominator_positive_predict_val : 0
    else
        sensitivity = 1
        positive_predict_val = 1
    end
    if true_positives != length(outputs)
        denominator_specificity = false_positives + true_negatives
        denominator_negative_predict_val = true_negatives + false_negatives
        specificity = denominator_specificity != 0 ? true_negatives / denominator_specificity : 0
        negative_predict_val = denominator_negative_predict_val != 0 ? true_negatives / denominator_negative_predict_val : 0
    else
        specificity = 1
        negative_predict_val = 1
    end
    # If both sensitivity and positive predictive values are equal to 0,
    # the value of F-score cannot be obtained, and thus it will be 0.
    f_score_denominator = positive_predict_val + sensitivity
    f_score = f_score_denominator == 0 ? 0 : 2 * positive_predict_val * sensitivity / f_score_denominator

    # Confusion matrix
    confusion_matrix = [true_negatives  false_positives; false_negatives true_positives]

    return accuracy, error_rate, sensitivity, specificity, positive_predict_val, negative_predict_val, f_score, confusion_matrix
end

function confusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)

    boolean_outputs = outputs .>= threshold
    return confusionMatrix(boolean_outputs, targets)
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1},targets::AbstractArray{Bool,1})

    metrics = confusionMatrix(outputs, targets)
    accuracy = metrics[1]
    error_rate = metrics[2]
    sensitivity = metrics[3]
    specificity = metrics[4]
    positive_predict_val = metrics[5]
    negative_predict_val = metrics[6]
    f_score = metrics[7]
    confusion_matrix  = metrics[8]
    println("--- Classification Metrics ---")
    @printf("Accuracy:                    %.2f\n", accuracy)
    @printf("Error rate:                  %.2f\n", error_rate)
    @printf("Sensitivity:                 %.2f\n", sensitivity)
    @printf("Specificity:                 %.2f\n", specificity)
    @printf("Precision (PPV):             %.2f\n", positive_predict_val)
    @printf("Negative Pred. Value (NPV):  %.2f\n", negative_predict_val)
    @printf("F1-score:                    %.2f\n", f_score)
    println("--- Confusion Matrix ---")
    println("                | Predicted Negative | Predicted Positive |")
    println("----------------|--------------------|--------------------|")
    @printf("| Real Negative | %-18d | %-18d |\n", confusion_matrix[1, :][1], confusion_matrix[1, :][2])
    println("----------------|--------------------|--------------------|")
    @printf("| Real Positive | %-18d | %-18d |\n", confusion_matrix[2, :][1], confusion_matrix[2, :][2])
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)

    println("Threshold=",threshold)
    boolean_outputs = outputs .>= threshold
    return printConfusionMatrix(boolean_outputs, targets)
end

# ---------------------------------------------------------
# Unit 4.2 Multiclass classification
# ---------------------------------------------------------

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    #TODO
    @assert size(outputs) == size(targets) "Both matrices have to be the same size "
    num_classes = size(targets, 2)
    @assert num_classes != 2 "Matrices with 2 clases (columns) must be reduced to 1 (Binary clasification) "
    if num_classes == 1
        return confusionMatrix(vec(outputs), vec(targets))
    end
    sensitivities = zeros(num_classes)
    specificities = zeros(num_classes)
    ppvs = zeros(num_classes)
    npvs = zeros(num_classes)
    f_scores = zeros(num_classes)

    # Keep track of which classes are non-empty
    is_non_empty = fill(false, num_classes)

    for i in 1:num_classes
        if sum(targets[:, i]) > 0 #class not empty
            is_non_empty[i] = true # Mark as non-empty
            _acc, _err, sens, spec, ppv, npv, f1, _cm = confusionMatrix(outputs[:, i], targets[:, i])#calls the function for each class
            sensitivities[i] = sens
            specificities[i] = spec
            ppvs[i] = ppv
            npvs[i] = npv
            f_scores[i] = f1
        end
    end

    mode = weighted ? :weighted : :macro
    local final_sensitivity, final_specificity, final_ppv, final_npv, final_f_score

    if !weighted #Macro (Arithmetic mean)
        # Filter metrics for only non-empty classes before averaging
        non_empty_sens = sensitivities[is_non_empty]
        non_empty_spec = specificities[is_non_empty]
        non_empty_ppv = ppvs[is_non_empty]
        non_empty_npv = npvs[is_non_empty]
        non_empty_f_scores = f_scores[is_non_empty]

        if length(non_empty_sens) == 0
            # Handle the edge case where no classes have any instances (shouldn't happen often)
            final_sensitivity, final_specificity, final_ppv, final_npv, final_f_score = 0.0, 0.0, 0.0, 0.0, 0.0
        else
            # Average only over non-empty classes
            final_sensitivity = mean(non_empty_sens)
            final_specificity = mean(non_empty_spec)
            final_ppv = mean(non_empty_ppv)
            final_npv = mean(non_empty_npv)
            final_f_score = mean(non_empty_f_scores)
        end
    else #Weighted
        patterns_per_class = vec(sum(targets, dims=1))
        total_patterns = sum(patterns_per_class)
        if total_patterns == 0
            final_sensitivity, final_specificity, final_ppv, final_npv, final_f_score = 0.0, 0.0, 0.0, 0.0, 0.0
        else
            final_sensitivity = sum(sensitivities .* patterns_per_class) / total_patterns
            final_specificity = sum(specificities .* patterns_per_class) / total_patterns
            final_ppv = sum(ppvs .* patterns_per_class) / total_patterns
            final_npv = sum(npvs .* patterns_per_class) / total_patterns
            final_f_score = sum(f_scores .* patterns_per_class) / total_patterns
        end
    end

    acc = accuracy(outputs, targets)
    err = 1.0 - acc

    # Comprehension for the confusion matrix LxL
    conf_matrix = [sum(targets[:, i] .& outputs[:, j]) for i in 1:num_classes, j in 1:num_classes]

    return (
        mode = mode,
        accuracy = acc,
        error_rate = err,
        sensitivity = final_sensitivity,
        specificity = final_specificity,
        ppv = final_ppv,
        npv = final_npv,
        f_score = final_f_score,
        per_class_metrics = (
            sensitivities = sensitivities,
            specificities = specificities,
            ppvs = ppvs,
            npvs = npvs,
            f_scores = f_scores
        ),
        confusion_matrix = conf_matrix
    )
end

function confusionMatrix(outputs::AbstractArray{<:Real,2},targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)

    outputsbool = outputs .>= threshold
    return confusionMatrix(outputsbool, targets, weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)

    @assert issubset(unique([outputs; targets]), classes) "Not all labels in outputs and targets  exist in classes."
    outputs_onehot = oneHotEncoding(outputs, classes)
    targets_onehot = oneHotEncoding(targets, classes)
    return confusionMatrix(outputs_onehot, targets_onehot, weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs))
    return confusionMatrix(outputs, targets, classes, weighted=weighted)
end

function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    results = confusionMatrix(outputs, targets, weighted=weighted)
    num_classes = size(targets, 2)
    println("Metrics (mode: $(results.mode))")
    @printf("Accuracy:      %.2f\n", results.accuracy )
    @printf("F1-Score:      %.2f\n", results.f_score)
    @printf("Sensitivity:   %.2f\n", results.sensitivity)
    @printf("PPV:           %.2f\n\n", results.ppv)

    println("--- Confusion Matrix ---")
    print("      |")
    for j in 1:num_classes
        @printf(" %-9d|", j)
    end
    println("\n" * "-"^ (7 + 11 * num_classes))
    for i in 1:num_classes
        @printf("  %-4d|", i)
        for j in 1:num_classes
            @printf(" %-9d|", results.confusion_matrix[i, j])
        end
        println()
    end
    println("-"^ (7 + 11 * num_classes))
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5 , weighted::Bool=true)

    println("Threshold=",threshold)
    outputsbool = outputs .>= threshold
    printConfusionMatrix(outputsbool, targets, weighted=weighted)
end

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    outputsonehot = oneHotEncoding(outputs, classes)
    targetsonehot = oneHotEncoding(targets, classes)
    printConfusionMatrix(outputsonehot, targetsonehot, weighted=weighted)
end

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs))
    printConfusionMatrix(outputs, targets, classes, weighted=weighted)
end

# Helper function to safely store metric values
function safeStore(value)
    # Checks if the value is a number (Float64, Int64, etc.).
    # If not (i.e., it's a Symbol like :NaN), returns NaN.
    # Otherwise, converts it to Float64.
    return isa(value, Number) ? convert(Float64, value) : NaN
end
