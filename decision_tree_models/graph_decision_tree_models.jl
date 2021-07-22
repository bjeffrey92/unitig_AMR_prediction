using Random
using DecisionTree # my fork of the original DecisionTree library

function train_test_split(X, y, test_size = 0.3, seed = 42)
    sample_size = length(y)
    @assert size(X)[1] == sample_size
    
    idx = collect(1:sample_size)
    rng = MersenneTwister(seed)
    shuffle!(rng, idx)

    test_size = Int(round(sample_size * test_size))
    test_idx = idx[1:test_size]
    train_idx = idx[test_size+1:sample_size]

    return X[train_idx,:], y[train_idx], X[test_idx,:], y[test_idx]
end

#TODO: functions to parse inputs
X_train, y_train, X_test, y_test = train_test_split(features, labels)
