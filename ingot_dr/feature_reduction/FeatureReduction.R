options(warn = 0)
library(combinat)
library(magrittr)
library(tidyverse)
library(optparse)

mainDriver = function(genoFile, phenoFile, outFile) {
  G = read_csv(genoFile)
  P = read_csv(phenoFile)
  Gr = removeDominatedFeatures(G, P)
  write_csv(Gr, outFile)
  return(TRUE)
}

### Inputs: G, a tibble with 2 int columns, "row", and "col", representing the positions with a 1 in the genotype;
### P, a tibble with 1 int column, "row", representing the positions with a 1 in the phenotype.
### Assumptions: no zero samples; no zero features; no duplicated samples (if there are any they should be collapsed).
### Output: Gr, a reduced tibble in the same format as G, but omitting any dominated features, where we say that a 
### feature f is dominated by a feature g if any 1 in both f and P is a 1 in g, and any 0 in both f and P is a 0 in g.
removeDominatedFeatures = function(G, P) {
  stopifnot(ncol(G) == 2)
  stopifnot(ncol(P) == 1)
  stopifnot(colnames(G) == c("row", "col"))
  stopifnot(colnames(P) == "row")
  G = removeDuplicateColumns(G)
  N = max(G$col)
  GPos = G %>%
    filter(row %in% P$row)
  GNeg = G %>%
    filter(!(row %in% P$row))
  groupsPos = identifyDuplicateColumns(GPos, numCols = N)
  groupsNeg = identifyDuplicateColumns(GNeg, numCols = N)
  dominatedPos = findLocallyDominatedColumns(GPos, groupsNeg, negate = FALSE)
  dominatedNeg = findLocallyDominatedColumns(GNeg, groupsPos, negate = TRUE)
  GPos = GPos %>%
    filter(!(col %in% c(dominatedPos, dominatedNeg)))
  GNeg = GNeg %>%
    filter(!(col %in% c(dominatedPos, dominatedNeg)))
  N = max(setdiff(1:N, c(dominatedPos, dominatedNeg)))
  dominated = c()
  if (nrow(GPos) > 0 && nrow(GNeg) > 0) {
    posSizes = computeSizes(GPos, col = TRUE, decreasing = TRUE, size = N)
    negSizes = computeSizes(GNeg, col = TRUE, decreasing = FALSE, size = N)
    allSizes = posSizes %>%
      full_join(negSizes, by = "col", suffix = c(".pos", ".neg"))
    dominated = findDominatedColumns(GPos, GNeg, allSizes)
  } else if (nrow(GPos) > 0) {
    curGroup = tibble(groupID = 1L, colGroup = sort(unique(GPos$col)))
    dominated = findLocallyDominatedColumns(GPos, curGroup, negate = FALSE)
  } else {
    curGroup = tibble(groupID = 1L, colGroup = sort(unique(GNeg$col)))
    dominated = findLocallyDominatedColumns(GNeg, curGroup, negate = TRUE)
  }
  Gr = G %>%
    filter(!(col %in% c(dominated, dominatedPos, dominatedNeg)))
  Gr
}

convertToSparse = function(posMat, negMat) {
  M = nrow(posMat)
  N = ncol(posMat)
  stopifnot(ncol(negMat) == N)
  GPos = which(posMat == TRUE, arr.ind = TRUE) %>%
    as_tibble()
  GNeg = which(negMat == TRUE, arr.ind = TRUE) %>%
    as_tibble() %>%
    mutate_at("row", ~ { magrittr::add(., nrow(posMat)) })
  G = bind_rows(GPos, GNeg)
  ### eliminate zero features
  goodCols = sort(unique(G$col))
  G = G %>%
    mutate(col = match(col, goodCols))
  ### eliminate zero samples
  goodRows = sort(unique(G$row))
  G = G %>%
    mutate(row = match(row, goodRows))
  MPos = max(which(goodRows <= M))
  goodPRows = 1:MPos
  P = enframe(1:MPos, name = NULL, value = "row")
  ### eliminate duplicate samples
  duplicates = identifyDuplicateRows(G)
  badRows = duplicates %>%
    mutate_at("rowGroup", ~ { map(., ~ { str_split(., ", ") %>% unlist() %>% magrittr::extract(-1) %>% as.integer }) }) %>%
    unnest(rowGroup) %>%
    pull(rowGroup)
  if (length(badRows) > 0) {
    goodRows = setdiff(1:max(G$row), badRows)
    goodPRows = goodRows[goodRows <= MPos]
    G = G %>%
      filter(row %in% goodRows) %>%
      mutate(row = match(row, goodRows))
    P = enframe(match(goodPRows, goodRows), name = NULL, value = "row")
  }
  output = list(G = G, P = P)
  output
}

### For backward compatibility, this function works on full binary matrices pre-split according to a binary phenotype
### The output is a list containing the reduced positive and negative full binary matrices and the eliminated columns
removeDominated = function(posMat, negMat) {
  output = convertToSparse(posMat, negMat)
  result = removeDominatedFeatures(output$G, output$P)
  outNonDominated = sort(unique(result$col))
  outPos = posMat[, outNonDominated, drop = FALSE]
  outNeg = negMat[, outNonDominated, drop = FALSE]
  output = list(outPos, outNeg, outNonDominated)
  output
}

convertToBinary = function(Tibble, numRows = NULL, numCols = NULL) {
  if (is.null(numRows)) {
    numRows = max(Tibble$row)
  }
  if (is.null(numCols)) {
    numCols = max(Tibble$col)
  }
  Mat = matrix(FALSE, numRows, numCols)
  Mat[cbind(Tibble$row, Tibble$col)] = TRUE
  Mat
}

identifyDuplicateColumns = function(G, numCols = NULL) {
  if (is.null(numCols)) {
    numCols = max(G$col)
  }
  colGroups = G %>%
    group_by(col) %>%
    mutate(List = paste0(sort(row), collapse = ", ")) %>%
    slice(1) %>%
    ungroup() %>%
    group_by(List) %>%
    mutate(N = n()) %>%
    filter(N > 1) %>%
    mutate(colGroup = paste0(sort(col), collapse = ", ")) %>%
    slice(1) %>%
    ungroup() %>%
    select(colGroup) %>%
    rowid_to_column("groupID") %>%
    mutate_at("colGroup", ~ { str_split(., ", ") }) %>%
    unnest(colGroup) %>%
    mutate_at("colGroup", as.integer)
  emptyCols = setdiff(1:numCols, unique(G$col))
  if (length(emptyCols) > 0) {
    groupNum = ifelse(nrow(colGroups) == 0, 1, max(colGroups$groupID) + 1)
    colGroups = bind_rows(colGroups,
                          tibble(groupID = groupNum, colGroup = emptyCols))
  }
  colGroups
}

removeDuplicateColumns = function(G) {
  keepCols = G %>%
    group_by(col) %>%
    mutate(List = paste0(sort(row), collapse = ", ")) %>%
    ungroup %>%
    distinct(List, .keep_all = TRUE) %>%
    pull(col)
  redG = G %>%
    filter(col %in% keepCols)
  redG
}

identifyDuplicateRows = function(G) {
  rowGroups = G %>%
    group_by(row) %>%
    mutate(List = paste0(sort(col), collapse = ", ")) %>%
    slice(1) %>%
    ungroup() %>%
    group_by(List) %>%
    mutate(N = n()) %>%
    filter(N > 1) %>%
    mutate(rowGroup = paste0(sort(row), collapse = ", ")) %>%
    slice(1) %>%
    ungroup() %>%
    select(rowGroup)
  rowGroups
}

findLocallyDominatedColumns = function(G, groups, negate = FALSE) {
  dominated = c()
  if (nrow(groups) > 0) {
    M = max(groups$groupID)
    for (index in 1:M) {
      curGroup = groups %>%
        filter(groupID == index) %>%
        pull(colGroup)
      curMat = G %>%
        filter(col %in% curGroup) %>%
        mutate(col = match(col, curGroup))
      curNum = length(curGroup)
      goodCols = findOptimalColumns(curMat, minimal = negate, numCols = curNum)
      badCols = setdiff(1:curNum, goodCols)
      dominated = dominated %>%
        c(curGroup[badCols])
    }
  }
  dominated
}

findDominatedColumns = function(GPositive, GNegative, allSizes) {
  maxSizePos = allSizes %>%
    slice(1) %>%
    pull(size.pos)
  survivors = allSizes %>%
    filter(size.pos == maxSizePos) %>%
    pull(col)
  print(paste("There are", nrow(allSizes), "rows to process"))
  if (nrow(allSizes) > length(survivors)) {
    for (ind in (length(survivors) + 1):nrow(allSizes)) {
      if (ind %% 100 == 0) { print(ind) }
      survive = TRUE
      curInfo = allSizes %>%
        slice(ind)
      curCol = curInfo %>%
        pull(col)
      curSizePos = curInfo %>%
        pull(size.pos)
      curSizeNeg = curInfo %>%
        pull(size.neg)
      curCandidates = allSizes %>%
        slice(1:(ind - 1)) %>%
        filter(size.pos > curSizePos & size.neg < curSizeNeg & col %in% survivors) %>%
        pull(col)
      if (length(curCandidates) > 0) {
        redCandidates = findSubsets(GPositive, curCol, curSizePos, curCandidates, super = TRUE)
        if (length(redCandidates) > 0) {
          finalCandidates = findSubsets(GNegative, curCol, curSizeNeg, redCandidates, super = FALSE)
          survive = (length(finalCandidates) == 0)
        }
      }
      if (survive) {
        survivors = survivors %>%
          c(curCol)
      }
    }
  }
  dominated = setdiff(allSizes$col, survivors) %>%
    sort()
  dominated
}

findOptimalColumns = function(G, minimal = TRUE, numCols = NULL) {
  if (is.null(numCols)) {
    numCols = max(G$col)
  }
  sizes = computeSizes(G, col = TRUE, decreasing = !(minimal), size = numCols)
  if (minimal && any(sizes$size == 0) || (!minimal && all(sizes$size == 0))) {
    optimal = sizes %>%
      filter(size == 0) %>%
      slice(1) %>%
      pull(col)
    return(optimal)
  }
  optSize = sizes %>%
    slice(1) %>%
    pull(size)
  numOpt = sizes %>%
    filter(size == optSize) %>%
    nrow()
  optimal = sizes %>%
    slice(1:numOpt) %>%
    pull(col)
  if (nrow(sizes) > numOpt) {
    for (ind in (numOpt + 1):nrow(sizes)) {
      curInfo = sizes %>%
        slice(ind)
      curSize = curInfo %>%
        pull(size)
      curCol = curInfo %>%
        pull(col)
      if (length(findSubsets(G, curCol, curSize, optimal, super = !minimal)) == 0) {
        optimal = optimal %>%
          c(curCol)
      }
    }
  }
  optimal
}

findSubsets = function(G, curCol, curSize, searchSpace, super = FALSE) {
  curPos = G %>%
    filter(col == curCol) %>%
    pull(row)
  compPos = G %>%
    filter(col %in% searchSpace) %>%
    mutate(mark = (row %in% curPos)) %>%
    group_by(col)
  if (super) {
    compPos = compPos %>%
      summarize(N = sum(mark))
    output = compPos %>%
      filter(N == curSize) %>%
      pull(col)
  } else {
    compPos = compPos %>%
      summarize(N = sum(!mark))
    output = compPos %>%
      filter(N == 0) %>%
      pull(col)
  }
  output
}

computeSizes = function(G, col = TRUE, decreasing = TRUE, size = NULL) {
  if (col) {
    sizes = G %>%
      group_by(col)
    if (is.null(size)) {
      size = max(G$col)
    }
    extraEnts = setdiff(1:size, G$col)
  } else {
    sizes = G %>%
      group_by(row)
    if (is.null(size)) {
      size = max(G$row)
    }
    extraEnts = setdiff(1:size, G$row)
  }
  sizes = sizes %>%
    summarize(size = n()) %>%
    ungroup()
  if (length(extraEnts) > 0) {
    sizes = tibble(extraEnts, size = 0) %>%
      set_colnames(colnames(sizes)) %>%
      bind_rows(sizes)
  }
  if (decreasing) {
    sizes = sizes %>%
      arrange(-size)
  } else {
    sizes = sizes %>%
      arrange(size)
  }
  sizes
}


testDomination = function(writeBinary = FALSE, writeSparse = TRUE) {
  ### positive control: only one column survives when all the columns are identical
  vec1 <- (runif(5) < 0.5)
  setsP0 <- matrix(vec1, 5, 5)
  colnames(setsP0) <- LETTERS[1:ncol(setsP0)]
  vec2 <- (runif(5) > 0.5)
  setsN0 <- matrix(vec2, 5, 5)
  colnames(setsN0) <- colnames(setsP0)
  test0 <- removeDominated(setsP0, setsN0)
  stopifnot(all(test0[[3]] == 1))
  if (writeBinary) {
    write_csv(setsP0, "P0.csv")
    write_csv(setsN0, "N0.csv")
    write_csv(tibble(test0[[3]]), "Out0.csv")
  }
  if (writeSparse) {
    res0 = convertToSparse(setsP0, setsN0)
    write_csv(res0$G, "genoFile0.csv")
    write_csv(res0$P, "phenoFile0.csv")
  }
  ### negative control: no domination when one or both of the matrices are diagonal
  setsP1 <- matrix(FALSE, 5, 5)
  diag(setsP1) <- TRUE
  colnames(setsP1) <- LETTERS[1:ncol(setsP1)]
  setsN1 <- matrix(TRUE, 5, 5)
  diag(setsN1) <- FALSE
  colnames(setsN1) <- colnames(setsP1)
  test1 <- removeDominated(setsP1, setsN1)
  stopifnot(all(sort(test1[[3]]) == 1:5))
  if (writeBinary) {
    write_csv(setsP1, "P1.csv")
    write_csv(setsN1, "N1.csv")
    write_csv(tibble(test1[[3]]), file = "Out1.csv")
  }
  if (writeSparse) {
    res1 = convertToSparse(setsP1, setsN1)
    write_csv(res1$G, "genoFile1.csv")
    write_csv(res1$P, "phenoFile1.csv")
  }
  ### positive control: every set except the last one is dominated by the last one
  setsP2 <- matrix(FALSE, 5, 5)
  setsP2[combn2(1:5)] <- TRUE
  colnames(setsP2) <- LETTERS[1:ncol(setsP2)]
  setsN2 <- matrix(TRUE, 5, 5)
  setsN2[combn2(1:5)] <- FALSE
  colnames(setsN2) <- colnames(setsP2)
  test2 <- removeDominated(setsP2, setsN2)
  stopifnot(all(test2[[3]] == 5))
  if (writeBinary) {
    write_csv(setsP2, "P2.csv")
    write_csv(setsN2, "N2.csv")
    write_csv(tibble(test2[[3]]), file = "Out2.csv")
  }
  if (writeSparse) {
    res2 = convertToSparse(setsP2, setsN2)
    write_csv(res2$G, "genoFile2.csv")
    write_csv(res2$P, "phenoFile2.csv")
  }
  ### positive control: 1 dominated by 2 (strict in the -ve); 3 dominated by 4 (in the +ve) dominated by 5 (in the -ve)
  setsP3 <- matrix(FALSE, 5, 5)
  setsP3[1:2, 1:2] <- TRUE
  setsP3[3:5, 3:5] <- TRUE;
  setsP3[5, 3] <- FALSE
  setsN3 <- matrix(FALSE, 5, 5)
  setsN3[1, 1:2] <- TRUE
  setsN3[2, 1] <- TRUE
  setsN3[3:5, 3:5] <- TRUE
  setsN3[5, 5] <- FALSE
  colnames(setsP3) <- LETTERS[1:ncol(setsP3)]
  colnames(setsN3) <- LETTERS[1:ncol(setsN3)]
  test3 <- removeDominated(setsP3, setsN3)
  stopifnot(all(sort(test3[[3]]) == c(2, 5)))
  if (writeBinary) {
    write_csv(setsP3, "P3.csv")
    write_csv(setsN3, "N3.csv")
    write_csv(tibble(test3[[3]]), "Out3.csv")
  }
  if (writeSparse) {
    res3 = convertToSparse(setsP3, setsN3)
    write_csv(res3$G, "genoFile3.csv")
    write_csv(res3$P, "phenoFile3.csv")
  }
  return(TRUE)
}

prepareTabs = function() {
  Tab = read_csv("SimulatedData/genotypes0.csv") %>%
    select(row, col, isolates, Phenotype)
  P = Tab %>%
    select(row, Phenotype) %>%
    distinct() %>%
    arrange(row) %>%
    rowid_to_column("index")
  G = Tab %>%
    select(row, col) %>%
    arrange(row) %>%
    left_join(P)
  P = P %>%
    select(index, Phenotype) %>%
    rename("row" = "index") %>%
    filter(Phenotype == 1) %>%
    select(row)
  G = G %>%
    select(index, col) %>%
    rename("row" = "index")
  output = list(G, P)
  output
}

# Q = prepareTabs(); G = Q[[1]]; P = Q[[2]]

option_list = list(
  make_option(c("-g", "--genoFile"), type = "character", help = "full path to genotype file", metavar = "string"),
  make_option(c("-p", "--phenoFile"), type = "character", help = "full path to phenotype file", metavar = "string"),
  make_option(c("-o", "--outFile"), type = "character", help = "full path to output file", metavar = "string"))
optParser <- OptionParser(option_list = option_list)
opt <- parse_args(optParser)
print("Here are the options that I parsed from your input:")
print(opt)
if (length(opt) > 1) {
  ### more than just the help parameter is available
  print(paste("The files I am now processing are", opt$genoFile, "and", opt$phenoFile))
  result <- mainDriver(genoFile = opt$genoFile, phenoFile = opt$phenoFile, outFile = opt$outFile)
}
