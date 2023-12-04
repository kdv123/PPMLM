// Program that tests out the PPMLM class.
// These tests follow those in https://github.com/google-research/google-research/blob/master/jslm/example.js

import Foundation

// TODO: Change to match your system, used to find training text files.
let PPMLM_HOME = "/Users/vertanen/PPMLM/"

// Run training on a large file instead of the normal unit tests
let LOAD_TEST = false

// Actual filenames some of the test depend on.
let DAILY_DIALOG_TRAIN = "\(PPMLM_HOME)/data/daily_train_10k.txt"
let AAC_DEV_TEST = "\(PPMLM_HOME)/data/aac_dev_test.txt"

// Not in github due to size, but you can download from:
// https://data.imagineville.org/daily_train.txt.gz
let DAILY_DIALOG_TRAIN_FULL = "\(PPMLM_HOME)/data/daily_train.txt"

if !LOAD_TEST
{
    // Create a small vocabulary.
    var v = Vocabulary()
    let aSymbol = v.add(token: "a")
    let bSymbol = v.add(token: "b")
    
    // Build the PPM language model trie and update the counts.
    let maxOrder = 5
    var lm = PPMLanguageModel(vocab: v, maxOrder: maxOrder)
    var c = lm.createContext()
    lm.addSymbolToContextAndUpdate(context: c, symbol: aSymbol)
    lm.addSymbolToContextAndUpdate(context: c, symbol: bSymbol)
    print("Initial count trie:")
    lm.printTree()
    
    // Check static (non-adaptive) mode.
    // In the example below we always ignore the 0th symbol. It is a special symbol
    // corresponding to the root of the trie.
    var test = 0
    print("*** Test \(test)"); test += 1
    c = lm.createContext();
    var probs = lm.getProbs(context: c)
    print(probs)
    assert(probs.count == 3, "Expected \"a\", \"b\" and root")
    
    // Nothing has been entered yet. Since we've observed both "a" and "b", there is
    // an equal likelihood of getting either.
    assert(probs[1] > 0 && probs[1] == probs[2], "Probabilities for both symbols should be equal")
    
    // Enter "a" and check the probability estimates. Since we've seen the sequence
    // "ab" during the training, the "b" should be more likely than "a".
    print("*** Test \(test)"); test += 1
    lm.addSymbolToContext(context: c, symbol: aSymbol)
    probs = lm.getProbs(context: c)
    print(probs)
    assert(probs[1] > 0 && probs[1] < probs[2], "Probability for \"b\" should be more likely")
    
    // Enter "b". The context becomes "ab". Now it's back to square one: Any symbol
    // is likely again.
    print("*** Test \(test)"); test += 1
    lm.addSymbolToContext(context: c, symbol: bSymbol)
    probs = lm.getProbs(context: c)
    print(probs)
    assert(probs[1] > 0 && probs[1] == probs[2], "Probabilities for both symbols should be equal")
    
    // Try to enter "ba". Since the model has only observed "ab" sequence, it is
    // expecting the next most probable symbol to be "b".
    print("*** Test \(test)"); test += 1
    c = lm.createContext()
    lm.addSymbolToContext(context: c, symbol: bSymbol)
    lm.addSymbolToContext(context: c, symbol: aSymbol)
    probs = lm.getProbs(context: c)
    print(probs)
    assert(probs[1] > 0 && probs[2] > probs[1], "Probability for \"b\" should be more likely")
    
    // Check adaptive mode in which the model is updated as symbols are entered.
    // Enter "a" and update the model. At this point the frequency for "a" is
    // higher, so it's more probable.
    print("*** Test \(test)"); test += 1
    lm = PPMLanguageModel(vocab: v, maxOrder: maxOrder)
    c = lm.createContext()
    lm.addSymbolToContextAndUpdate(context: c, symbol: aSymbol)
    probs = lm.getProbs(context: c)
    print(probs)
    assert(probs[1] > 0 && probs[1] > probs[2], "Probability for \"a\" should be more likely")
    
    // Enter "b" and update the model. At this point both symbols should become
    // equally likely again.
    print("*** Test \(test)"); test += 1
    lm.addSymbolToContextAndUpdate(context: c, symbol: bSymbol)
    probs = lm.getProbs(context: c)
    print(probs)
    assert(probs[1] > 0 && probs[1] == probs[2], "Probabilities for both symbols should be the same")
    
    // Enter "b" and update the model. Current context "abb". Since we've seen
    // "ab" and "abb" by now, the "b" becomes more likely.
    print("*** Test \(test)"); test += 1
    lm.addSymbolToContextAndUpdate(context: c, symbol: bSymbol)
    probs = lm.getProbs(context: c)
    print(probs)
    assert(probs[1] > 0 && probs[1] < probs[2], "Probability for \"b\" should be more likely")
    
    print("Final count trie:")
    lm.printTree()
    
    // ======================================================================
    // Tests doing language modeling on full alphabet.
    v = Vocabulary()
    let alphabet = "abcdefghijklmnopqrstuvwxyz' "
    v.addAllCharacters(valid: alphabet)
    print("Lowercase plus apostrophe and space, size = \(v.size)")
    print(v)
    
    // Some juicy sentences to train our language model on
    let sentences = ["the cat sat on a mat",
                     "it was the best of times, it was the worst of times!",
                     "the quick brown fox jumps over the lazy dog's tail"]
    
    // Letter unigram language model
    lm = PPMLanguageModel(vocab: v, maxOrder: 0)
    
    // Sanity tests that number of nodes matches unqiue tokens in the training sentences.
    print("*** Test \(test)"); test += 1
    var skipped = lm.train(text: sentences[0])
    assert(skipped == 0, "Should not have skipped any characters in: \(sentences[0])")
    assert((sentences[0].numUniqueCharacters() + 1) == lm.numNodes,
           "Unique characters and node count mismatch, sentence[0]!")
    
    print("*** Test \(test)"); test += 1
    skipped = lm.train(text: sentences[1])
    assert(skipped == 2, "Should have skipped 2 characters in: \(sentences[1])")
    assert(((sentences[0] + sentences[1]).numUniqueCharacters() + 1 - 2) == lm.numNodes,
           "Unique characters and node count mismatch, sentence[0..1]!")
    
    print("*** Test \(test)"); test += 1
    skipped = lm.train(text: sentences[2])
    assert(skipped == 0, "Should not have skipped any characters in: \(sentences[2])")
    assert(lm.numNodes == v.size, "After sentences[0..2] PPM nodes and vocab should be same size!")
    lm.printTree()
    
    // Dictionary version of probability results
    print("*** Test \(test)"); test += 1
    c = lm.createContext()
    var probsDict = lm.getProbsAsDictionary(context: c)
    print(probsDict)
    assert(probsDict.count == alphabet.count, "Dictionary probs should be same size as alphabet!")
    assert(abs(1.0 - probsDict.values.sum()) < Constants.EPSILON, "Dictionary probs don't sum to 1!")
    
    // Train on the three sentences in a single call
    print("*** Test \(test)"); test += 1
    lm = PPMLanguageModel(vocab: v, maxOrder: 0)
    skipped = lm.train(texts: sentences)
    var probsDictAll = lm.getProbsAsDictionary(context: c)
    assert(skipped == 2, "Should have skipped 2 characters in all three sentences!")
    for entry in probsDict
    {
        assert(abs(entry.value - (probsDictAll[entry.key] ?? 0.0)) < Constants.EPSILON, "Mismatch probability \(entry)!")
    }
    
    // Training on a bunch of sentences with a longer order model.
    // Track how long it takes and about how much memory it took.
    print("*** Test \(test)"); test += 1
    var lines = try Utils.readLinesFrom(filename: DAILY_DIALOG_TRAIN)
    var startMem = Utils.memoryInMB()
    var startTime = ProcessInfo.processInfo.systemUptime
    lm = PPMLanguageModel(vocab: v, maxOrder: 8)
    skipped = lm.train(texts: lines)
    var endTime = ProcessInfo.processInfo.systemUptime
    var endMem = Utils.memoryInMB()
    var trainChars = Utils.countCharacters(texts: lines)
    print("Training lines \(lines.count), chars \(trainChars), skipped chars \(skipped), PPM nodes \(lm.numNodes)")
    var elapsed = endTime - startTime
    print("Train time: \(String(format: "%.4f", elapsed))" +
          ", chars/second: \(String(format: "%.1f", (Double(trainChars) / elapsed)))")
    var memMB = Double(endMem - startMem)
    print("Memory increase in MB: \(String(format: "%.2f", memMB))")
    var bytesPerNode = Double(endMem - startMem) / Double(lm.numNodes)
    print("Estimated bytes per Node: \(String(format: "%.2f", bytesPerNode))")
    
    // ======================================================================
    // Inference time! First we'll test things out with a static LM.
    
    // First try a single sentence with no skipped characters.
    print("*** Test \(test)"); test += 1
    var result = lm.evaluate(text: sentences[0])
    print(result)
    // NOTE: test answers were not validated against anything else.
    assert(abs(-18.518700861203644 - result.sumLogProb) < Constants.EPSILON, "Inference single sentence logprob didn't match!")
    assert(abs(8.432086305605736 - result.perplexity) < Constants.EPSILON, "Inference single sentence perplexity didn't match!")
    assert(result.tokensGood == sentences[0].count, "Inference single sentence wrong number of good tokens!")
    assert(result.tokensSkipped == 0, "Inference single sentence wrong number of skipped tokens!")
    
    // Doing it again without adaptation should yield same answer.
    print("*** Test \(test)"); test += 1
    var result2 = lm.evaluate(text: sentences[0])
    assert(result == result2, "Mismatch in second inference on single sentence!")
    
    // Calculate on the set of three sentences.
    print("*** Test \(test)"); test += 1
    result = lm.evaluate(texts: sentences)
    print(result)
    // NOTE: test answers were not validated against anything else.
    assert(abs(-86.18911019761052 - result.sumLogProb) < Constants.EPSILON, "Inference multiple sentences logprob didn't match!")
    assert(abs(5.226880666850305 - result.perplexity) < Constants.EPSILON, "Inference multiple sentences perplexity didn't match!")
    assert(result.tokensGood == Utils.countCharacters(texts: sentences) - 2, "Inference multiple sentences wrong number of good tokens!")
    assert(result.tokensSkipped == 2, "Inference multiple sentences wrong number of skipped tokens!")
    
    // Calculate on 1K sentences from the daily dialog dev set.
    // Also time how long this takes.
    print("*** Test \(test)"); test += 1
    lines = try Utils.readLinesFrom(filename: AAC_DEV_TEST)
    startTime = ProcessInfo.processInfo.systemUptime
    result = lm.evaluate(texts: lines)
    endTime = ProcessInfo.processInfo.systemUptime
    var evalChars = Utils.countCharacters(texts: lines)
    print(result)
    elapsed = endTime - startTime
    print("Eval time: \(String(format: "%.4f", elapsed))" +
          ", chars/second: \(String(format: "%.1f", (Double(evalChars) / elapsed)))")
    // NOTE: test answers were not validated against anything else.
    assert(abs(-87724.20036642274 - result.sumLogProb) < Constants.EPSILON, "Inference multiple sentences logprob didn't match!")
    assert(result.tokensGood == evalChars, "Eval dev wrong number of good tokens!")
    assert(result.tokensSkipped == 0, "Eval dev wrong number of skipped tokens!")
    
    // ======================================================================
    // Inference time! Now test out adapting the model while we evaluate.
    print("*** Test \(test)"); test += 1
    result = lm.evaluate(text: sentences[0], updateModel: true)
    print(result)
    // Got slightly worse compared to first time with static model.
    assert(abs(-18.523323481788257 - result.sumLogProb) < Constants.EPSILON, "Inference single sentence logprob didn't match!")
    
    // Keep doing the previous sentence and verify it gets more and more probable.
    print("*** Test \(test)"); test += 1
    var resultLast = result
    for i in 0...9
    {
        result = lm.evaluate(text: sentences[0], updateModel: true)
        print(result)
        assert(result.sumLogProb > resultLast.sumLogProb, "Repeated eval with updated didn't get better, iteration \(i)")
        resultLast = result
    }
    
    // Calculate on 1K sentences from the daily dialog dev set.
    // Also time how long this takes.
    print("*** Test \(test)"); test += 1
    // Reset the PPM model since we update it in the last two tests
    lines = try Utils.readLinesFrom(filename: DAILY_DIALOG_TRAIN)
    lm = PPMLanguageModel(vocab: v, maxOrder: 8)
    skipped = lm.train(texts: lines)
    lines = try Utils.readLinesFrom(filename: AAC_DEV_TEST)
    startTime = ProcessInfo.processInfo.systemUptime
    result = lm.evaluate(texts: lines, updateModel: true)
    endTime = ProcessInfo.processInfo.systemUptime
    evalChars = Utils.countCharacters(texts: lines)
    print(result)
    elapsed = endTime - startTime
    print("Eval time with update: \(String(format: "%.4f", elapsed))" +
          ", chars/second: \(String(format: "%.1f", (Double(evalChars) / elapsed)))")
    // NOTE: test answers were not validated against anything else.
    // This logprob sum was a bit lower than without adapting.
    assert(abs(-82724.08132924688 - result.sumLogProb) < Constants.EPSILON, "Eval dev adaptive logprob didn't match!")
    
    // Training and evaluate on the full daily dialog training and dev sets.
    print("*** Test \(test)"); test += 1
    lines = try Utils.readLinesFrom(filename: DAILY_DIALOG_TRAIN_FULL)
    startMem = Utils.memoryInMB()
    startTime = ProcessInfo.processInfo.systemUptime
    
    // Automatically grow the tree
    //lm = PPMLanguageModel(vocab: v, maxOrder: 9)
    // Size of the training tree
    //lm = PPMLanguageModel(vocab: v, maxOrder: 9, reserveCapacity: 2234348)
    // Size of the training + eval tree
    lm = PPMLanguageModel(vocab: v, maxOrder: 9, reserveCapacity: 2380791)
    
    skipped = lm.train(texts: lines)
    endTime = ProcessInfo.processInfo.systemUptime
    endMem = Utils.memoryInMB()
    trainChars = Utils.countCharacters(texts: lines)
    print("Training lines \(lines.count), chars \(trainChars), skipped chars \(skipped), PPM nodes \(lm.numNodes)")
    elapsed = endTime - startTime
    print("Train time: \(String(format: "%.4f", elapsed))" +
          ", chars/second: \(String(format: "%.1f", (Double(trainChars) / elapsed)))")
    memMB = Double(endMem - startMem)
    print("Memory increase in MB: \(String(format: "%.2f", memMB))")
    bytesPerNode = Double(endMem - startMem) / Double(lm.numNodes)
    print("Estimated bytes per Node: \(String(format: "%.2f", bytesPerNode))")
    print("Num nodes: \(lm.numNodes)")
    let stats = lm.statsTree()
    print("Tree stats: \(stats)")
    
    lines = try Utils.readLinesFrom(filename: AAC_DEV_TEST)
    startTime = ProcessInfo.processInfo.systemUptime
    result = lm.evaluate(texts: lines, updateModel: true)
    endTime = ProcessInfo.processInfo.systemUptime
    evalChars = Utils.countCharacters(texts: lines)
    print(result)
    elapsed = endTime - startTime
    print("Eval time with update: \(String(format: "%.4f", elapsed))" +
          ", chars/second: \(String(format: "%.1f", (Double(evalChars) / elapsed)))")
    assert(abs(-78138.42692423137 - result.sumLogProb) < Constants.EPSILON, "Eval dev adaptive logprob didn't match!")
    print("Num nodes: \(lm.numNodes)")
    
    print("*** TESTS COMPLETED")
}
else
{
    // For testing training and evaluation on a large amount of data.
    // These files aren't on github.
    let SUBTITLE_CE_TRAIN = "\(PPMLM_HOME)/data/subtitle_single_opt_train_lower_ce0.10_word.txt"
    let REDDIT_CE_TRAIN = "\(PPMLM_HOME)/data/reddit_single_opt_train_lower_ce0.00_word.txt"
    let COMMON_CE_TRAIN = "\(PPMLM_HOME)/data/common_single_opt_train_lower_ce0.00_word.txt"
    let TWITTER_CE_TRAIN = "\(PPMLM_HOME)/data/twitter_single_opt_train_lower_ce0.05_word.txt"

    // Single training test
    var v = Vocabulary()
    let alphabet = "abcdefghijklmnopqrstuvwxyz' "
    v.addAllCharacters(valid: alphabet)

    var startMem = Utils.memoryInMB()
    var startTime = ProcessInfo.processInfo.systemUptime
    var lm = PPMLanguageModel(vocab: v, maxOrder: 9)

    var skipped = 0
    var lineCount = 0
    var trainChars = 0
    if freopen(COMMON_CE_TRAIN, "r", stdin) != nil
    {
        while let line = readLine()
        {
            skipped += lm.train(text: line)
            lineCount += 1
            trainChars += line.count
            if lineCount > 1_000_000
            {
                break
            }
        }
    }

    var endTime = ProcessInfo.processInfo.systemUptime
    var endMem = Utils.memoryInMB()

    print("Training lines \(lineCount), chars \(trainChars), skipped chars \(skipped), PPM nodes \(lm.numNodes)")
    var elapsed = endTime - startTime
    print("Train time: \(String(format: "%.4f", elapsed))" +
          ", chars/second: \(String(format: "%.1f", (Double(trainChars) / elapsed)))")
    var memMB = endMem - startMem
    print("Memory increase in MB: \(memMB)")
    var bytesPerNode = Double(endMem - startMem) / Double(lm.numNodes)
    print("Estimated bytes per Node: \(String(format: "%.2f", bytesPerNode))")
    print("Num nodes: \(lm.numNodes)")
    let stats = lm.statsTree()
    print("Tree stats: \(stats)")

    var lines = try Utils.readLinesFrom(filename: AAC_DEV_TEST)
    startTime = ProcessInfo.processInfo.systemUptime
    var result = lm.evaluate(texts: lines, updateModel: true)
    endTime = ProcessInfo.processInfo.systemUptime
    var evalChars = Utils.countCharacters(texts: lines)
    print(result)
    elapsed = endTime - startTime
    print("Eval time with update: \(String(format: "%.4f", elapsed))" +
          ", chars/second: \(String(format: "%.1f", (Double(evalChars) / elapsed)))")
    print("Num nodes: \(lm.numNodes)")
    endMem = Utils.memoryInMB()
    memMB = endMem - startMem
    print("End memory increase in MB: \(memMB)")
}
