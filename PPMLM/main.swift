// Program that tests out the PPMLM class.
// These tests follow those in https://github.com/google-research/google-research/blob/master/jslm/example.js

import Foundation

// TODO: Change to match your system, used to find training text files.
let PPMLM_HOME = "/Users/vertanen/PPMLM/"

// Actual filenames some of the test depend on.
let DAILY_DIALOG_TRAIN = "\(PPMLM_HOME)/data/daily_train_10k.txt"
let AAC_DEV_TEST = "\(PPMLM_HOME)/data/aac_dev_test.txt"

// Not in github due to size, but you can download from:
// https://data.imagineville.org/daily_train.txt.gz
let DAILY_DIALOG_TRAIN_FULL = "\(PPMLM_HOME)/data/daily_train.txt"

let SUBTITLE_CE_TRAIN = "\(PPMLM_HOME)/data/subtitle_single_opt_train_lower_ce0.10_word.txt"
let REDDIT_CE_TRAIN = "\(PPMLM_HOME)/data/reddit_single_opt_train_lower_ce0.00_word.txt"
let COMMON_CE_TRAIN = "\(PPMLM_HOME)/data/common_single_opt_train_lower_ce0.00_word.txt"
let TWITTER_CE_TRAIN = "\(PPMLM_HOME)/data/twitter_single_opt_train_lower_ce0.05_word.txt"

//print("sizeof Node \(MemoryLayout<Node>.size) stride \(MemoryLayout<Node>.stride)")

// Single training test
var v = Vocabulary()
let alphabet = "abcdefghijklmnopqrstuvwxyz' "
v.addAllCharacters(valid: alphabet)

//var lines = try Utils.readLinesFrom(filename: SUBTITLE_CE_TRAIN)

var startMem = Utils.memoryInMB()
var startTime = ProcessInfo.processInfo.systemUptime
var lm = PPMLanguageModel(vocab: v, maxOrder: 9, capacityIncrease: 100_000)     // , capacityIncrease: 1_000_000
//var lm = PPMLanguageModel(vocab: v, maxOrder: 9, reserveCapacity: 30250645)  // Common all, after eval
//var skipped = lm.train(texts: lines)

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
//lm.shrink()

/*
 With shrink
 Training lines 1000001, chars 59513637, skipped chars 1699112, PPM nodes 30210611
 Train time: 95.1004, chars/second: 625797.7
 Memory increase in MB: 1157.04
 Estimated bytes per Node: 38.30
 Num nodes: 30210611
 Tree stats: TreeStats(nodes: 30210611, leaves: 12544232, singletons: 21427711, maxCount: 230132)
 (sumLogProb: -73251.31972974617, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.93035238472366)
 Eval time with update: 8.4826, chars/second: 18494.5
 Num nodes: 30250645
 End memory increase in MB: 1280.62
 
 Without shrink
 Training lines 1000001, chars 59513637, skipped chars 1699112, PPM nodes 30210611
 Train time: 98.6789, chars/second: 603103.9
 Memory increase in MB: 1026.59
 Estimated bytes per Node: 33.98
 Num nodes: 30210611
 Tree stats: TreeStats(nodes: 30210611, leaves: 12544232, singletons: 21427711, maxCount: 230132)
 (sumLogProb: -73251.31972974617, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.93035238472366)
 Eval time with update: 8.3777, chars/second: 18726.1
 Num nodes: 30250645
 End memory increase in MB: 1029.37
 
 With shrink and removeall
 Training lines 1000001, chars 59513637, skipped chars 1699112, PPM nodes 30210611
 Train time: 98.1389, chars/second: 606422.8
 Memory increase in MB: 1149.88
 Estimated bytes per Node: 38.06
 Num nodes: 30210611
 Tree stats: TreeStats(nodes: 30210611, leaves: 12544232, singletons: 21427711, maxCount: 230132)
 (sumLogProb: -73251.31972974617, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.93035238472366)
 Eval time with update: 8.5579, chars/second: 18331.7
 Num nodes: 30250645
 End memory increase in MB: 1273.53
 
 No capacityIncrease set
 Training lines 1000001, chars 59513637, skipped chars 1699112, PPM nodes 30210611
 Train time: 96.2627, chars/second: 618241.8
 Memory increase in MB: 1023.16
 Estimated bytes per Node: 33.87
 Num nodes: 30210611
 Tree stats: TreeStats(nodes: 30210611, leaves: 12544232, singletons: 21427711, maxCount: 230132)
 (sumLogProb: -73251.31972974617, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.93035238472366)
 Eval time with update: 8.3976, chars/second: 18681.9
 Num nodes: 30250645
 End memory increase in MB: 1025.92
 
 capacityIncrease = 100_000
 Training lines 1000001, chars 59513637, skipped chars 1699112, PPM nodes 30210611
 Train time: 109.7869, chars/second: 542083.3
 Memory increase in MB: 8337.18
 Estimated bytes per Node: 275.97
 Num nodes: 30210611
 Tree stats: TreeStats(nodes: 30210611, leaves: 12544232, singletons: 21427711, maxCount: 230132)
 (sumLogProb: -73251.31972974617, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.93035238472366)
 Eval time with update: 8.5209, chars/second: 18411.5
 Num nodes: 30250645
 End memory increase in MB: 8339.80
 
 Memory function was wrong???
 capacityIncrease = 100_000         468MB
 capacityIncrease = N/A             529MB
 capacityIncrease = N/A, shrink     185MB ???
 capacityIncrease = 1_000           469MB
 capacityIncrease = 1_000_000       467MB
 
 capacity increase
 END mem1 = 5972
 END mem2 = 470
 END mem3 = (5972.203, 65536.0)
 
 no capacity increase
 END mem1 = 998
 END mem2 = 532 (close to IDE report)
 END mem3 = (998.9219, 65536.0)
 
 exact reserved size
 END mem1 = 508
 END mem2 = 497 (close to IDE report)
 END mem3 = (508.5, 65536.0)
 
 exact reserved size with shrink
 END mem1 = 796
 END mem2 = 209
 END mem3 = (796.625, 65536.0)
 
 Shrink, old size 30210611, old capacity 30261216
 Shrink, new size 30210611, new capacity 30212064
 
 Shrink, old size 30210611, old capacity 33554400
 Shrink, new size 30210611, new capacity 33554400
 */

// Subtitle
// 1M lines, 1051MB, TreeStats(nodes: 25575597, leaves: 10799411, singletons: 17842420, maxCount: 230132), (sumLogProb: -72476.0074792203, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.8971957530228796), Train time: 116.5842, chars/second: 495855.8, Eval time with update: 10.4502, chars/second: 15012.4
// 2M lines, 1841MB, TreeStats(nodes: 38246982, leaves: 16846796, singletons: 25867424, maxCount: 460458), (sumLogProb: -71524.29776768915, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.857007859706569), Train time: 261.8971, chars/second: 441519.4, Eval time with update: 10.6628, chars/second: 14713.0
// 4M lines, 2202MB, TreeStats(nodes: 56338491, leaves: 25749999, singletons: 36893727, maxCount: 921773), (sumLogProb: -70662.71546195316, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.8211067630483138), Train time: 515.6805, chars/second: 448260.4, Eval time with update: 11.0392, chars/second: 14211.3
// All, 12.8M lines, 3155MB,TreeStats(nodes: 103986055, leaves: 50029653, singletons: 64107506, maxCount: 2947004), (sumLogProb: -69728.68492586802, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.782696285017178), Train time: 1869.5387, chars/second: 395222.9, Eval time with update: 12.0241, chars/second: 13047.3
// Reddit
// 1M lines, 993MB, Tree stats: TreeStats(nodes: 22706040, leaves: 9136793, singletons: 16482575, maxCount: 231541), (sumLogProb: -73243.06551434175, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.929997398162907), Train time: 91.0106, chars/second: 445481.4, Eval time with update: 10.6269, chars/second: 14762.7
// All, 134M lines, 9094MB, TreeStats(nodes: 400909648, leaves: 196625729, singletons: 255533560, maxCount: 31095959), (sumLogProb: -68772.77876910598, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.743927639908104), Train time: 19483.8951, chars/second: 279327.6, Eval time with update: 13.6609, chars/second: 11484.0
// Common
// All, 108M lines, 8794MB, TreeStats(nodes: 439705487, leaves: 214303977, singletons: 275045256, maxCount: 24954566), (sumLogProb: -68898.14844766873, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.7489813155727894), Train time: 23789.6164, chars/second: 271051.7, Eval time with update: 13.7759, chars/second: 11388.1
// Twitter
// All, 191M lines, 12411MB, Tree stats: TreeStats(nodes: 569071828, leaves: 284438708, singletons: 362009127, maxCount: 48255001), (sumLogProb: -67193.37051461659, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.6810515382613245), Training lines 191426321, chars 8463023184, skipped chars 265880867, PPM nodes 569071828, Eval time with update: 14.0788, chars/second: 11143.1 (missing train time)

// Common after eliminating struct, parallel arrays, 30% faster training. Higher memory???
// All, 108M lines, 11340MB, Tree stats: TreeStats(nodes: 439705487, leaves: 214303977, singletons: 275045256, maxCount: 24954566), (sumLogProb: -68898.14844766873, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.7489813155727894), Eval time with update: 11.2538, chars/second: 13940.4, Train time: 16183.9394, chars/second: 398433.0, Num nodes: 439711711
// Reserving exact capacity:
// 7478MB, Tree stats: TreeStats(nodes: 439705487, leaves: 214303977, singletons: 275045256, maxCount: 24954566), (sumLogProb: -68898.14844766873, tokensGood: 156882, tokensSkipped: 0, perplexity: 2.7489813155727894), Eval time with update: 11.1970, chars/second: 14011.1, Train time: 16120.7659, chars/second: 399994.3, Num nodes: 439711711
// All the above memory numbers are before changing memory function.

var endTime = ProcessInfo.processInfo.systemUptime
var endMem = Utils.memoryInMB()

//var trainChars = Utils.countCharacters(texts: lines)
//print("Training lines \(lines.count), chars \(trainChars), skipped chars \(skipped), PPM nodes \(lm.numNodes)")
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

print("END mem1 = \(Utils.memoryUsed() / 1024 / 1024)")
print("END mem2 = \(Utils.memoryInMB())")
print("END mem3 = \(Utils.getMemoryUsedAndDeviceTotalInMegabytes())")


/*

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
var startMem = Utils.memoryUsed()
var startTime = ProcessInfo.processInfo.systemUptime
lm = PPMLanguageModel(vocab: v, maxOrder: 8)
skipped = lm.train(texts: lines)
var endTime = ProcessInfo.processInfo.systemUptime
var endMem = Utils.memoryUsed()
var trainChars = Utils.countCharacters(texts: lines)
print("Training lines \(lines.count), chars \(trainChars), skipped chars \(skipped), PPM nodes \(lm.numNodes)")
var elapsed = endTime - startTime
print("Train time: \(String(format: "%.4f", elapsed))" +
      ", chars/second: \(String(format: "%.1f", (Double(trainChars) / elapsed)))")
var memMB = Double(endMem - startMem) / 1000000.0
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
startMem = Utils.memoryUsed()
startTime = ProcessInfo.processInfo.systemUptime

// Automatically grow the tree
//lm = PPMLanguageModel(vocab: v, maxOrder: 9)
// Size of the training tree
//lm = PPMLanguageModel(vocab: v, maxOrder: 9, reserveCapacity: 2234348)
// Size of the training + eval tree
lm = PPMLanguageModel(vocab: v, maxOrder: 9, reserveCapacity: 2380791)

skipped = lm.train(texts: lines)
endTime = ProcessInfo.processInfo.systemUptime
endMem = Utils.memoryUsed()
trainChars = Utils.countCharacters(texts: lines)
print("Training lines \(lines.count), chars \(trainChars), skipped chars \(skipped), PPM nodes \(lm.numNodes)")
elapsed = endTime - startTime
print("Train time: \(String(format: "%.4f", elapsed))" +
      ", chars/second: \(String(format: "%.1f", (Double(trainChars) / elapsed)))")
memMB = Double(endMem - startMem) / 1000000.0
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

*/

// *** Test 20, using pointers
// Training lines 46427, chars 1752271, skipped chars 0, PPM nodes 2234348
// Train time: 1.5661, chars/second: 1118905.3
// Memory increase in MB: 143.11
// Estimated bytes per Node: 64.05
// Num nodes: 2234348
// Tree stats: (nodes: 2234348, leaves: 705995, singletons: 1742596)
// (sumLogProb: -78138.42692423137, tokensGood: 156882, tokensSkipped: 0, perplexity: 3.1482653779731993)
// Eval time with update: 5.2502, chars/second: 29881.1

// *** Test 20, using array indexes
// Training lines 46427, chars 1752271, skipped chars 0, PPM nodes 2234348
// Train time: 4.4110, chars/second: 397246.7
// Memory increase in MB: 118.33
// Estimated bytes per Node: 52.96
// Num nodes: 2234348
// Tree stats: (nodes: 2234348, leaves: 705995, singletons: 1742596)
// (sumLogProb: -78138.42692423137, tokensGood: 156882, tokensSkipped: 0, perplexity: 3.1482653779731993)
// Eval time with update: 10.8456, chars/second: 14465.1
// On AC power:                 9.0055, 9.1733, 9.0110
// After some cast fixes:       7.8796, 8.2732
// Change to get/set scheme:    8.1169, 8.1877

// *** Test 20, exact capacity for train model (but not eval)
// Training lines 46427, chars 1752271, skipped chars 0, PPM nodes 2234348
// Train time: 3.9304, chars/second: 445827.4
// Memory increase in MB: 106.79
// Estimated bytes per Node: 47.80
// Num nodes: 2234348
// Tree stats: (nodes: 2234348, leaves: 705995, singletons: 1742596)
// (sumLogProb: -78138.42692423137, tokensGood: 156882, tokensSkipped: 0, perplexity: 3.1482653779731993)
// Eval time with update: 8.9263, chars/second: 17575.3
/*

 *** Test 20, exact size of train + eval
Training lines 46427, chars 1752271, skipped chars 0, PPM nodes 2234348
Train time: 3.9790, chars/second: 440382.0
Memory increase in MB: 88.90
Estimated bytes per Node: 39.79
Num nodes: 2234348
Tree stats: (nodes: 2234348, leaves: 705995, singletons: 1742596)
(sumLogProb: -78138.42692423137, tokensGood: 156882, tokensSkipped: 0, perplexity: 3.1482653779731993)
Eval time with update: 7.9799, chars/second: 19659.7
Num nodes: 2380791
 
 *** Test 20, exact size train + eval, change Node, PPMLanguageModel, Vocabulary to structs
 Training lines 46427, chars 1752271, skipped chars 0, PPM nodes 2234348
 Train time: 2.6345, chars/second: 665132.1
 Memory increase in MB: 71.50
 Estimated bytes per Node: 32.00
 Num nodes: 2234348
 Tree stats: (nodes: 2234348, leaves: 705995, singletons: 1742596)
 (sumLogProb: -78138.42692423137, tokensGood: 156882, tokensSkipped: 0, perplexity: 3.1482653779731993)
 Eval time with update: 6.2513, chars/second: 25096.1
 Num nodes: 2380791
 
 *** Test 20, count to UInt32
 24 bytes per node
 Training lines 46427, chars 1752271, skipped chars 0, PPM nodes 2234348
 Train time: 2.6846, chars/second: 652719.9
 Memory increase in MB: 53.62
 Estimated bytes per Node: 24.00
 Num nodes: 2234348
 Tree stats: (nodes: 2234348, leaves: 705995, singletons: 1742596)
 (sumLogProb: -78138.42692423137, tokensGood: 156882, tokensSkipped: 0, perplexity: 3.1482653779731993)
 Eval time with update: 6.6608, chars/second: 23553.1
 Num nodes: 2380791
 
 *** Test 20, symbol to UInt8
 17 bytes per Node (probably aligned longer?)
 Training lines 46427, chars 1752271, skipped chars 0, PPM nodes 2234348
 Train time: 3.4954, chars/second: 501309.5
 Memory increase in MB: 44.70
 Estimated bytes per Node: 20.00
 Num nodes: 2234348
 Tree stats: (nodes: 2234348, leaves: 705995, singletons: 1742596)
 (sumLogProb: -78138.42692423137, tokensGood: 156882, tokensSkipped: 0, perplexity: 3.1482653779731993)
 Eval time with update: 8.1363, chars/second: 19281.7
 Num nodes: 2380791

 *** Test 20, symbol and count to UInt16
 sizeof Node 16 stride 16
 Training lines 46427, chars 1752271, skipped chars 0, PPM nodes 2234348
 Train time: 3.3107, chars/second: 529269.8
 Memory increase in MB: 35.75
 Estimated bytes per Node: 16.00
 Num nodes: 2234348
 Tree stats: TreeStats(nodes: 2234348, leaves: 705995, singletons: 1742596, maxCount: 11616)
 (sumLogProb: -78138.42692423137, tokensGood: 156882, tokensSkipped: 0, perplexity: 3.1482653779731993)
 Eval time with update: 8.0452, chars/second: 19500.0
 Num nodes: 2380791
 
 *** Test 20, count UInt32, symbol UInt16
 Training lines 46427, chars 1752271, skipped chars 0, PPM nodes 2234348
 Train time: 3.3349, chars/second: 525434.0
 Memory increase in MB: 44.70
 Estimated bytes per Node: 20.00
 Num nodes: 2234348
 Tree stats: TreeStats(nodes: 2234348, leaves: 705995, singletons: 1742596, maxCount: 11616)
 (sumLogProb: -78138.42692423137, tokensGood: 156882, tokensSkipped: 0, perplexity: 3.1482653779731993)
 Eval time with update: 8.0003, chars/second: 19609.4
 Num nodes: 2380791
 
 *** Test 20, parallel symbol UInt8 array
 Training lines 46427, chars 1752271, skipped chars 0, PPM nodes 2234348
 Train time: 2.9948, chars/second: 585110.5
 Memory increase in MB: 35.75
 Estimated bytes per Node: 16.00
 Num nodes: 2234348
 Tree stats: TreeStats(nodes: 2234348, leaves: 705995, singletons: 1742596, maxCount: 11616)
 (sumLogProb: -78138.42692423137, tokensGood: 156882, tokensSkipped: 0, perplexity: 3.1482653779731993)
 Eval time with update: 7.5531, chars/second: 20770.5
 Num nodes: 2380791
 
 *** Test 20, parallel symbol + count array
 Training lines 46427, chars 1752271, skipped chars 0, PPM nodes 2234348
 Train time: 2.6238, chars/second: 667835.3
 Memory increase in MB: 35.77
 Estimated bytes per Node: 16.01
 Num nodes: 2234348
 Tree stats: TreeStats(nodes: 2234348, leaves: 705995, singletons: 1742596, maxCount: 11616)
 (sumLogProb: -78138.42692423137, tokensGood: 156882, tokensSkipped: 0, perplexity: 3.1482653779731993)
 Eval time with update: 6.6238, chars/second: 23684.6
 Num nodes: 2380791
 
 *** Test 20, parallel symbol, count, backoff
 Training lines 46427, chars 1752271, skipped chars 0, PPM nodes 2234348
 Train time: 2.7519, chars/second: 636743.5
 Memory increase in MB: 38.04
 Estimated bytes per Node: 17.03
 Num nodes: 2234348
 Tree stats: TreeStats(nodes: 2234348, leaves: 705995, singletons: 1742596, maxCount: 11616)
 (sumLogProb: -78138.42692423137, tokensGood: 156882, tokensSkipped: 0, perplexity: 3.1482653779731993)
 Eval time with update: 6.9306, chars/second: 22636.1
 Num nodes: 2380791
 
 *** Test 20, parallel symbol, count, backoff, next
 Training lines 46427, chars 1752271, skipped chars 0, PPM nodes 2234348
 Train time: 2.8072, chars/second: 624198.3
 Memory increase in MB: 38.04
 Estimated bytes per Node: 17.03
 Num nodes: 2234348
 Tree stats: TreeStats(nodes: 2234348, leaves: 705995, singletons: 1742596, maxCount: 11616)
 (sumLogProb: -78138.42692423137, tokensGood: 156882, tokensSkipped: 0, perplexity: 3.1482653779731993)
 Eval time with update: 6.4615, chars/second: 24279.5
 Num nodes: 2380791
 
 *** Test 20, all parallel
 Training lines 46427, chars 1752271, skipped chars 0, PPM nodes 2234348
 Train time: 2.6498, chars/second: 661296.2
 Memory increase in MB: 35.80
 Estimated bytes per Node: 16.02
 Num nodes: 2234348
 Tree stats: TreeStats(nodes: 2234348, leaves: 705995, singletons: 1742596, maxCount: 11616)
 (sumLogProb: -78138.42692423137, tokensGood: 156882, tokensSkipped: 0, perplexity: 3.1482653779731993)
 Eval time with update: 6.5316, chars/second: 24018.9
 Num nodes: 2380791

*/

// Results training on full daily dialog training set.
// Evaluating on AAC dev/test set from https://imagineville.org/software/lm/feb21_dasher/
// ppm-order log prob (adaptive)   log prob (non-adaptive)  equiv WB n-gram   interpolated WB
// 7         -78621.98141020491    -80789.82665258418
// 8         -78144.00404470577    -80479.03825578607       -87122.5216
// 9         -78138.42692423137    -80593.34598803819       -88159.6325
// 10        -78301.01233848355    -80828.55373719665       -89112.5575
// 11        -78567.49759372474    -81151.09168887722       -89940.0984       -104849.1723
// 12        -78794.46570954652    -81415.94914405495

print("*** TESTS COMPLETED")
/*
// Test that the probability of each character is frequency in sentences.
print("*** Test \(test)"); test += 1
let sentencesAll = sentences[0] + sentences[1] + sentences[2]
let totalChars = sentencesAll.count
let charCounts = sentencesAll.charactersToCount()
for ch in probsDict
{
    let count = charCounts[ch.key] ?? 0
    let prob = Double(count) / Double(totalChars)
    
    print("\(ch.key) = \(prob) vs \(ch.value) = \(abs(ch.value - prob))")
}
print(charCounts)
*/
