// Program that tests out the PPMLM class.
// These tests follow those in https://github.com/google-research/google-research/blob/master/jslm/example.js

import Foundation

// TODO: Change to match your system, used to find training text files.
let PPMLM_HOME = "/Users/vertanen/PPMLM/"

// Actual filenames some of the test depend on
let DAILY_DIALOG_TRAIN = "\(PPMLM_HOME)/data/daily_train_10k.txt"

// Create a small vocabulary.
var v = Vocabulary()
let aSymbol = v.add(token: "a")
let bSymbol = v.add(token: "b")

// Build the PPM language model trie and update the counts.
let maxOrder = 5
var lm = PPMLanguageModel(vocab: v, maxOrder: maxOrder)
var c = lm.createContext()
lm.addSymbolAndUpdate(context: c, symbol: aSymbol)
lm.addSymbolAndUpdate(context: c, symbol: bSymbol)
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
lm.addSymbolAndUpdate(context: c, symbol: aSymbol)
probs = lm.getProbs(context: c)
print(probs)
assert(probs[1] > 0 && probs[1] > probs[2], "Probability for \"a\" should be more likely")

// Enter "b" and update the model. At this point both symbols should become
// equally likely again.
print("*** Test \(test)"); test += 1
lm.addSymbolAndUpdate(context: c, symbol: bSymbol)
probs = lm.getProbs(context: c)
print(probs)
assert(probs[1] > 0 && probs[1] == probs[2], "Probabilities for both symbols should be the same")

// Enter "b" and update the model. Current context "abb". Since we've seen
// "ab" and "abb" by now, the "b" becomes more likely.
print("*** Test \(test)"); test += 1
lm.addSymbolAndUpdate(context: c, symbol: bSymbol)
probs = lm.getProbs(context: c)
print(probs)
assert(probs[1] > 0 && probs[1] < probs[2], "Probability for \"b\" should be more likely")

print("Final count trie:")
lm.printTree()

//print(malloc_size(Unmanaged.passRetained(c).toOpaque()))
//print(class_getInstanceSize(type(of: c)))

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
var lines = [String]()
let fileURL = URL(fileURLWithPath: DAILY_DIALOG_TRAIN)
let trainData = try Data(contentsOf: fileURL)
if let trainLines = String(data: trainData, encoding: .utf8)
{
    lines = trainLines.lines
}
let startMem = Utils.memoryUsed()
let startTime = ProcessInfo.processInfo.systemUptime
lm = PPMLanguageModel(vocab: v, maxOrder: 8)
skipped = lm.train(texts: lines)
let endTime = ProcessInfo.processInfo.systemUptime
let endMem = Utils.memoryUsed()
var trainChars = 0
for line in lines
{
    trainChars += line.count
}
print("Training lines \(lines.count), chars \(trainChars), skipped chars \(skipped), PPM nodes \(lm.numNodes)")
let elapsed = endTime - startTime
print("Train time: \(String(format: "%.4f", elapsed))" +
      ", chars/second: \(String(format: "%.1f", (Double(trainChars) / elapsed)))")
let memMB = Double(endMem - startMem) / 1000000.0
print("Memory increase in MB: \(String(format: "%.2f", memMB))")
let bytesPerNode = Double(endMem - startMem) / Double(lm.numNodes)
print("Estimated bytes per Node: \(String(format: "%.2f", bytesPerNode))")

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
