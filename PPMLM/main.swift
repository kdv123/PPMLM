// Program that tests out the PPMLM class.
// These tests follow those in https://github.com/google-research/google-research/blob/master/jslm/example.js

import Foundation

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

// Test that the probability of each character is frequency in sentences.
print("*** Test \(test)"); test += 1
c = lm.createContext()
probs = lm.getProbs(context: c)
print(probs)

let totalChars = sentences[0].count + sentences[1].count + sentences[2].count
